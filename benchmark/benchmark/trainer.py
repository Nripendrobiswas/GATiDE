"""
Training & Evaluation Engine
============================
- MSE loss, AdamW, schedulers (CosineAnnealing, StepLR)
- EarlyStopping on val_loss
- Throughput tracking: training time per epoch (seconds), GPU peak memory (MB)
- Clean PyTorch loop (no Lightning required, but compatible)
- Mixed precision optional

All metrics on inverse-scaled predictions (original scale) via scaler.
"""
from __future__ import annotations

import time
import copy
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark.utils.metrics import mse, mae


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return True  # improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

    def state_dict(self):
        return {"best": self.best, "counter": self.counter}


def _get_scheduler(optimizer, name: str, params: dict, n_epochs: int):
    name = (name or "none").lower()
    if name == "cosine":
        T_max = params.get("T_max", n_epochs)
        eta_min = params.get("eta_min", 0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif name == "step":
        step_size = params.get("step_size", 30)
        gamma = params.get("gamma", 0.5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name in ("none", "null", ""):
        return None
    else:
        raise ValueError(f"unknown scheduler {name}")


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    scaler,  # StandardScaler instance for inverse transform
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer_name: str = "adamw",
    scheduler_name: str = "cosine",
    scheduler_params: Optional[dict] = None,
    patience: int = 10,
    min_delta: float = 1e-4,
    grad_clip: Optional[float] = 1.0,
    device: str = "auto",
    amp: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Train with MSE loss, AdamW, early stopping.

    Returns dict with:
      model (best state), history, train_time_per_epoch, peak_memory_mb,
      test_mse, test_mae, val_mse, val_mae, epochs_run, best_val_loss
    """
    scheduler_params = scheduler_params or {}

    # Device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device)

    # Naive baseline has no trainable params – skip training
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    is_naive = n_params == 0

    criterion = nn.MSELoss()
    if not is_naive:
        if optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"unknown optimizer {optimizer_name}")
        scheduler = _get_scheduler(optimizer, scheduler_name, scheduler_params, n_epochs)
        early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)
    else:
        optimizer = None
        scheduler = None
        early_stopper = None

    # GPU memory tracking
    peak_mem_mb = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    history: List[Dict] = []
    best_state = None
    best_val_loss = float("inf")
    epochs_run = 0
    train_seconds_total = 0.0

    scaler_amp = torch.cuda.amp.GradScaler() if (amp and device.type == "cuda") else None

    # Early exit for naive
    if is_naive:
        # Direct evaluation without training
        val_mse, val_mae, _, _ = evaluate(model, val_loader, scaler, device)
        test_mse, test_mae, preds, trues = evaluate(model, test_loader, scaler, device, return_arrays=True)
        return {
            "best_state": None,
            "history": [],
            "train_time_per_epoch": 0.0,
            "peak_memory_mb": 0.0,
            "epochs_run": 0,
            "best_val_loss": val_mse,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "preds": preds,
            "trues": trues,
            "n_params": 0,
        }

    # Training loop
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)  # (B, L, C)
            yb = yb.to(device)  # (B, H, C)
            optimizer.zero_grad()
            if scaler_amp is not None:
                with torch.cuda.amp.autocast():
                    pred = model(xb)
                    loss = criterion(pred, yb)
                scaler_amp.scale(loss).backward()
                if grad_clip:
                    scaler_amp.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            train_losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        epoch_time = time.time() - t0
        train_seconds_total += epoch_time
        epochs_run = epoch

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        improved = False
        if early_stopper is not None:
            is_best = val_loss < best_val_loss - min_delta
            if is_best:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                improved = True
            early_stopper.step(val_loss)
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                improved = True

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"] if optimizer else 0,
            "epoch_time": epoch_time,
            "improved": improved,
        })

        if verbose and (epoch % 10 == 0 or epoch == 1 or early_stopper and early_stopper.should_stop):
            tqdm.write(f"  epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | lr {optimizer.param_groups[0]['lr']:.2e} | {epoch_time:.1f}s {'*' if improved else ''}")

        if early_stopper and early_stopper.should_stop:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (patience {patience})")
            break

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # GPU peak memory
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        torch.cuda.empty_cache()
    else:
        peak_mem_mb = 0.0

    # Final evaluation on inverse-scaled predictions
    val_mse, val_mae, _, _ = evaluate(model, val_loader, scaler, device)
    test_mse, test_mae, preds, trues = evaluate(model, test_loader, scaler, device, return_arrays=True)

    per_epoch = train_seconds_total / max(epochs_run, 1)

    return {
        "best_state": best_state,
        "history": history,
        "train_time_per_epoch": per_epoch,
        "peak_memory_mb": float(peak_mem_mb),
        "epochs_run": epochs_run,
        "best_val_loss": best_val_loss,
        "val_mse": val_mse,
        "val_mae": val_mae,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "preds": preds,  # (N, H, C) in original scale, may be None if large
        "trues": trues,
        "n_params": n_params,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    scaler,
    device: torch.device | str = "cpu",
    return_arrays: bool = False,
) -> Tuple[float, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Evaluate on loader, inverse-scale, compute MSE/MAE in original scale.

    Returns (mse, mae, preds_array, trues_array) where arrays are (N, H, C) if requested.
    """
    if isinstance(device, str):
        device = torch.device(device)
    model.eval()
    preds_list = []
    trues_list = []
    for xb, yb in loader:
        xb = xb.to(device)
        pred_scaled = model(xb)  # (B, H, C) scaled
        # Move to cpu numpy
        pred_np = pred_scaled.detach().cpu().numpy()
        true_np = yb.numpy()  # already cpu
        # Inverse transform per (N, H, C)
        # scaler expects (H, C) or (N*H, C) -> we do batch-wise
        B, H, C = pred_np.shape
        # Reshape to (-1, C), inverse, reshape back
        pred_flat = pred_np.reshape(-1, C)
        true_flat = true_np.reshape(-1, C)
        # scaler.inverse_transform expects 2-D
        pred_inv = scaler.inverse_transform(pred_flat).reshape(B, H, C)
        true_inv = scaler.inverse_transform(true_flat).reshape(B, H, C)
        preds_list.append(pred_inv)
        trues_list.append(true_inv)

    if not preds_list:
        return float("nan"), float("nan"), None, None

    preds = np.concatenate(preds_list, axis=0)  # (N, H, C)
    trues = np.concatenate(trues_list, axis=0)
    m = mse(trues, preds)
    a = mae(trues, preds)
    if return_arrays:
        return m, a, preds, trues
    return m, a, None, None

"""
Training & Evaluation Engine
============================
- MSE loss, AdamW/Adam, schedulers (CosineAnnealing, StepLR) – TiDE paper uses Adam with MSE,
  requirement asks AdamW; both supported via optimizer_name (TiDE §5.1: "we optimize using the
  default settings of the Adam optimizer")
- EarlyStopping on val_loss (normalized scale, as training loss)
- Throughput tracking: training time per epoch (s), GPU peak memory (MB) via
  torch.cuda.max_memory_allocated (TiDE Fig.2 reports training time per epoch and inference
  time per batch vs L on Electricity, batch 8, T4 GPU – we report both)
- Metrics on BOTH scales:
    * normalized (standardized) MSE/MAE – TiDE Table 2: "All metrics are reported on standard
      normalized datasets (using mean and std in training period)"
    * inverse-scaled (original) MSE/MAE – requirement prompt: "Compute metrics on inverse-scaled
      predictions"
  Both are returned; normalized is primary for paper comparability, inverse is for requirement.
- Clean PyTorch loop (no Lightning required, but compatible)
- Mixed precision optional
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
    Train with MSE loss, AdamW/Adam, early stopping.

    Returns dict with:
      model (best state), history, train_time_per_epoch, peak_memory_mb,
      inference_time_ms_per_batch, test_mse (original), test_mae (original),
      test_mse_norm/test_mae_norm (normalized, TiDE Table 2), val equivalents, epochs_run
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
        # Direct evaluation without training – report both scales
        val_mse, val_mae, val_mse_n, val_mae_n, _, _ = evaluate(model, val_loader, scaler, device)
        test_mse, test_mae, test_mse_n, test_mae_n, preds, trues = evaluate(model, test_loader, scaler, device, return_arrays=True)
        # Inference timing for naive (per batch)
        infer_ms = measure_inference_time(model, test_loader, device)
        return {
            "best_state": None,
            "history": [],
            "train_time_per_epoch": 0.0,
            "peak_memory_mb": 0.0,
            "inference_ms_per_batch": infer_ms,
            "epochs_run": 0,
            "best_val_loss": val_mse_n if not np.isnan(val_mse_n) else val_mse,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "val_mse_norm": val_mse_n,
            "val_mae_norm": val_mae_n,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "test_mse_norm": test_mse_n,
            "test_mae_norm": test_mae_n,
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

    # Final evaluation – both normalized (TiDE Table 2) and inverse-scaled (requirement)
    val_mse, val_mae, val_mse_n, val_mae_n, _, _ = evaluate(model, val_loader, scaler, device)
    test_mse, test_mae, test_mse_n, test_mae_n, preds, trues = evaluate(model, test_loader, scaler, device, return_arrays=True)
    infer_ms = measure_inference_time(model, test_loader, device)

    per_epoch = train_seconds_total / max(epochs_run, 1)

    return {
        "best_state": best_state,
        "history": history,
        "train_time_per_epoch": per_epoch,
        "peak_memory_mb": float(peak_mem_mb),
        "inference_ms_per_batch": infer_ms,
        "epochs_run": epochs_run,
        "best_val_loss": best_val_loss,
        "val_mse": val_mse,
        "val_mae": val_mae,
        "val_mse_norm": val_mse_n,
        "val_mae_norm": val_mae_n,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_mse_norm": test_mse_n,
        "test_mae_norm": test_mae_n,
        "preds": preds,  # (N, H, C) in original scale
        "trues": trues,
        "n_params": n_params,
    }


@torch.no_grad()
def measure_inference_time(model: nn.Module, loader: DataLoader, device: torch.device | str = "cpu", warmup: int = 3) -> float:
    """Measure average inference time per batch in ms (TiDE Fig.2: inference time for one batch).
    Runs warmup + 10 timed batches on device, synchronized if CUDA.
    """
    if isinstance(device, str):
        device = torch.device(device)
    model.eval()
    # Warmup
    for i, (xb, _) in enumerate(loader):
        if i >= warmup:
            break
        xb = xb.to(device)
        _ = model(xb)
        if device.type == "cuda":
            torch.cuda.synchronize()
    # Timed
    times = []
    for i, (xb, _) in enumerate(loader):
        if i >= 10:
            break
        xb = xb.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        _ = model(xb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000.0)
    return float(np.mean(times)) if times else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    scaler,
    device: torch.device | str = "cpu",
    return_arrays: bool = False,
) -> Tuple[float, float, float, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Evaluate on loader, compute BOTH normalized (TiDE Table 2) and inverse-scaled metrics.

    Returns (mse_orig, mae_orig, mse_norm, mae_norm, preds_array, trues_array)
    where mse_norm/mae_norm are on standardized scale (training-period mean/std),
    directly comparable to TiDE paper Table 2. mse_orig/mae_orig are on original scale
    as per requirement prompt. Arrays are (N, H, C) in original scale if requested.
    """
    if isinstance(device, str):
        device = torch.device(device)
    model.eval()
    preds_inv_list = []
    trues_inv_list = []
    preds_norm_list = []
    trues_norm_list = []
    for xb, yb in loader:
        xb = xb.to(device)
        pred_scaled = model(xb)  # (B, H, C) scaled (normalized)
        pred_np = pred_scaled.detach().cpu().numpy()
        true_np = yb.numpy()  # already cpu, normalized
        preds_norm_list.append(pred_np)
        trues_norm_list.append(true_np)
        # Inverse for original-scale metrics (requirement)
        B, H, C = pred_np.shape
        pred_flat = pred_np.reshape(-1, C)
        true_flat = true_np.reshape(-1, C)
        pred_inv = scaler.inverse_transform(pred_flat).reshape(B, H, C)
        true_inv = scaler.inverse_transform(true_flat).reshape(B, H, C)
        preds_inv_list.append(pred_inv)
        trues_inv_list.append(true_inv)

    if not preds_inv_list:
        return float("nan"), float("nan"), float("nan"), float("nan"), None, None

    preds_inv = np.concatenate(preds_inv_list, axis=0)  # (N, H, C) original
    trues_inv = np.concatenate(trues_inv_list, axis=0)
    preds_norm = np.concatenate(preds_norm_list, axis=0)
    trues_norm = np.concatenate(trues_norm_list, axis=0)
    m_orig = mse(trues_inv, preds_inv)
    a_orig = mae(trues_inv, preds_inv)
    m_norm = mse(trues_norm, preds_norm)
    a_norm = mae(trues_norm, preds_norm)
    if return_arrays:
        return m_orig, a_orig, m_norm, a_norm, preds_inv, trues_inv
    return m_orig, a_orig, m_norm, a_norm, None, None

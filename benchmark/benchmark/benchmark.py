"""
Core benchmark orchestration
============================
Loops over datasets x horizons x models, trains, evaluates, aggregates,
saves predictions, and writes benchmark_results.csv.

This is imported by run_benchmark.py CLI.

Protocol matches requirements:
  - Lookback L=720, horizons H in {96,192,336,720}
  - Sequential split 70/10/20, standardized with train-only params
  - MSE loss, AdamW, Cosine/StepLR, EarlyStopping
  - Metrics on inverse-scaled test set: MSE, MAE
  - Throughput: training time per epoch (s), GPU peak memory (MB)
"""
from __future__ import annotations

import os
import time
import json
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import torch

from benchmark.datasets import load_and_split, make_loaders, discover_datasets
from benchmark.models import get_model
from benchmark.trainer import train_one_model


def run_benchmark(
    csv_dir: str,
    datasets: Optional[List[str]] = None,
    horizons: List[int] = [96, 192, 336, 720],
    models: List[str] = ["gatide", "tide", "dlinear", "patchtst", "naive"],
    lookback: int = 720,
    batch_size: int = 32,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer: str = "adamw",
    scheduler: str = "cosine",
    scheduler_params: Optional[dict] = None,
    patience: int = 10,
    min_delta: float = 1e-4,
    grad_clip: float = 1.0,
    device: str = "auto",
    amp: bool = False,
    seed: int = 42,
    save_dir: str = "./benchmark_outputs",
    save_predictions: bool = True,
    model_kwargs: Optional[Dict[str, dict]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run full benchmark grid.

    Args:
        csv_dir: path to folder containing *.csv datasets
        datasets: list of dataset stems; if None, auto-discover
        horizons: list of H values
        models: list of model keys (see benchmark.models.MODEL_REGISTRY)
        lookback: L
        ... training hyperparams
        save_dir: where to save benchmark_results.csv and .npy predictions
        model_kwargs: dict mapping model name -> kwargs override

    Returns:
        DataFrame with one row per (dataset, horizon, model)
    """
    scheduler_params = scheduler_params or {}
    model_kwargs = model_kwargs or {}

    if datasets is None or len(datasets) == 0:
        datasets = discover_datasets(csv_dir)
        print(f"[benchmark] Auto-discovered datasets: {datasets}")

    os.makedirs(save_dir, exist_ok=True)
    # Fix seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    results: List[Dict] = []
    total_runs = len(datasets) * len(horizons) * len(models)
    print(f"[benchmark] Planned runs: {total_runs} = {len(datasets)} datasets x {len(horizons)} horizons x {len(models)} models")
    print(f"[benchmark] Lookback L={lookback} | Device={device} | Epochs={n_epochs} | Batch={batch_size}")
    print("-" * 80)

    run_idx = 0
    for dataset in datasets:
        for horizon in horizons:
            # Load split once per dataset-horizon (horizon only affects windowing)
            try:
                split = load_and_split(csv_dir, dataset, lookback=lookback, horizon=horizon)
            except Exception as e:
                print(f"[skip] {dataset} H={horizon} – load failed: {e}")
                # Record failed row
                for model_name in models:
                    results.append({
                        "dataset": dataset,
                        "horizon": horizon,
                        "model": model_name,
                        "lookback": lookback,
                        "mse": np.nan,
                        "mae": np.nan,
                        "val_mse": np.nan,
                        "val_mae": np.nan,
                        "train_time_per_epoch_s": np.nan,
                        "peak_memory_mb": np.nan,
                        "epochs_run": 0,
                        "n_params": 0,
                        "batch_size": batch_size,
                        "lr": lr,
                        "status": f"load_failed: {e}",
                    })
                continue

            if verbose:
                print(f"\n[dataset] {dataset} | T={split.T_total} (train {split.T_train}, val {split.T_val}, test {split.T_test}) "
                      f"| C={split.n_features} | L={lookback} H={horizon}")

            # Check if windows are viable
            try:
                train_loader, val_loader, test_loader = make_loaders(
                    split, lookback=lookback, horizon=horizon,
                    batch_size=batch_size, num_workers=0
                )
            except ValueError as e:
                print(f"[skip] {dataset} H={horizon} – window failed: {e}")
                for model_name in models:
                    results.append({
                        "dataset": dataset,
                        "horizon": horizon,
                        "model": model_name,
                        "lookback": lookback,
                        "mse": np.nan,
                        "mae": np.nan,
                        "val_mse": np.nan,
                        "val_mae": np.nan,
                        "train_time_per_epoch_s": np.nan,
                        "peak_memory_mb": np.nan,
                        "epochs_run": 0,
                        "n_params": 0,
                        "batch_size": batch_size,
                        "lr": lr,
                        "status": f"window_failed: {e}",
                    })
                continue

            if verbose:
                print(f"  windows: train {len(train_loader.dataset)} | val {len(val_loader.dataset)} | test {len(test_loader.dataset)}")

            for model_name in models:
                run_idx += 1
                tag = f"[{run_idx}/{total_runs}] {dataset} H={horizon} model={model_name}"
                print(f"\n{tag}", flush=True)

                # Build model
                try:
                    ModelClass = get_model(model_name)
                    # Merge kwargs
                    kwargs = dict(
                        num_features=split.n_features,
                        lookback=lookback,
                        horizon=horizon,
                    )
                    # Apply per-model overrides
                    if model_name in model_kwargs:
                        kwargs.update(model_kwargs[model_name])
                    # Also pass common hidden etc if provided at top level? keep simple
                    model = ModelClass(**kwargs)
                    n_params = sum(p.numel() for p in model.parameters())
                    print(f"  model: {ModelClass.__name__} | params: {n_params:,} | kwargs: {kwargs}")
                except Exception as e:
                    print(f"  FAILED model build: {e}")
                    results.append({
                        "dataset": dataset, "horizon": horizon, "model": model_name,
                        "lookback": lookback,
                        "mse": np.nan, "mae": np.nan, "val_mse": np.nan, "val_mae": np.nan,
                        "train_time_per_epoch_s": np.nan, "peak_memory_mb": np.nan,
                        "epochs_run": 0, "n_params": 0, "batch_size": batch_size, "lr": lr,
                        "status": f"build_failed: {e}",
                    })
                    continue

                # Train & evaluate
                t_start = time.time()
                try:
                    out = train_one_model(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        scaler=split.scaler,
                        n_epochs=n_epochs,
                        lr=lr,
                        weight_decay=weight_decay,
                        optimizer_name=optimizer,
                        scheduler_name=scheduler,
                        scheduler_params=scheduler_params,
                        patience=patience,
                        min_delta=min_delta,
                        grad_clip=grad_clip,
                        device=device,
                        amp=amp,
                        verbose=False,  # inner verbose off, outer will print summary
                    )
                    status = "ok"
                except Exception as e:
                    import traceback
                    print(f"  FAILED training: {e}")
                    traceback.print_exc()
                    out = None
                    status = f"train_failed: {e}"

                elapsed = time.time() - t_start

                if out is not None:
                    # Save predictions
                    if save_predictions and out.get("preds") is not None:
                        pred_dir = os.path.join(save_dir, "predictions")
                        os.makedirs(pred_dir, exist_ok=True)
                        # Filenames: {dataset}_H{H}_{model}_pred.npy / _true.npy
                        pred_path = os.path.join(pred_dir, f"{dataset}_H{horizon}_{model_name}_pred.npy")
                        true_path = os.path.join(pred_dir, f"{dataset}_H{horizon}_{model_name}_true.npy")
                        try:
                            np.save(pred_path, out["preds"])
                            np.save(true_path, out["trues"])
                            print(f"  saved predictions: {pred_path} shape {out['preds'].shape}")
                        except Exception as e:
                            print(f"  warn: failed to save npy: {e}")

                    row = {
                        "dataset": dataset,
                        "horizon": horizon,
                        "model": model_name,
                        "lookback": lookback,
                        "mse": out["test_mse"],
                        "mae": out["test_mae"],
                        "val_mse": out["val_mse"],
                        "val_mae": out["val_mae"],
                        "train_time_per_epoch_s": out["train_time_per_epoch"],
                        "peak_memory_mb": out["peak_memory_mb"],
                        "epochs_run": out["epochs_run"],
                        "n_params": out["n_params"],
                        "batch_size": batch_size,
                        "lr": lr,
                        "status": status,
                    }
                    print(f"  -> MSE {out['test_mse']:.4f} | MAE {out['test_mae']:.4f} | "
                          f"val MSE {out['val_mse']:.4f} | {out['train_time_per_epoch']:.2f}s/epoch | "
                          f"peak {out['peak_memory_mb']:.1f} MB | epochs {out['epochs_run']}/{n_epochs}")
                else:
                    row = {
                        "dataset": dataset, "horizon": horizon, "model": model_name,
                        "lookback": lookback,
                        "mse": np.nan, "mae": np.nan, "val_mse": np.nan, "val_mae": np.nan,
                        "train_time_per_epoch_s": np.nan, "peak_memory_mb": np.nan,
                        "epochs_run": 0, "n_params": n_params if 'n_params' in locals() else 0,
                        "batch_size": batch_size, "lr": lr, "status": status,
                    }

                results.append(row)

                # Incremental CSV save after each run (resume-safe)
                df_tmp = pd.DataFrame(results)
                tmp_path = os.path.join(save_dir, "benchmark_results.csv")
                df_tmp.to_csv(tmp_path, index=False)

                # Also free GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Final DataFrame
    df = pd.DataFrame(results)
    # Sort
    df = df.sort_values(["dataset", "horizon", "model"]).reset_index(drop=True)
    out_csv = os.path.join(save_dir, "benchmark_results.csv")
    df.to_csv(out_csv, index=False)
    print("\n" + "=" * 80)
    print(f"[done] Results saved to {out_csv}")
    print(f"[done] Predictions (if enabled) under {os.path.join(save_dir, 'predictions')}")

    # Pretty summary via pandas/tabulate
    try:
        from tabulate import tabulate
        # Pivot for readability: dataset x model MSE per horizon? We'll just print full table
        print("\n=== Summary (MSE/MAE) ===")
        print(tabulate(df[["dataset", "horizon", "model", "mse", "mae", "train_time_per_epoch_s", "peak_memory_mb"]],
                       headers="keys", tablefmt="psql", floatfmt=".4f", showindex=False))
        # Aggregated mean per model
        print("\n=== Mean across datasets & horizons (excl. NaN) ===")
        agg = df.groupby("model")[["mse", "mae"]].mean().round(4)
        print(tabulate(agg.reset_index(), headers="keys", tablefmt="psql", showindex=False))
    except ImportError:
        # Fallback to pandas string
        print(df.to_string())
        print("\nMean per model:")
        print(df.groupby("model")[["mse", "mae"]].mean())

    return df

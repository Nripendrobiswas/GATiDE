"""
Core benchmark orchestration
============================
Loops over datasets x horizons x models, trains, evaluates, aggregates,
saves predictions, and writes benchmark_results.csv.

This is imported by run_benchmark.py CLI.

Protocol – TiDE-paper faithful (Das et al. 2023, TMLR §5.1) + requirement prompt:
  - Lookback L=720 (TiDE always 720, baselines tuned 24..720), horizons H in {96,192,336,720}
  - Sequential split 7:1:2 (70%/10%/20%) for all 7 datasets (Electricity 321, Traffic 862,
    Weather 21, ETTh1/ETTh2 7, ETTm1/ETTm2 7) – TiDE Table 1/2: "train:validation:test ratio is
    7:1:2 as dictated by prior work". Requirement prompt also 70/10/20 (identical). Alternative
    "prior-work" (6:2:2 for ETT) supported via --split-convention.
  - Standardize with train-only mean/std (TiDE: "using the mean and the standard deviations in
    the training period") – metrics in Table 2 are on normalized scale; requirement asks inverse-
    scaled. This harness reports BOTH (mse/mse_norm).
  - MSE loss, Adam (TiDE default) / AdamW (requirement), CosineAnnealing/StepLR, EarlyStopping
  - Rolling evaluation: sliding windows stride 1 from test period (TiDE §3: "evaluated on every
    (look-back, horizon) pair that can be constructed from the test set"), averaged over origins
    and channels. 5 independent seeds averaged for TiDE Table 2 (we run per-seed rows; aggregation
    prints mean±std).
  - Covariates: TiDE uses time-derived global dynamic covariates (hour, dayofweek, month,
    dayofyear + minute if subhourly), normalized train-only (§5.1). Enabled via
    --use-covariates, in which case GATiDE's SegmentAttentionFusion has ≥2 segments.
  - Throughput: training time per epoch (s), inference time per batch (ms), GPU peak memory (MB)
    – TiDE Fig.2 reports training/inference time vs L on Electricity batch 8 T4 GPU.
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
    seeds: Optional[List[int]] = None,  # if set, run 5 seeds and aggregate (TiDE reports mean±std over 5 runs)
    split_convention: str = "tide",
    use_covariates: bool = False,
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
    # Seed handling – TiDE reports mean±std over 5 runs; requirement single seed. Support both.
    seeds_list = seeds if seeds is not None else [seed]
    # Fix initial seed for data loading determinism; per-run seeds set inside loop
    torch.manual_seed(seeds_list[0])
    np.random.seed(seeds_list[0])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeds_list[0])

    results: List[Dict] = []
    total_runs = len(datasets) * len(horizons) * len(models) * len(seeds_list)
    print(f"[benchmark] Planned runs: {total_runs} = {len(datasets)} datasets x {len(horizons)} horizons x {len(models)} models x {len(seeds_list)} seed(s)")
    print(f"[benchmark] Lookback L={lookback} | Horizons {horizons} | Device={device} | Epochs={n_epochs} | Batch={batch_size} | split={split_convention} | covariates={use_covariates}")
    print(f"[benchmark] Metrics: mse/mae (original, requirement) + mse_norm/mae_norm (normalized, TiDE Table 2 comparable)")
    print("-" * 80)

    run_idx = 0
    for dataset in datasets:
        for horizon in horizons:
            # Load split once per dataset-horizon (horizon only affects windowing)
            # TiDE §5.1: 7:1:2 for all datasets; prior-work alternative supported
            try:
                split = load_and_split(csv_dir, dataset, lookback=lookback, horizon=horizon,
                                       split_convention=split_convention, use_covariates=use_covariates)
            except Exception as e:
                print(f"[skip] {dataset} H={horizon} – load failed: {e}")
                for model_name in models:
                  for cur_seed in seeds_list:
                    results.append({
                        "dataset": dataset,
                        "horizon": horizon,
                        "model": model_name,
                        "seed": cur_seed,
                        "lookback": lookback,
                        "mse": np.nan, "mae": np.nan, "mse_norm": np.nan, "mae_norm": np.nan,
                        "val_mse": np.nan, "val_mae": np.nan, "val_mse_norm": np.nan, "val_mae_norm": np.nan,
                        "train_time_per_epoch_s": np.nan, "peak_memory_mb": np.nan, "inference_ms_per_batch": np.nan,
                        "epochs_run": 0, "n_params": 0, "batch_size": batch_size, "lr": lr,
                        "split_convention": split_convention, "status": f"load_failed: {e}",
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
                  for cur_seed in seeds_list:
                    results.append({
                        "dataset": dataset,
                        "horizon": horizon,
                        "model": model_name,
                        "seed": cur_seed,
                        "lookback": lookback,
                        "mse": np.nan, "mae": np.nan, "mse_norm": np.nan, "mae_norm": np.nan,
                        "val_mse": np.nan, "val_mae": np.nan, "val_mse_norm": np.nan, "val_mae_norm": np.nan,
                        "train_time_per_epoch_s": np.nan, "peak_memory_mb": np.nan, "inference_ms_per_batch": np.nan,
                        "epochs_run": 0, "n_params": 0, "batch_size": batch_size, "lr": lr,
                        "split_convention": split_convention, "status": f"window_failed: {e}",
                    })
                continue

            if verbose:
                print(f"  windows: train {len(train_loader.dataset)} | val {len(val_loader.dataset)} | test {len(test_loader.dataset)}")
                if use_covariates and split.cov_train is not None:
                    print(f"  covariates: train {split.cov_train.shape} (time-derived, TiDE §5.1) – GATiDE fusion active")

            for model_name in models:
              for cur_seed in seeds_list:
                run_idx += 1
                tag = f"[{run_idx}/{total_runs}] {dataset} H={horizon} model={model_name} seed={cur_seed}"
                print(f"\n{tag}", flush=True)
                # Set per-run seed (TiDE reports mean over 5 seeds)
                torch.manual_seed(cur_seed)
                np.random.seed(cur_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(cur_seed)

                # Build model – supports both flat per-model and nested per-dataset-horizon tuned configs
                try:
                    ModelClass = get_model(model_name)
                    # Merge kwargs
                    kwargs = dict(
                        num_features=split.n_features,
                        lookback=lookback,
                        horizon=horizon,
                    )
                    # Nested tuned config: {dataset: {horizon: {model: params}}}
                    cur_lr = lr
                    cur_batch = batch_size
                    if dataset in model_kwargs and str(horizon) in model_kwargs[dataset] and model_name in model_kwargs[dataset][str(horizon)]:
                        tuned = model_kwargs[dataset][str(horizon)][model_name]
                        # training overrides
                        if "lr" in tuned:
                            cur_lr = tuned["lr"]
                        if "batch_size" in tuned:
                            cur_batch = tuned["batch_size"]
                        # model kwargs (filter training keys)
                        model_tuned = {k: v for k, v in tuned.items() if k not in ("lr", "batch_size")}
                        kwargs.update(model_tuned)
                    elif model_name in model_kwargs:
                        # flat per-model
                        # check for lr/batch inside flat (rare)
                        flat = model_kwargs[model_name]
                        if isinstance(flat, dict) and "lr" in flat:
                            cur_lr = flat["lr"]
                        if isinstance(flat, dict) and "batch_size" in flat:
                            cur_batch = flat["batch_size"]
                        model_flat = {k: v for k, v in flat.items() if k not in ("lr", "batch_size")} if isinstance(flat, dict) else {}
                        kwargs.update(model_flat)
                    model = ModelClass(**kwargs)
                    n_params = sum(p.numel() for p in model.parameters())
                    print(f"  model: {ModelClass.__name__} | params: {n_params:,} | kwargs: {kwargs}")
                except Exception as e:
                    print(f"  FAILED model build: {e}")
                    results.append({
                        "dataset": dataset, "horizon": horizon, "model": model_name,
                        "seed": cur_seed,
                        "lookback": lookback,
                        "mse": np.nan, "mae": np.nan, "mse_norm": np.nan, "mae_norm": np.nan,
                        "val_mse": np.nan, "val_mae": np.nan, "val_mse_norm": np.nan, "val_mae_norm": np.nan,
                        "train_time_per_epoch_s": np.nan, "peak_memory_mb": np.nan, "inference_ms_per_batch": np.nan,
                        "epochs_run": 0, "n_params": 0, "batch_size": batch_size, "lr": lr,
                        "split_convention": split_convention, "status": f"build_failed: {e}",
                    })
                    continue

                # Train & evaluate – use cur_lr/cur_batch if tuned
                cur_train_loader, cur_val_loader, cur_test_loader = train_loader, val_loader, test_loader
                if cur_batch != batch_size:
                    cur_train_loader, cur_val_loader, cur_test_loader = make_loaders(split, lookback, horizon, batch_size=cur_batch)
                t_start = time.time()
                try:
                    out = train_one_model(
                        model=model,
                        train_loader=cur_train_loader,
                        val_loader=cur_val_loader,
                        test_loader=cur_test_loader,
                        scaler=split.scaler,
                        n_epochs=n_epochs,
                        lr=cur_lr,
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
                    # Save predictions – include seed in filename when multi-seed
                    if save_predictions and out.get("preds") is not None:
                        pred_dir = os.path.join(save_dir, "predictions")
                        os.makedirs(pred_dir, exist_ok=True)
                        suffix = f"_seed{cur_seed}" if len(seeds_list) > 1 else ""
                        pred_path = os.path.join(pred_dir, f"{dataset}_H{horizon}_{model_name}{suffix}_pred.npy")
                        true_path = os.path.join(pred_dir, f"{dataset}_H{horizon}_{model_name}{suffix}_true.npy")
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
                        "seed": cur_seed,
                        "lookback": lookback,
                        "mse": out["test_mse"],
                        "mae": out["test_mae"],
                        "mse_norm": out["test_mse_norm"],
                        "mae_norm": out["test_mae_norm"],
                        "val_mse": out["val_mse"],
                        "val_mae": out["val_mae"],
                        "val_mse_norm": out["val_mse_norm"],
                        "val_mae_norm": out["val_mae_norm"],
                        "train_time_per_epoch_s": out["train_time_per_epoch"],
                        "peak_memory_mb": out["peak_memory_mb"],
                        "inference_ms_per_batch": out["inference_ms_per_batch"],
                        "epochs_run": out["epochs_run"],
                        "n_params": out["n_params"],
                        "batch_size": cur_batch,
                        "lr": cur_lr,
                        "split_convention": split_convention,
                        "status": status,
                    }
                    print(f"  -> MSE {out['test_mse']:.4f} (norm {out['test_mse_norm']:.4f}) | MAE {out['test_mae']:.4f} (norm {out['test_mae_norm']:.4f}) | "
                          f"val MSE {out['val_mse']:.4f} | {out['train_time_per_epoch']:.2f}s/epoch | infer {out['inference_ms_per_batch']:.1f}ms/batch | "
                          f"peak {out['peak_memory_mb']:.1f} MB | epochs {out['epochs_run']}/{n_epochs}")
                else:
                    row = {
                        "dataset": dataset, "horizon": horizon, "model": model_name,
                        "seed": cur_seed,
                        "lookback": lookback,
                        "mse": np.nan, "mae": np.nan, "mse_norm": np.nan, "mae_norm": np.nan,
                        "val_mse": np.nan, "val_mae": np.nan, "val_mse_norm": np.nan, "val_mae_norm": np.nan,
                        "train_time_per_epoch_s": np.nan, "peak_memory_mb": np.nan, "inference_ms_per_batch": np.nan,
                        "epochs_run": 0, "n_params": n_params if 'n_params' in locals() else 0,
                        "batch_size": batch_size, "lr": lr, "split_convention": split_convention, "status": status,
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
    # Sort – include seed if multi-seed
    sort_cols = ["dataset", "horizon", "model", "seed"] if "seed" in df.columns else ["dataset", "horizon", "model"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    out_csv = os.path.join(save_dir, "benchmark_results.csv")
    df.to_csv(out_csv, index=False)
    print("\n" + "=" * 80)
    print(f"[done] Results saved to {out_csv}")
    print(f"[done] Predictions (if enabled) under {os.path.join(save_dir, 'predictions')}")
    print(f"[done] Note: mse/mse_norm – original vs normalized (TiDE Table 2). Requirement uses mse (original); paper comparability uses mse_norm.")

    # Pretty summary via pandas/tabulate – TiDE Table 2 style (normalized) plus requirement (original)
    try:
        from tabulate import tabulate
        cols = ["dataset", "horizon", "model", "seed", "mse", "mse_norm", "mae", "mae_norm", "train_time_per_epoch_s", "inference_ms_per_batch", "peak_memory_mb"]
        cols = [c for c in cols if c in df.columns]
        print("\n=== Summary (MSE/MAE) – original + normalized (TiDE Table 2) ===")
        print(tabulate(df[cols], headers="keys", tablefmt="psql", floatfmt=".4f", showindex=False))
        # Aggregated mean per model – both scales, with std for multi-seed (TiDE reports mean±std over 5 runs)
        print("\n=== Mean across datasets & horizons (excl. NaN) – original ===")
        agg = df.groupby("model")[["mse", "mae"]].mean().round(4)
        print(tabulate(agg.reset_index(), headers="keys", tablefmt="psql", showindex=False))
        print("\n=== Mean normalized (TiDE Table 2 comparable) ===")
        agg_n = df.groupby("model")[["mse_norm", "mae_norm"]].mean().round(4)
        print(tabulate(agg_n.reset_index(), headers="keys", tablefmt="psql", showindex=False))
        if len(seeds_list) > 1:
            print("\n=== Per (dataset,horizon,model) mean±std over seeds (normalized) ===")
            g = df.groupby(["dataset", "horizon", "model"])[["mse_norm", "mae_norm"]].agg(["mean", "std"]).round(4)
            print(g.to_string())
    except ImportError:
        print(df.to_string())
        print("\nMean per model (original):")
        print(df.groupby("model")[["mse", "mae"]].mean())
        print("\nMean normalized:")
        print(df.groupby("model")[["mse_norm", "mae_norm"]].mean())

    return df

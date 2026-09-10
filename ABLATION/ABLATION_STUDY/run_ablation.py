#!/usr/bin/env python
"""
GA-TiDE ablation study orchestrator (2x2 factorial: gate x segment-attention)
=============================================================================

Compares GA-TiDE against the original TiDE architecture by toggling the two
architectural deltas independently, under IDENTICAL data protocol, training
loop, and hyperparameters:

  tide         : vanilla TiDE (no gate, no attention)          [0, 0]
  gatide-gate  : gate + vanilla concat cov mixing              [1, 0]
  gatide-attn  : segment attention + plain residual blocks     [0, 1]
  gatide       : full GA-TiDE (gate + segment attention)       [1, 1]

Shared-tuned policy: the hyperparameters tuned for `gatide` (full) via
tune_optuna.py are reused for ALL variants at each (dataset, horizon) setting,
so observed differences are attributable to architecture alone. Falls back to
configs/default.yaml gatide params when no tuned config exists for a setting.

Usage (from the benchmark/ directory):

  # 1) tune gatide once per setting (the shared hyperparameter source)
  python tune_optuna.py --dataset ETTh1 --horizon 96 --model gatide \
      --batch-sizes 32 64 128 256 512 --n-trials 50 --n-epochs 100 --device auto
  # repeat for: ETTh1 H336, weather H96, weather H336

  # 2) run the ablation grid
  python ABLATION_STUDY/run_ablation.py --datasets ETTh1 weather \
      --horizons 96 336 --seeds 0 1 2 --n-epochs 50 --device auto

Outputs (in ABLATION_STUDY/outputs/ by default):
  benchmark_results.csv  -- one row per (dataset, horizon, variant, seed)
  ablation_summary.csv   -- mean+-std per variant + delta vs tide
  predictions/*.npy      -- per-run pred/true arrays
  console                -- psql-style ablation table with deltas
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# make the benchmark package importable (this file lives in benchmark/ABLATION_STUDY/)
_BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BENCH_ROOT)

from benchmark.benchmark import run_benchmark                      # noqa: E402
from ABLATION_STUDY.variants import register_ablation_variants, ABLATION_ORDER  # noqa: E402

DEFAULT_CSV_DIR = "E:/Machine Learning Research/GATiDE Final Verse/GATiDE/data"
DEFAULT_TUNED_CONFIG = os.path.join(_BENCH_ROOT, "tuned_configs", "tuned_best.yaml")


def load_shared_tuned_kwargs(datasets, horizons, tuned_config: str) -> dict:
    """Build model_kwargs with gatide's tuned params duplicated for all variants.

    Returns {dataset: {str(horizon): {variant: params}}} where `params` includes
    lr/batch_size (consumed by run_benchmark) + model kwargs.
    """
    variants = [m for m in ABLATION_ORDER]
    tuned = {}
    if tuned_config and os.path.exists(tuned_config):
        import yaml
        with open(tuned_config) as f:
            tuned = yaml.safe_load(f) or {}
        if not any(k in tuned for k in datasets):
            tuned = {}

    fallback_params = {}
    default_cfg_path = os.path.join(_BENCH_ROOT, "configs", "default.yaml")
    if os.path.exists(default_cfg_path):
        import yaml
        with open(default_cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        fallback_params = dict(cfg.get("models", {}).get("gatide", {}))

    model_kwargs: dict = {}
    for dataset in datasets:
        for h in horizons:
            key = str(h)
            src = None
            if dataset in tuned and key in tuned[dataset]:
                src = tuned[dataset][key]
            if isinstance(src, dict) and "gatide" in src:
                params = dict(src["gatide"])
                source = "tuned"
            else:
                params = dict(fallback_params)
                source = "default"
            if not params:
                continue
            per_dataset = model_kwargs.setdefault(dataset, {})
            setting = per_dataset.setdefault(key, {})
            for v in variants:
                setting[v] = dict(params)
            print(f"[ablation] shared params for {dataset} H={h} "
                  f"(source: {source} gatide): {params}")
    return model_kwargs


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw run rows into the ablation summary with deltas vs tide."""
    rows = []
    for (dataset, horizon), g in df.dropna(subset=["mse"]).groupby(["dataset", "horizon"]):
        base = g[g["model"] == "tide"]
        if base.empty:
            continue
        base_mse = base["mse"].mean()
        base_mae = base["mae"].mean()
        for model in ABLATION_ORDER:
            gm = g[g["model"] == model]
            if gm.empty:
                continue
            mse_m, mae_m = gm["mse"].mean(), gm["mae"].mean()
            rows.append({
                "dataset": dataset,
                "horizon": horizon,
                "variant": model,
                "n_seeds": int(len(gm)),
                "mse_mean": mse_m,
                "mse_std": float(gm["mse"].std(ddof=0)) if len(gm) > 1 else 0.0,
                "mae_mean": mae_m,
                "mae_std": float(gm["mae"].std(ddof=0)) if len(gm) > 1 else 0.0,
                "mse_norm_mean": gm["mse_norm"].mean(),
                "mae_norm_mean": gm["mae_norm"].mean(),
                "delta_mse_vs_tide": mse_m - base_mse,
                "delta_mae_vs_tide": mae_m - base_mae,
                "pct_mse_change": (100.0 * (mse_m - base_mse) / base_mse) if base_mse else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(
        description="GA-TiDE ablation study (2x2: gate x segment-attention vs vanilla TiDE)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv-dir", type=str, default=DEFAULT_CSV_DIR,
                   help="Folder containing the LTSF CSVs")
    p.add_argument("--datasets", nargs="+", default=["ETTh1", "weather"])
    p.add_argument("--horizons", type=int, nargs="+", default=[96, 336])
    p.add_argument("--models", nargs="+", default=ABLATION_ORDER,
                   help="Subset of the ablation variants to run")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--lookback", type=int, default=720)
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--use-covariates", action=argparse.BooleanOptionalAction, default=True,
                   help="Feed time covariates to all variants (required for the "
                        "attention axis to be active). Default on.")
    p.add_argument("--split-convention", type=str, default="tide", choices=["tide", "prior-work"])
    p.add_argument("--tuned-config", type=str, default=DEFAULT_TUNED_CONFIG,
                   help="Nested tuned yaml {dataset: {horizon: {gatide: params}}}; "
                        "gatide's params are shared across all variants. "
                        "Falls back to configs/default.yaml per setting.")
    p.add_argument("--save-dir", type=str,
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    p.add_argument("--no-save-predictions", dest="save_predictions", action="store_false")
    p.add_argument("--save-predictions", dest="save_predictions", action="store_true", default=True)
    args = p.parse_args()

    register_ablation_variants()

    # equalize variants: share gatide's tuned hyperparams across the 2x2
    model_kwargs = load_shared_tuned_kwargs(args.datasets, args.horizons, args.tuned_config)

    print("\n" + "=" * 80)
    print(" GA-TiDE Ablation Study -- 2x2 factorial (gate x segment-attention)")
    print("=" * 80)
    print(f" datasets={args.datasets} horizons={args.horizons} seeds={args.seeds}")
    print(f" variants={args.models} | epochs={args.n_epochs} | device={args.device} "
          f"| covariates={args.use_covariates} | split={args.split_convention}")
    print(f" save_dir={args.save_dir}")
    print("=" * 80 + "\n")

    df = run_benchmark(
        csv_dir=args.csv_dir,
        datasets=args.datasets,
        horizons=args.horizons,
        models=args.models,
        lookback=args.lookback,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        optimizer="adamw",
        scheduler="cosine",
        patience=args.patience,
        min_delta=1e-4,
        grad_clip=1.0,
        device=args.device,
        seed=args.seeds[0],
        seeds=args.seeds,
        split_convention=args.split_convention,
        use_covariates=args.use_covariates,
        save_dir=args.save_dir,
        save_predictions=args.save_predictions,
        model_kwargs=model_kwargs,
        verbose=True,
    )

    summary = build_summary(df)
    summary_path = os.path.join(args.save_dir, "ablation_summary.csv")
    summary.to_csv(summary_path, index=False)

    # console tables
    try:
        from tabulate import tabulate
        cols = ["dataset", "horizon", "variant", "n_seeds", "mse_mean", "mse_std",
                "mae_mean", "mae_std", "delta_mse_vs_tide", "pct_mse_change"]
        cols = [c for c in cols if c in summary.columns]
        print("\n=== ABLATION SUMMARY (original scale, mean over seeds) ===")
        print(tabulate(summary[cols], headers="keys", tablefmt="psql",
                       floatfmt=".4f", showindex=False))
        print("\n=== ABLATION SUMMARY (normalized scale, TiDE Table 2 comparable) ===")
        cols_n = ["dataset", "horizon", "variant", "mse_norm_mean", "mae_norm_mean",
                  "delta_mse_vs_tide"]
        cols_n = [c for c in cols_n if c in summary.columns]
        print(tabulate(summary[cols_n], headers="keys", tablefmt="psql",
                       floatfmt=".4f", showindex=False))
    except ImportError:
        print(summary.to_string(index=False))

    print(f"\n[done] ablation summary saved to {summary_path}")
    print(f"[done] raw rows in {os.path.join(args.save_dir, 'benchmark_results.csv')}")


if __name__ == "__main__":
    main()

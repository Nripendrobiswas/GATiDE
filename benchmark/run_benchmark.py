#!/usr/bin/env python
"""
Benchmark CLI – entry point

Usage examples:

  # Quick test: one dataset, one horizon, all models, 2 epochs
  python run_benchmark.py --csv-dir "E:/Machine Learning Research/GATiDE Final Verse/GATiDE/data" \
      --datasets ETTh1 --horizons 96 --models gatide tide dlinear naive --epochs 2 --batch-size 32

  # Full benchmark (requirements): L=720, H={96,192,336,720}, all datasets
  python run_benchmark.py --csv-dir ./GATiDE/data --all-horizons --models all --epochs 100 --batch-size 32 --device auto

  # With YAML config
  python run_benchmark.py --config configs/default.yaml --csv-dir ./GATiDE/data --epochs 50

  # Custom save dir
  python run_benchmark.py --csv-dir ./GATiDE/data --save-dir ./my_results --save-predictions

Outputs:
  - {save_dir}/benchmark_results.csv      (aggregated metrics)
  - {save_dir}/predictions/*.npy         (per-run pred/true)
  - Console tabular summary (via tabulate)
"""
from __future__ import annotations

import argparse
import os
import sys
import yaml

# Ensure benchmark package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark.benchmark import run_benchmark
from benchmark.models import list_models
from benchmark.datasets import discover_datasets


def parse_args():
    p = argparse.ArgumentParser(description="GATiDE Benchmark – PyTorch unified loop",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--config", type=str, default=None, help="YAML config file (configs/default.yaml)")

    # Data
    p.add_argument("--csv-dir", type=str, required=False, default=None,
                   help="Path to data/ directory containing *.csv (e.g., E:/.../GATiDE/data)")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Dataset names (stems without .csv), e.g., ETTh1 ETTh2 weather")
    p.add_argument("--all-datasets", action="store_true", help="Use all CSVs in csv_dir (auto-discovery)")
    p.add_argument("--lookback", type=int, default=720, help="Look-back context L")
    p.add_argument("--horizons", type=int, nargs="+", default=None, help="Prediction horizons H, e.g., 96 192 336 720")
    p.add_argument("--all-horizons", action="store_true", help="Use all horizons [96,192,336,720]")

    # Models
    p.add_argument("--models", nargs="+", default=None,
                   help=f"Models to benchmark {list_models()} or 'all'")
    p.add_argument("--model", type=str, default=None, help="Alias for --models single value")

    # Training
    p.add_argument("--epochs", type=int, default=None, help="n_epochs")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--optimizer", type=str, default=None, choices=["adamw", "adam"])
    p.add_argument("--scheduler", type=str, default=None, choices=["cosine", "step", "none"])
    p.add_argument("--patience", type=int, default=None, help="EarlyStopping patience")
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--device", type=str, default=None, help="auto | cpu | cuda")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Multiple seeds for TiDE 5-run mean±std, e.g., --seeds 0 1 2 3 4")
    p.add_argument("--split-convention", type=str, default=None, choices=["tide", "prior-work"], help="tide=7:1:2 all datasets (TiDE paper §5.1), prior-work=6:2:2 for ETT")
    p.add_argument("--use-covariates", action="store_true", help="Generate time covariates (TiDE §5.1) for GATiDE segment attention")

    # Saving
    p.add_argument("--save-dir", type=str, default=None, help="Output directory")
    p.add_argument("--save-predictions", action="store_true", help="Save pred/true .npy")
    p.add_argument("--no-save-predictions", dest="save_predictions", action="store_false")
    p.set_defaults(save_predictions=None)

    # Misc
    p.add_argument("--verbose", action="store_true", default=True)

    args = p.parse_args()

    # Load YAML if provided
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        print(f"[config] Loaded {args.config}")

    # Helper to get value with precedence: CLI > YAML > default
    def get(key_path, cli_val, default=None):
        if cli_val is not None:
            return cli_val
        # nested lookup in yaml
        cur = cfg
        for k in key_path.split("."):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    # Resolve csv_dir
    csv_dir = args.csv_dir or get("data.csv_dir", None, default="E:/Machine Learning Research/GATiDE Final Verse/GATiDE/data")
    if csv_dir is None:
        # Try sibling GATiDE/data
        candidate = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GATiDE", "data")
        if os.path.isdir(candidate):
            csv_dir = candidate
        else:
            # fallback to ./data
            csv_dir = "./data"

    # Datasets
    if args.all_datasets or args.datasets is None:
        # If CLI didn't specify and yaml says auto, discover
        auto = get("data.datasets_auto", None, default=True)
        if args.datasets is not None:
            datasets = args.datasets
        elif auto and os.path.isdir(csv_dir):
            # will be discovered inside run_benchmark
            datasets = None
        else:
            datasets = get("data.datasets", None, default=None)
    else:
        datasets = args.datasets

    # Horizons
    if args.all_horizons:
        horizons = [96, 192, 336, 720]
    elif args.horizons is not None:
        horizons = args.horizons
    else:
        horizons = get("data.horizons", None, default=[96])

    # Models
    if args.model:
        models = [args.model]
    elif args.models is not None:
        if len(args.models) == 1 and args.models[0] == "all":
            models = ["gatide", "tide", "dlinear", "patchtst", "naive"]
        else:
            models = args.models
    else:
        models_cfg = get("benchmark.models", None, default=None)
        # Actually model list not in yaml benchmark, use defaults
        models = ["gatide", "tide", "dlinear", "patchtst", "naive"]

    # Training params with defaults from yaml
    lookback = get("data.lookback", args.lookback, default=720)
    n_epochs = get("training.n_epochs", args.epochs, default=100)
    batch_size = get("training.batch_size", args.batch_size, default=32)
    lr = get("training.lr", args.lr, default=1e-3)
    weight_decay = get("training.weight_decay", args.weight_decay, default=1e-4)
    optimizer = get("training.optimizer", args.optimizer, default="adamw")
    scheduler = get("training.scheduler", args.scheduler, default="cosine")
    scheduler_params = get("training.scheduler_params", None, default={})
    patience = get("training.early_stopping.patience", args.patience, default=10)
    # min_delta from yaml
    min_delta = 1e-4
    if cfg and "training" in cfg and "early_stopping" in cfg["training"]:
        min_delta = cfg["training"]["early_stopping"].get("min_delta", 1e-4)
    grad_clip = get("training.grad_clip", args.grad_clip, default=1.0)
    device = get("training.device", args.device, default="auto")
    seed = get("training.seed", args.seed, default=42)
    seeds = args.seeds if args.seeds is not None else get("training.seeds", None, default=None)
    split_convention = get("data.split_convention", args.split_convention, default="tide")
    use_covariates = args.use_covariates or get("data.use_covariates", None, default=False)
    amp = args.amp or get("training.amp", None, default=False)
    save_dir = get("benchmark.save_dir", args.save_dir, default="./benchmark_outputs")
    # save_predictions tri-state
    if args.save_predictions is None:
        save_predictions = get("benchmark.save_predictions", None, default=True)
    else:
        save_predictions = args.save_predictions
    if save_predictions is None:
        save_predictions = True

    # Validation
    if not os.path.isdir(csv_dir):
        print(f"[error] csv_dir not found: {csv_dir}")
        print(f"  Tip: set --csv-dir to the folder containing ETTh1.csv etc., e.g.,")
        print(f"       --csv-dir \"E:/Machine Learning Research/GATiDE Final Verse/GATiDE/data\"")
        # Don't exit immediately; allow discovery to be attempted but warn
    else:
        print(f"[info] Using csv_dir: {csv_dir}")
        try:
            discovered = discover_datasets(csv_dir)
            print(f"[info] Found datasets: {discovered}")
        except Exception as e:
            print(f"[warn] discover failed: {e}")

    return argparse.Namespace(
        csv_dir=csv_dir,
        datasets=datasets,
        lookback=lookback,
        horizons=horizons,
        models=models,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_params=scheduler_params,
        patience=patience,
        min_delta=min_delta,
        grad_clip=grad_clip,
        device=device,
        amp=amp,
        seed=seed,
        seeds=seeds,
        split_convention=split_convention,
        use_covariates=use_covariates,
        save_dir=save_dir,
        save_predictions=save_predictions,
        verbose=args.verbose,
    )


def main():
    args = parse_args()
    print("\n" + "="*80)
    print(" GATiDE Benchmark – Unified PyTorch Loop (TiDE-paper faithful)")
    print("="*80)
    print(f" Datasets   : {args.datasets if args.datasets else 'auto-discover'} | split={args.split_convention} covariates={args.use_covariates}")
    print(f" Horizons   : {args.horizons}  | Lookback L={args.lookback} (TiDE always 720)")
    print(f" Models     : {args.models}")
    print(f" Training   : epochs={args.n_epochs} batch={args.batch_size} lr={args.lr} opt={args.optimizer} sched={args.scheduler}")
    print(f" Device     : {args.device} | AMP={args.amp} | Seed(s)={args.seeds if args.seeds else args.seed}")
    print(f" Save dir   : {args.save_dir} | Save preds={args.save_predictions}")
    print("="*80 + "\n")

    # Per-model kwargs from YAML if available
    model_kwargs = {}
    if args.__dict__.get("config") and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        if "models" in cfg:
            # flatten per-model overrides except generic 'model'
            for k, v in cfg["models"].items():
                if isinstance(v, dict):
                    model_kwargs[k] = v
        if "model" in cfg and isinstance(cfg["model"], dict):
            # generic overrides – apply to all? We'll merge into each
            generic = cfg["model"]
            for mk in ["gatide", "tide", "dlinear", "patchtst", "naive"]:
                if mk not in model_kwargs:
                    model_kwargs[mk] = {}
                # only set if not already set per-model
                for gk, gv in generic.items():
                    if gk not in model_kwargs[mk]:
                        model_kwargs[mk][gk] = gv

    df = run_benchmark(
        csv_dir=args.csv_dir,
        datasets=args.datasets,
        horizons=args.horizons,
        models=args.models,
        lookback=args.lookback,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        scheduler_params=args.scheduler_params,
        patience=args.patience,
        min_delta=args.min_delta,
        grad_clip=args.grad_clip,
        device=args.device,
        amp=args.amp,
        seed=args.seed,
        seeds=args.seeds,
        split_convention=args.split_convention,
        use_covariates=args.use_covariates,
        save_dir=args.save_dir,
        save_predictions=args.save_predictions,
        model_kwargs=model_kwargs,
        verbose=args.verbose,
    )

    print("\nBenchmark complete.")
    print(f"CSV: {os.path.join(args.save_dir, 'benchmark_results.csv')}")


if __name__ == "__main__":
    main()

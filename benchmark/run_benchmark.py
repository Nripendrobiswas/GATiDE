#!/usr/bin/env python
"""
Benchmark CLI – entry point

Usage examples:

  # Quick test: one dataset, one horizon, all models, 2 epochs
  python run_benchmark.py --csv-dir GATiDE/data \
      --datasets ETTh1 --horizons 96 --models gatide tide dlinear naive --epochs 2 --batch-size 32

  # Full benchmark (TiDE paper §5.1): L=720, H={96,192,336,720}, all datasets, 5 seeds
  python run_benchmark.py --csv-dir GATiDE/data --all-horizons --models all --epochs 100 \
      --batch-size 32 --device auto --seeds 0 1 2 3 4

  # With tuned model params (NO YAML needed):
  python run_benchmark.py --csv-dir GATiDE/data --datasets ETTh1 --horizons 96 \
      --models gatide --model-kwargs '{"hidden_size":128,"num_encoder_layers":1}' \
      --tune-lr 0.004 --tune-batch-size 512 --epochs 100 --device cuda

  # Per-model tuned params via --model-kwargs-{model}:
  python run_benchmark.py --csv-dir GATiDE/data --all-horizons --models gatide tide \
      --model-kwargs-gatide '{"hidden_size":128,"dropout":0.1}' \
      --model-kwargs-tide '{"hidden_size":256}' \
      --epochs 100 --device cuda

  # With YAML config (still supported):
  python run_benchmark.py --config configs/default.yaml --csv-dir GATiDE/data --epochs 50

Outputs:
  - {save_dir}/benchmark_results.csv      (aggregated metrics)
  - {save_dir}/predictions/*.npy           (per-run pred/true)
  - Console tabular summary (via tabulate)
"""
from __future__ import annotations

import argparse
import json
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

    p.add_argument("--config", type=str, default=None, help="YAML config file (optional, CLI args override)")

    # ── Data ──
    p.add_argument("--csv-dir", type=str, required=False, default=None,
                   help="Path to data/ directory containing *.csv")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Dataset names (stems without .csv), e.g., ETTh1 ETTh2 weather")
    p.add_argument("--all-datasets", action="store_true", help="Use all CSVs in csv_dir")
    p.add_argument("--lookback", type=int, default=None, help="Look-back context L (default 720)")
    p.add_argument("--horizons", type=int, nargs="+", default=None, help="Prediction horizons H, e.g., 96 192 336 720")
    p.add_argument("--all-horizons", action="store_true", help="Use [96, 192, 336, 720]")

    # ── Models ──
    p.add_argument("--models", nargs="+", default=None,
                   help=f"Models to benchmark {list_models()} or 'all'")
    p.add_argument("--model", type=str, default=None, help="Single model (alias for --models)")

    # ── Training ──
    p.add_argument("--epochs", type=int, default=None, help="n_epochs")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--optimizer", type=str, default=None, choices=["adamw", "adam"])
    p.add_argument("--scheduler", type=str, default=None, choices=["cosine", "step", "none"])
    p.add_argument("--patience", type=int, default=None, help="EarlyStopping patience")
    p.add_argument("--min-delta", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--device", type=str, default=None, help="auto | cpu | cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Multiple seeds for 5-run mean±std, e.g., --seeds 0 1 2 3 4")
    p.add_argument("--split-convention", type=str, default=None,
                   choices=["tide", "prior-work"])
    p.add_argument("--use-covariates", action="store_true")

    # ── Tuned model hyperparams (NO YAML) ──
    p.add_argument("--model-kwargs", type=str, default=None,
                   help='JSON string applied to ALL models, e.g., \'{"hidden_size":128}\'')
    p.add_argument("--model-kwargs-gatide", type=str, default=None,
                   help='JSON string applied to GATiDE only')
    p.add_argument("--model-kwargs-tide", type=str, default=None,
                   help='JSON string applied to TiDE only')
    p.add_argument("--model-kwargs-dlinear", type=str, default=None,
                   help='JSON string applied to DLinear only')
    p.add_argument("--model-kwargs-patchtst", type=str, default=None,
                   help='JSON string applied to PatchTST only')
    p.add_argument("--model-kwargs-naive", type=str, default=None,
                   help='JSON string applied to Naive only')
    p.add_argument("--tune-lr", type=float, default=None,
                   help="Override learning rate (from tuning)")
    p.add_argument("--tune-batch-size", type=int, default=None,
                   help="Override batch size (from tuning)")

    # ── Saving ──
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--no-save-predictions", dest="save_predictions", action="store_false")
    p.set_defaults(save_predictions=None)

    # ── Misc ──
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--quiet", action="store_true")

    args = p.parse_args()

    if args.quiet:
        args.verbose = False

    # ── Merge YAML defaults (CLI always wins) ──
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        print(f"[config] Loaded {args.config}")

    def get(key_path, cli_val, default=None):
        if cli_val is not None:
            return cli_val
        cur = cfg
        for k in key_path.split("."):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    # ── Resolve csv_dir ──
    csv_dir = args.csv_dir or get("data.csv_dir", None, default=None)
    if csv_dir is None:
        candidate = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GATiDE", "data")
        if os.path.isdir(candidate):
            csv_dir = candidate
        else:
            csv_dir = "./data"

    # ── Datasets ──
    if args.all_datasets:
        datasets = None  # auto-discover
    elif args.datasets is not None:
        datasets = args.datasets
    else:
        datasets = get("data.datasets", None, default=None)

    # ── Horizons ──
    if args.all_horizons:
        horizons = [96, 192, 336, 720]
    elif args.horizons is not None:
        horizons = args.horizons
    else:
        horizons = get("data.horizons", None, default=[96])

    # ── Models ──
    if args.model:
        models = [args.model]
    elif args.models is not None:
        if len(args.models) == 1 and args.models[0] == "all":
            models = ["gatide", "tide", "dlinear", "patchtst", "naive"]
        else:
            models = args.models
    else:
        models = ["gatide", "tide", "dlinear", "patchtst", "naive"]

    # ── Training params ──
    lookback      = get("data.lookback", args.lookback, default=720)
    n_epochs      = get("training.n_epochs", args.epochs, default=100)
    batch_size    = get("training.batch_size", args.batch_size, default=32)
    lr            = get("training.lr", args.lr, default=1e-3)
    weight_decay  = get("training.weight_decay", args.weight_decay, default=1e-4)
    optimizer     = get("training.optimizer", args.optimizer, default="adamw")
    scheduler     = get("training.scheduler", args.scheduler, default="cosine")
    scheduler_params = get("training.scheduler_params", None, default={})
    patience      = get("training.early_stopping.patience", args.patience, default=10)
    min_delta     = args.min_delta if args.min_delta is not None else get("training.early_stopping.min_delta", None, default=1e-4)
    grad_clip     = get("training.grad_clip", args.grad_clip, default=1.0)
    device        = get("training.device", args.device, default="auto")
    seed          = get("training.seed", args.seed, default=42)
    seeds         = args.seeds if args.seeds is not None else get("training.seeds", None, default=None)
    split_convention = get("data.split_convention", args.split_convention, default="tide")
    use_covariates   = args.use_covariates or get("data.use_covariates", None, default=False)
    amp           = args.amp or get("training.amp", None, default=False)
    save_dir      = get("benchmark.save_dir", args.save_dir, default="./benchmark_outputs")
    if args.save_predictions is None:
        save_predictions = get("benchmark.save_predictions", None, default=True)
    else:
        save_predictions = args.save_predictions
    if save_predictions is None:
        save_predictions = True

    # ── Apply --tune-lr / --tune-batch-size overrides ──
    if args.tune_lr is not None:
        lr = args.tune_lr
    if args.tune_batch_size is not None:
        batch_size = args.tune_batch_size

    # ── Validation ──
    if not os.path.isdir(csv_dir):
        print(f"[error] csv_dir not found: {csv_dir}")
        print(f"  Tip: --csv-dir <path-to-folder-with-ETTh1.csv>")
    else:
        print(f"[info] csv_dir: {csv_dir}")
        try:
            discovered = discover_datasets(csv_dir)
            print(f"[info] Found: {discovered}")
        except Exception as e:
            print(f"[warn] discover failed: {e}")

    # ── Build model_kwargs from --model-kwargs / --model-kwargs-{model} / YAML ──
    model_kwargs = {}

    # 1. YAML flat overrides
    if cfg:
        tuned_datasets = {"ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "weather", "traffic",
                          "Electricity", "Weather", "Traffic"}
        if any(k in tuned_datasets for k in cfg.keys()):
            model_kwargs = cfg  # nested tuned – pass as-is
        else:
            if "models" in cfg:
                for k, v in cfg["models"].items():
                    if isinstance(v, dict):
                        model_kwargs[k] = v
            if "model" in cfg and isinstance(cfg["model"], dict):
                generic = cfg["model"]
                for mk in ["gatide", "tide", "dlinear", "patchtst", "naive"]:
                    if mk not in model_kwargs:
                        model_kwargs[mk] = {}
                    for gk, gv in generic.items():
                        if gk not in model_kwargs[mk]:
                            model_kwargs[mk][gk] = gv

    # 2. --model-kwargs JSON (applied to ALL models, overrides YAML)
    if args.model_kwargs:
        common = json.loads(args.model_kwargs)
        for mk in models:
            if mk not in model_kwargs:
                model_kwargs[mk] = {}
            model_kwargs[mk].update(common)

    # 3. --model-kwargs-{model} per-model JSON (overrides everything above)
    per_model_flags = {
        "gatide":   args.model_kwargs_gatide,
        "tide":     args.model_kwargs_tide,
        "dlinear":  args.model_kwargs_dlinear,
        "patchtst": args.model_kwargs_patchtst,
        "naive":    args.model_kwargs_naive,
    }
    for mk, flag_val in per_model_flags.items():
        if flag_val is not None:
            override = json.loads(flag_val)
            if mk not in model_kwargs:
                model_kwargs[mk] = {}
            model_kwargs[mk].update(override)

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
        model_kwargs=model_kwargs,
        verbose=args.verbose,
    )


def main():
    args = parse_args()
    print("\n" + "=" * 80)
    print(" GATiDE Benchmark – Unified PyTorch Loop (TiDE-paper faithful)")
    print("=" * 80)
    print(f" Datasets   : {args.datasets if args.datasets else 'auto-discover'} | split={args.split_convention} covariates={args.use_covariates}")
    print(f" Horizons   : {args.horizons}  | Lookback L={args.lookback}")
    print(f" Models     : {args.models}")
    print(f" Training   : epochs={args.n_epochs} batch={args.batch_size} lr={args.lr} opt={args.optimizer} sched={args.scheduler}")
    print(f" Device     : {args.device} | AMP={args.amp} | Seed(s)={args.seeds if args.seeds else args.seed}")
    print(f" Save dir   : {args.save_dir} | Save preds={args.save_predictions}")
    if args.model_kwargs:
        print(f" model_kwargs  : {args.model_kwargs}")
    per_model = {mk: getattr(args, f"model_kwargs_{mk}", None)
                 for mk in ["gatide", "tide", "dlinear", "patchtst", "naive"]
                 if getattr(args, f"model_kwargs_{mk}", None) is not None}
    if per_model:
        for mk, j in per_model.items():
            print(f" model-kwargs-{mk}: {j}")
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
        model_kwargs=args.model_kwargs,
        verbose=args.verbose,
    )

    print("\nBenchmark complete.")
    print(f"CSV: {os.path.join(args.save_dir, 'benchmark_results.csv')}")


if __name__ == "__main__":
    main()

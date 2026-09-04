#!/usr/bin/env python
"""
Standalone Ablation Runner - does NOT modify benchmark/ or src/
===============================================================
Drop this folder at GATiDE/ablation_study/ and run from repo root:

  python ablation_study/run_ablation.py --csv-dir GATiDE/data --datasets ETTh1 --horizons 96 --ablations gatide gatide_nogate tide --epochs 100 --seeds 0 1 2 3 4 --save-dir ./ablation_study/results/grb_96

Uses benchmark/ core (datasets, trainer, utils) but with local ablation models.
Registers ablation variants at runtime without patching benchmark/models/__init__.py.

Protocol mirrors benchmark/benchmark.py: TiDE faithful L=720, 7:1:2 split, StandardScaler train-only, MSE+AdamW+Cosine+EarlyStopping, dual metrics mse/mse_norm.

Ablations:
  gatide         -> GATiDEAblation(use_gating=True, use_layer_norm=True, use_skip=True)  full
  gatide_nogate  -> GATiDEAblation(use_gating=False, ...) isolates gate (true GRB ablation)
  gatide_no_ln   -> GRB without LN
  gatide_no_skip -> GRB without global skip
  tide/dlinear/patchtst/naive -> benchmark/models baselines (imported)

Example with tuned hyperparams (no YAML):
  python ablation_study/run_ablation.py --csv-dir GATiDE/data --datasets ETTh1 --horizons 96 --ablations gatide --model-kwargs '{"hidden_size":128,"dropout":0.1}' --tune-lr 0.004 --tune-batch-size 512 --epochs 100 --seeds 0 1 2 3 4
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import yaml
from pathlib import Path

# Ensure repo root and benchmark are importable regardless of cwd
# Robust search for benchmark/datasets.py across Kaggle + local layouts
THIS_DIR = Path(__file__).resolve().parent
# Make THIS_DIR importable for `import models.gatide_ablation`
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

def _find_benchmark_root():
    # Explicit candidates (Kaggle + local)
    explicit = [
        THIS_DIR.parent / "benchmark",  # GATiDE/benchmark when ablation_study at GATiDE/ablation_study
        THIS_DIR / "benchmark",
        Path.cwd() / "benchmark",
        Path.cwd() / "GATiDE" / "benchmark",
        Path.cwd() / "ablation" / "benchmark",
        THIS_DIR.parent.parent / "benchmark",
        THIS_DIR.parent.parent / "ablation" / "benchmark",
        Path(__file__).resolve().parents[2] / "benchmark",
        Path(__file__).resolve().parents[3] / "benchmark",
        Path("E:/Machine Learning Research/GATiDE Final Verse/ablation/benchmark"),
        Path("E:/Machine Learning Research/GATiDE Final Verse/GATiDE/benchmark"),
        Path("/kaggle/working/GATiDE/benchmark"),
        Path("/kaggle/working/benchmark"),
    ]
    candidates = []
    # Add explicit first
    for cand in explicit:
        try:
            if (cand / "datasets.py").exists():
                candidates.append(cand.parent)
        except Exception:
            pass
    # Walk up from THIS_DIR and cwd (including GATiDE subfolder check)
    for base in [THIS_DIR, THIS_DIR.parent, Path.cwd(), Path.cwd() / "GATiDE"]:
        try:
            cur = base.resolve()
        except Exception:
            continue
        for _ in range(8):
            if (cur / "benchmark" / "datasets.py").exists():
                if cur not in candidates:
                    candidates.append(cur)
                break
            if (cur / "GATiDE" / "benchmark" / "datasets.py").exists():
                if (cur / "GATiDE") not in candidates:
                    candidates.append(cur / "GATiDE")
                break
            if (cur / "ablation" / "benchmark" / "datasets.py").exists():
                if (cur / "ablation") not in candidates:
                    candidates.append(cur / "ablation")
                break
            if cur.parent == cur:
                break
            cur = cur.parent
    # Return first candidate that exists
    for c in candidates:
        if (c / "benchmark" / "datasets.py").exists():
            return c
    return None

BENCHMARK_ROOT = _find_benchmark_root()
if BENCHMARK_ROOT is None:
    # Try fallback: THIS_DIR.parent
    BENCHMARK_ROOT = THIS_DIR.parent
    print(f"[warn] benchmark/datasets.py not found via search, fallback to {BENCHMARK_ROOT}")
    print(f"[warn] Searched candidates, sys.path will be {sys.path}, cwd={Path.cwd()}, THIS_DIR={THIS_DIR}")
else:
    print(f"[info] Found benchmark at {BENCHMARK_ROOT / 'benchmark'}")

# Ensure benchmark root is in sys.path
for p in [BENCHMARK_ROOT, THIS_DIR, Path.cwd(), Path.cwd() / "GATiDE"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
# Also ensure parent of benchmark root (for nested layouts) is not needed, but add REPO_ROOT for compatibility
REPO_ROOT = BENCHMARK_ROOT  # alias for later code that uses REPO_ROOT

# Debug hint if benchmark still not found (helps Kaggle)
try:
    import benchmark.datasets  # noqa: F401
except ModuleNotFoundError as e:
    print(f"[error] benchmark not found. sys.path={sys.path}")
    print(f"[debug] THIS_DIR={THIS_DIR}, BENCHMARK_ROOT={BENCHMARK_ROOT}, cwd={Path.cwd()}")
    print(f"[debug] BENCHMARK_ROOT/benchmark exists={(BENCHMARK_ROOT/'benchmark').exists() if BENCHMARK_ROOT else 'N/A'}")
    print(f"[debug] cwd/benchmark={(Path.cwd()/'benchmark').exists()}, cwd/GATiDE/benchmark={(Path.cwd()/'GATiDE'/'benchmark').exists()}")
    print(f"[debug] THIS_DIR.parent/benchmark={(THIS_DIR.parent/'benchmark').exists()}, THIS_DIR.parent.parent/ablation/benchmark={(THIS_DIR.parent.parent/'ablation'/'benchmark').exists()}")
    raise

# Imports after path setup
from benchmark.datasets import load_and_split, make_loaders, discover_datasets
from benchmark.trainer import train_one_model
from benchmark.models import get_model as get_benchmark_model  # for tide/dlinear baselines

# Local ablation models - robust import (works whether `models` is top-level or ablation_study.models)
try:
    from models.gatide_ablation import GATiDEAblation, GATiDE_Gated, GATiDE_NoGate, GATiDE_NoLN, GATiDE_NoSkip
except ModuleNotFoundError:
    # Fallback: load via importlib from file path (Kaggle-safe)
    import importlib.util
    spec = importlib.util.spec_from_file_location("gatide_ablation", THIS_DIR / "models" / "gatide_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    GATiDEAblation = mod.GATiDEAblation
    GATiDE_Gated = mod.GATiDE_Gated
    GATiDE_NoGate = mod.GATiDE_NoGate
    GATiDE_NoLN = mod.GATiDE_NoLN
    GATiDE_NoSkip = mod.GATiDE_NoSkip

ABLATION_REGISTRY = {
    "gatide": GATiDE_Gated,
    "gatide_gated": GATiDE_Gated,
    "gatide_nogate": GATiDE_NoGate,
    "gatide_no_gate": GATiDE_NoGate,
    "gatide-rb": GATiDE_NoGate,
    "gatide_no_ln": GATiDE_NoLN,
    "gatide_noln": GATiDE_NoLN,
    "gatide_no_skip": GATiDE_NoSkip,
    "gatide_noskip": GATiDE_NoSkip,
    # allow direct class name
    "gatide_ablation": GATiDEAblation,
}

BASELINE_KEYS = {"tide", "dlinear", "patchtst", "naive", "persistence"}

def parse_args():
    p = argparse.ArgumentParser(description="GATiDE Ablation Study - standalone", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=str, default=None, help="yaml config (optional, CLI overrides)")
    p.add_argument("--csv-dir", type=str, default=None, help="path to GATiDE/data with *.csv")
    p.add_argument("--datasets", nargs="+", default=None, help="dataset stems e.g., ETTh1 weather")
    p.add_argument("--all-datasets", action="store_true")
    p.add_argument("--horizons", type=int, nargs="+", default=None)
    p.add_argument("--all-horizons", action="store_true")
    p.add_argument("--lookback", type=int, default=None)
    p.add_argument("--ablations", nargs="+", default=None, help=f"ablation keys {list(ABLATION_REGISTRY)} + {list(BASELINE_KEYS)} or 'all'")
    p.add_argument("--ablation", type=str, default=None, help="single ablation alias")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--optimizer", type=str, choices=["adamw", "adam"], default=None)
    p.add_argument("--scheduler", type=str, choices=["cosine", "step", "none"], default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="auto|cpu|cuda")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--split-convention", type=str, choices=["tide", "prior-work"], default=None)
    p.add_argument("--use-covariates", action="store_true")
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--model-kwargs", type=str, default=None, help='JSON for ALL ablations e.g. \'{"hidden_size":128}\'')
    p.add_argument("--model-kwargs-gatide", type=str, default=None)
    p.add_argument("--model-kwargs-gatide-nogate", type=str, default=None)
    p.add_argument("--tune-lr", type=float, default=None)
    p.add_argument("--tune-batch-size", type=int, default=None)
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()

def load_yaml_config(path):
    if path and os.path.exists(path):
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        print(f"[config] Loaded {path}")
        return cfg
    return {}

def main():
    args = parse_args()
    if args.quiet:
        args.verbose = False
    cfg = load_yaml_config(args.config) if args.config else load_yaml_config(str(THIS_DIR / "configs" / "ablation_config.yaml"))

    def get(key, cli_val, default=None):
        if cli_val is not None:
            return cli_val
        cur = cfg
        for k in key.split("."):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    # Resolve csv_dir
    csv_dir = args.csv_dir or get("data.csv_dir", None, default="GATiDE/data")
    # Try alternative relative to repo root
    candidates = [csv_dir, str(REPO_ROOT / "GATiDE" / "data"), str(REPO_ROOT / "data"), str(Path.cwd() / "GATiDE" / "data"), "./data", "../GATiDE/data"]
    found_csv = None
    for c in candidates:
        if c and os.path.isdir(c):
            found_csv = c
            break
    if found_csv:
        csv_dir = found_csv
    if not os.path.isdir(csv_dir):
        print(f"[error] csv_dir not found: {csv_dir}")
        print(f"  tried: {candidates}")
    else:
        print(f"[info] csv_dir: {csv_dir}")
        try:
            print(f"[info] Found: {discover_datasets(csv_dir)}")
        except Exception as e:
            print(f"[warn] discover failed: {e}")

    datasets = None
    if args.all_datasets:
        datasets = None
    elif args.datasets is not None:
        datasets = args.datasets
    else:
        datasets = get("data.datasets", None, default=["ETTh1"])

    if args.all_horizons:
        horizons = [96, 192, 336, 720]
    elif args.horizons is not None:
        horizons = args.horizons
    else:
        horizons = get("data.horizons", None, default=get("data.horizons", None, default=[96]))

    # Ablations
    if args.ablation:
        ablations = [args.ablation]
    elif args.ablations is not None:
        if len(args.ablations)==1 and args.ablations[0]=="all":
            ablations = ["gatide", "gatide_nogate", "gatide_no_ln", "gatide_no_skip", "tide", "dlinear", "naive"]
        else:
            ablations = args.ablations
    else:
        ablations = get("ablation.variants", None, default=["gatide", "gatide_nogate", "tide"])

    lookback = get("data.lookback", args.lookback, default=720)
    n_epochs = get("training.n_epochs", args.epochs, default=100)
    batch_size = get("training.batch_size", args.batch_size, default=32)
    lr = get("training.lr", args.lr, default=1e-3)
    weight_decay = get("training.weight_decay", args.weight_decay, default=1e-4)
    optimizer = get("training.optimizer", args.optimizer, default="adamw")
    scheduler = get("training.scheduler", args.scheduler, default="cosine")
    scheduler_params = get("training.scheduler_params", None, default={})
    patience = get("training.early_stopping.patience", args.patience, default=10)
    min_delta = get("training.early_stopping.min_delta", None, default=1e-4)
    grad_clip = get("training.grad_clip", None, default=1.0)
    device = get("training.device", args.device, default="auto")
    seed = get("training.seed", args.seed, default=42)
    seeds = args.seeds if args.seeds is not None else get("training.seeds", None, default=None)
    split_convention = get("data.split_convention", args.split_convention, default="tide")
    use_covariates = args.use_covariates or get("data.use_covariates", None, default=False)
    save_dir = get("benchmark.save_dir", args.save_dir, default=str(THIS_DIR / "results"))
    if args.tune_lr is not None:
        lr = args.tune_lr
    if args.tune_batch_size is not None:
        batch_size = args.tune_batch_size

    # Model kwargs merging
    model_kwargs = {}
    # from yaml ablation.model_kwargs
    if "ablation" in cfg and "model_kwargs" in cfg["ablation"]:
        for k, v in cfg["ablation"]["model_kwargs"].items():
            model_kwargs[k] = v
    if args.model_kwargs:
        common = json.loads(args.model_kwargs)
        for ab in ablations:
            model_kwargs.setdefault(ab, {}).update(common)
    per_flag = {
        "gatide": args.model_kwargs_gatide,
        "gatide_nogate": args.model_kwargs_gatide_nogate,
    }
    for k, v in per_flag.items():
        if v is not None:
            override = json.loads(v)
            model_kwargs.setdefault(k, {}).update(override)

    print("\n" + "="*80)
    print(" GATiDE Ablation Study - standalone (no benchmark/ modification)")
    print("="*80)
    print(f" Datasets   : {datasets if datasets else 'auto'} | split={split_convention} covariates={use_covariates}")
    print(f" Horizons   : {horizons} | L={lookback}")
    print(f" Ablations  : {ablations}")
    print(f" Training   : epochs={n_epochs} batch={batch_size} lr={lr} opt={optimizer} sched={scheduler}")
    print(f" Device     : {device} | Seeds={seeds if seeds else seed}")
    print(f" Save dir   : {save_dir}")
    if model_kwargs:
        print(f" model_kwargs keys: {list(model_kwargs.keys())}")
    print("="*80 + "\n")

    # Run
    import numpy as np, pandas as pd, torch
    os.makedirs(save_dir, exist_ok=True)
    if datasets is None:
        datasets = discover_datasets(csv_dir)
        print(f"[ablation] Auto-discovered: {datasets}")
    seeds_list = seeds if seeds is not None else [seed]
    torch.manual_seed(seeds_list[0]); np.random.seed(seeds_list[0])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeds_list[0])

    results = []
    total = len(datasets) * len(horizons) * len(ablations) * len(seeds_list)
    print(f"[ablation] Planned runs: {total} = {len(datasets)} datasets x {len(horizons)} horizons x {len(ablations)} ablations x {len(seeds_list)} seeds")

    run_idx = 0
    for dataset in datasets:
        for horizon in horizons:
            try:
                split = load_and_split(csv_dir, dataset, lookback=lookback, horizon=horizon, split_convention=split_convention, use_covariates=use_covariates)
            except Exception as e:
                print(f"[skip] {dataset} H={horizon} load failed: {e}")
                for ab in ablations:
                    for s in seeds_list:
                        results.append({"dataset":dataset,"horizon":horizon,"ablation":ab,"model":ab,"seed":s,"lookback":lookback,"mse":np.nan,"mae":np.nan,"mse_norm":np.nan,"mae_norm":np.nan,"val_mse":np.nan,"val_mae":np.nan,"val_mse_norm":np.nan,"val_mae_norm":np.nan,"train_time_per_epoch_s":np.nan,"peak_memory_mb":np.nan,"inference_ms_per_batch":np.nan,"epochs_run":0,"n_params":0,"batch_size":batch_size,"lr":lr,"split_convention":split_convention,"status":f"load_failed:{e}"})
                continue
            print(f"\n[dataset] {dataset} | T={split.T_total} train{split.T_train} val{split.T_val} test{split.T_test} | C={split.n_features} | L={lookback} H={horizon}")
            try:
                train_loader, val_loader, test_loader = make_loaders(split, lookback=lookback, horizon=horizon, batch_size=batch_size, num_workers=0)
            except ValueError as e:
                print(f"[skip] {dataset} H={horizon} window failed: {e}")
                for ab in ablations:
                    for s in seeds_list:
                        results.append({"dataset":dataset,"horizon":horizon,"ablation":ab,"model":ab,"seed":s,"lookback":lookback,"mse":np.nan,"mae":np.nan,"mse_norm":np.nan,"mae_norm":np.nan,"val_mse":np.nan,"val_mae":np.nan,"val_mse_norm":np.nan,"val_mae_norm":np.nan,"train_time_per_epoch_s":np.nan,"peak_memory_mb":np.nan,"inference_ms_per_batch":np.nan,"epochs_run":0,"n_params":0,"batch_size":batch_size,"lr":lr,"split_convention":split_convention,"status":f"window_failed:{e}"})
                continue
            print(f"  windows: train {len(train_loader.dataset)} | val {len(val_loader.dataset)} | test {len(test_loader.dataset)}")
            for ab in ablations:
                for cur_seed in seeds_list:
                    run_idx += 1
                    tag = f"[{run_idx}/{total}] {dataset} H={horizon} ablation={ab} seed={cur_seed}"
                    print(f"\n{tag}", flush=True)
                    torch.manual_seed(cur_seed); np.random.seed(cur_seed)
                    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cur_seed)
                    # Resolve model class
                    is_baseline = ab.lower() in BASELINE_KEYS
                    try:
                        if is_baseline:
                            ModelClass = get_benchmark_model(ab)
                        else:
                            # ablation registry (normalize)
                            key = ab.lower().replace("-", "_")
                            ModelClass = ABLATION_REGISTRY.get(key)
                            if ModelClass is None:
                                # try benchmark registry as fallback (e.g., gatide)
                                try:
                                    ModelClass = get_benchmark_model(ab)
                                except:
                                    raise ValueError(f"Unknown ablation {ab}. Available ablations {list(ABLATION_REGISTRY)} + baselines {list(BASELINE_KEYS)}")
                        kwargs = dict(num_features=split.n_features, lookback=lookback, horizon=horizon)
                        # merge per-ablation kwargs
                        if ab in model_kwargs:
                            kwargs.update(model_kwargs[ab])
                        elif key in model_kwargs:
                            kwargs.update(model_kwargs[key])
                        # nested tuned config not used here (flat)
                        model = ModelClass(**kwargs)
                        n_params = sum(p.numel() for p in model.parameters())
                        print(f"  model: {ModelClass.__name__} | params: {n_params:,} | kwargs: {kwargs}")
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        print(f"  FAILED build: {e}")
                        results.append({"dataset":dataset,"horizon":horizon,"ablation":ab,"model":ab,"seed":cur_seed,"lookback":lookback,"mse":np.nan,"mae":np.nan,"mse_norm":np.nan,"mae_norm":np.nan,"val_mse":np.nan,"val_mae":np.nan,"val_mse_norm":np.nan,"val_mae_norm":np.nan,"train_time_per_epoch_s":np.nan,"peak_memory_mb":np.nan,"inference_ms_per_batch":np.nan,"epochs_run":0,"n_params":0,"batch_size":batch_size,"lr":lr,"split_convention":split_convention,"status":f"build_failed:{e}"})
                        continue
                    # handle batch override per run if needed (already batch_size)
                    cur_train_loader, cur_val_loader, cur_test_loader = train_loader, val_loader, test_loader
                    # Check if model_kwargs requested different batch (not in this simple impl)
                    try:
                        out = train_one_model(model=model, train_loader=cur_train_loader, val_loader=cur_val_loader, test_loader=cur_test_loader, scaler=split.scaler, n_epochs=n_epochs, lr=lr, weight_decay=weight_decay, optimizer_name=optimizer, scheduler_name=scheduler, scheduler_params=scheduler_params, patience=patience, min_delta=min_delta, grad_clip=grad_clip, device=device, amp=False, verbose=False)
                        status="ok"
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        print(f"  FAILED train: {e}")
                        out=None; status=f"train_failed:{e}"
                    if out is not None:
                        # save preds
                        if out.get("preds") is not None:
                            pred_dir = os.path.join(save_dir, "predictions")
                            os.makedirs(pred_dir, exist_ok=True)
                            suffix = f"_seed{cur_seed}" if len(seeds_list)>1 else ""
                            # keep dataset_H<hor>_ablation naming
                            pred_path = os.path.join(pred_dir, f"{dataset}_H{horizon}_{ab}{suffix}_pred.npy")
                            true_path = os.path.join(pred_dir, f"{dataset}_H{horizon}_{ab}{suffix}_true.npy")
                            try:
                                import numpy as np
                                np.save(pred_path, out["preds"]); np.save(true_path, out["trues"])
                                print(f"  saved: {pred_path} shape {out['preds'].shape}")
                            except Exception as e:
                                print(f"  warn save failed: {e}")
                        row = {"dataset":dataset,"horizon":horizon,"ablation":ab,"model":ab,"seed":cur_seed,"lookback":lookback,"mse":out["test_mse"],"mae":out["test_mae"],"mse_norm":out["test_mse_norm"],"mae_norm":out["test_mae_norm"],"val_mse":out["val_mse"],"val_mae":out["val_mae"],"val_mse_norm":out["val_mse_norm"],"val_mae_norm":out["val_mae_norm"],"train_time_per_epoch_s":out["train_time_per_epoch"],"peak_memory_mb":out["peak_memory_mb"],"inference_ms_per_batch":out["inference_ms_per_batch"],"epochs_run":out["epochs_run"],"n_params":out["n_params"],"batch_size":batch_size,"lr":lr,"split_convention":split_convention,"status":status}
                        print(f"  -> MSE {out['test_mse']:.4f} (norm {out['test_mse_norm']:.4f}) MAE {out['test_mae']:.4f} (norm {out['test_mae_norm']:.4f}) | {out['train_time_per_epoch']:.2f}s/epoch infer {out['inference_ms_per_batch']:.1f}ms peak {out['peak_memory_mb']:.1f}MB epochs {out['epochs_run']}/{n_epochs}")
                    else:
                        row={"dataset":dataset,"horizon":horizon,"ablation":ab,"model":ab,"seed":cur_seed,"lookback":lookback,"mse":np.nan,"mae":np.nan,"mse_norm":np.nan,"mae_norm":np.nan,"val_mse":np.nan,"val_mae":np.nan,"val_mse_norm":np.nan,"val_mae_norm":np.nan,"train_time_per_epoch_s":np.nan,"peak_memory_mb":np.nan,"inference_ms_per_batch":np.nan,"epochs_run":0,"n_params":n_params if 'n_params' in locals() else 0,"batch_size":batch_size,"lr":lr,"split_convention":split_convention,"status":status}
                    results.append(row)
                    # incremental save
                    import pandas as pd
                    pd.DataFrame(results).to_csv(os.path.join(save_dir, "ablation_results.csv"), index=False)
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

    import pandas as pd
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(["dataset","horizon","ablation","seed"]).reset_index(drop=True)
        out_csv = os.path.join(save_dir, "ablation_results.csv")
        df.to_csv(out_csv, index=False)
        print("\n"+"="*80)
        print(f"[done] Results saved to {out_csv}")
        print(f"[done] Predictions under {os.path.join(save_dir,'predictions')}")
        # summary
        try:
            from tabulate import tabulate
            cols=[c for c in ["dataset","horizon","ablation","model","seed","mse","mse_norm","mae","mae_norm","train_time_per_epoch_s","inference_ms_per_batch","peak_memory_mb"] if c in df.columns]
            print("\n=== Summary ===")
            print(tabulate(df[cols], headers="keys", tablefmt="psql", floatfmt=".4f", showindex=False))
            print("\n=== Mean per ablation (original) ===")
            print(tabulate(df.groupby("ablation")[["mse","mae"]].mean().round(4).reset_index(), headers="keys", tablefmt="psql", showindex=False))
            print("\n=== Mean normalized (TiDE Table 2) ===")
            print(tabulate(df.groupby("ablation")[["mse_norm","mae_norm"]].mean().round(4).reset_index(), headers="keys", tablefmt="psql", showindex=False))
            if len(seeds_list)>1:
                print("\n=== mean±std normalized ===")
                g=df.groupby(["dataset","horizon","ablation"])[["mse_norm","mae_norm"]].agg(["mean","std"]).round(4)
                print(g.to_string())
                g.to_csv(os.path.join(save_dir, "ablation_summary_meanstd.csv"))
        except ImportError:
            print(df.to_string())
        # write summary.txt
        with open(os.path.join(save_dir, "summary.txt"), "w") as f:
            f.write(df.to_string())
            f.write("\n\nMean per ablation (original):\n")
            f.write(str(df.groupby("ablation")[["mse","mae"]].mean()))
            f.write("\n\nMean normalized:\n")
            f.write(str(df.groupby("ablation")[["mse_norm","mae_norm"]].mean()))
    else:
        print("[warn] No results")
    print("\nAblation complete.")
    print(f"CSV: {os.path.join(save_dir,'ablation_results.csv')}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Optuna hyperparameter tuning for GATiDE benchmark – ICLR 2027 ready
====================================================================
Equal-budget tuning per model/dataset/horizon on validation mse_norm (TiDE paper §5.1:
"tune hyper-parameters using the validation set rolling validation error", Table 2 reports
test mean over 5 seeds). Requirement configs remain via configs/default.yaml.

Usage:
  pip install optuna

  # single setting – 50 trials (GATiDE vs TiDE equal budget)
  python tune_optuna.py --dataset ETTh1 --horizon 96 --model tide --n-trials 50 --epochs 100
  python tune_optuna.py --dataset ETTh1 --horizon 96 --model gatide --n-trials 50 --epochs 100

  # sweep all horizons for a dataset
  for H in 96 192 336 720; do python tune_optuna.py --dataset ETTh1 --horizon $H --model all --n-trials 50; done

  # full matrix (7 datasets × 4 horizons × 3 models × 50 trials) – use GPU
  python tune_optuna.py --dataset all --horizon all --model all --n-trials 50 --device cuda

Outputs:
  tuned_configs/{dataset}_H{horizon}_{model}_best.json  – best params
  tuned_configs/{dataset}_H{horizon}_{model}_study.db   – optuna study (sqlite)
  After tuning, merge into configs/tuned_best.yaml and run:
  python run_benchmark.py --config configs/tuned_best.yaml --seeds 0 1 2 3 4 --all-horizons --all-datasets --split-convention tide

Search spaces are defined per model to match TiDE Appendix B.3 and PatchTST/DLinear papers.
All trials use identical protocol: L=720, 7:1:2 split, StandardScaler train-only, MSE loss,
val mse_norm objective, EarlyStopping patience 10.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import yaml
from typing import Dict, Any

# Ensure benchmark package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError as e:
    print("[error] optuna not installed. Run: pip install optuna")
    raise

import numpy as np
import torch

from benchmark.datasets import load_and_split, make_loaders, discover_datasets
from benchmark.models import get_model
from benchmark.trainer import train_one_model


# ---------------------------------------------------------------------------
# Search spaces – per-model, equal-budget friendly
# ---------------------------------------------------------------------------

def sample_tide_gatide(trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
    """TiDE/GATiDE shared space – Appendix B.3 + hidden_size divisibility for GATiDE."""
    # For GATiDE, hidden must be divisible by num_heads (4)
    hidden_choices = [128, 256, 512, 1024]
    hidden_size = trial.suggest_categorical("hidden_size", hidden_choices)
    # GATiDE validation done in model, but keep choices divisible
    return {
        "hidden_size": hidden_size,
        "num_encoder_layers": trial.suggest_int("num_encoder_layers", [1, 2, 3]),
        "num_decoder_layers": trial.suggest_int("num_decoder_layers", [1, 2, 3]),
        "decoder_output_dim": trial.suggest_categorical("decoder_output_dim", [4, 8, 16, 32]),
        "temporal_decoder_hidden": trial.suggest_categorical("temporal_decoder_hidden", [32, 64, 128]),
        "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "use_layer_norm": trial.suggest_categorical("use_layer_norm", [False, True]),
        # num_attn_heads fixed 4 for gatide to keep hidden divisible; tide ignores
        "num_attn_heads": 4,
        # learning rate – log scale, TiDE default 1e-3, we search around it
        "_lr": trial.suggest_float("lr", 5e-5, 5e-3, log=True),
        "_batch_size": 512,
    }

def sample_dlinear(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "kernel_size": trial.suggest_categorical("kernel_size", [25, 51, 75]),
        "individual": trial.suggest_categorical("individual", [False, True]),
        "_lr": trial.suggest_float("lr", 5e-5, 5e-3, log=True),
        "_batch_size": 512,
    }

def sample_patchtst(trial: optuna.Trial) -> Dict[str, Any]:
    # Keep patch num reasonable for L=720: patch_len 16/24, stride 8/16
    patch_len = trial.suggest_categorical("patch_len", [16, 24])
    stride = trial.suggest_categorical("stride", [8, 16])
    # Ensure stride <= patch_len
    if stride > patch_len:
        stride = patch_len
    return {
        "patch_len": patch_len,
        "stride": stride,
        "n_layers": trial.suggest_int("n_layers", 1, 2),
        "d_model": trial.suggest_categorical("d_model", [64, 128]),
        "n_heads": 4,
        "d_ff": trial.suggest_categorical("d_ff", [128, 256]),
        "dropout": trial.suggest_categorical("dropout", [0.1, 0.2]),
        "_lr": trial.suggest_float("lr", 5e-5, 5e-3, log=True),
        "_batch_size": 512,
    }

def sample_naive(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "strategy": trial.suggest_categorical("strategy", ["last", "mean"]),
        "_lr": 1e-3,
        "_batch_size": 512,
    }


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def run_one_trial(
    dataset: str,
    horizon: int,
    model_name: str,
    params: Dict[str, Any],
    csv_dir: str,
    lookback: int = 720,
    n_epochs: int = 100,
    patience: int = 10,
    split_convention: str = "tide",
    device: str = "auto",
) -> float:
    """Train one trial and return val mse_norm (minimize)."""
    # Separate model kwargs from training overrides
    lr = params.pop("_lr", 1e-3)
    batch_size = params.pop("_batch_size", 512)

    split = load_and_split(csv_dir, dataset, lookback=lookback, horizon=horizon,
                           split_convention=split_convention, use_covariates=False)
    train_loader, val_loader, test_loader = make_loaders(split, lookback, horizon, batch_size=batch_size)

    ModelCls = get_model(model_name)
    model = ModelCls(num_features=split.n_features, lookback=lookback, horizon=horizon, **params)

    out = train_one_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler=split.scaler,
        n_epochs=n_epochs,
        lr=lr,
        weight_decay=1e-4,
        optimizer_name="adamw",
        scheduler_name="cosine",
        patience=patience,
        min_delta=1e-4,
        grad_clip=1.0,
        device=device,
        amp=False,
        verbose=False,
    )
    # TiDE tunes on validation mse_norm (normalized, Table 2)
    return float(out["val_mse_norm"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DATASETS_ALL = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "weather", "traffic"]
MODELS_ALL = ["gatide", "tide", "dlinear", "patchtst", "naive"]

def main():
    p = argparse.ArgumentParser(description="Optuna tuning for GATiDE benchmark (ICLR 2027 equal-budget)")
    p.add_argument("--csv-dir", type=str, default="E:/Machine Learning Research/GATiDE Final Verse/GATiDE/data")
    p.add_argument("--dataset", type=str, default="ETTh1", help="dataset name or 'all'")
    p.add_argument("--horizon", type=str, default="96", help="horizon int or 'all' (96,192,336,720)")
    p.add_argument("--model", type=str, default="tide", help="model name or 'all' (gatide,tide,dlinear,patchtst)")
    p.add_argument("--lookback", type=int, default=720, help="fixed L=720 per TiDE paper")
    p.add_argument("--n-trials", type=int, default=30, help="trials per setting (use 50 for paper)")
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--split-convention", type=str, default="tide", choices=["tide", "prior-work"])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42, help="study seed")
    p.add_argument("--out-dir", type=str, default="./tuned_configs")
    args = p.parse_args()

    # Resolve expands
    datasets = DATASETS_ALL if args.dataset == "all" else [args.dataset]
    if args.horizon == "all":
        horizons = [96, 192, 336, 720]
    else:
        # support comma-separated
        horizons = [int(h) for h in args.horizon.replace(",", " ").split()]
    models = MODELS_ALL if args.model == "all" else [m.strip() for m in args.model.split(",") if m.strip()]

    # Validate csv_dir
    if not os.path.isdir(args.csv_dir):
        print(f"[error] csv_dir not found: {args.csv_dir}")
        return

    # Discover available datasets if all
    try:
        available = set(discover_datasets(args.csv_dir))
        # Filter to those that exist
        datasets = [d for d in datasets if d in available or d in DATASETS_ALL]
    except Exception:
        pass

    os.makedirs(args.out_dir, exist_ok=True)

    total = len(datasets) * len(horizons) * len(models)
    print(f"[tune] Total settings: {total} ({len(datasets)} datasets × {len(horizons)} horizons × {len(models)} models) × {args.n_trials} trials each")
    print(f"[tune] Lookback {args.lookback} | Epochs {args.n_epochs} | Device {args.device} | Split {args.split_convention}")

    for dataset in datasets:
        for horizon in horizons:
            for model_name in models:
                # Skip naive tuning (no params) – just record baseline
                if model_name == "naive":
                    print(f"[skip] {dataset} H={horizon} {model_name} – no hyperparameters")
                    # Save dummy
                    best = {"strategy": "last", "batch_size": 512}
                    out_path = os.path.join(args.out_dir, f"{dataset}_H{horizon}_{model_name}_best.json")
                    with open(out_path, "w") as f:
                        json.dump({"best_params": best, "best_value": None, "n_trials": 0}, f, indent=2)
                    continue

                study_name = f"{dataset}_H{horizon}_{model_name}"
                storage = f"sqlite:///{os.path.join(args.out_dir, study_name + '_study.db')}"
                sampler = TPESampler(seed=args.seed)
                pruner = MedianPruner(n_warmup_steps=5)
                study = optuna.create_study(
                    study_name=study_name,
                    direction="minimize",
                    sampler=sampler,
                    pruner=pruner,
                    storage=storage,
                    load_if_exists=True,
                )

                # Check if already has n_trials
                if len(study.trials) >= args.n_trials:
                    print(f"[skip] {study_name} already has {len(study.trials)} trials ≥ {args.n_trials}")
                    # Ensure best json exists
                    if study.best_trial is not None:
                        best_params = {k: v for k, v in study.best_params.items() if not k.startswith("_")}
                        # Recover lr/batch
                        if "_lr" in study.best_params:
                            best_params["lr"] = study.best_params["_lr"]
                        best_params["batch_size"] = study.best_params.get("_batch_size", 512)
                        out_path = os.path.join(args.out_dir, f"{study_name}_best.json")
                        if not os.path.exists(out_path):
                            with open(out_path, "w") as f:
                                json.dump({"best_params": best_params, "best_value": study.best_value, "n_trials": len(study.trials)}, f, indent=2)
                    continue

                print(f"\n{'='*80}\n[tune] {study_name} – starting {args.n_trials - len(study.trials)} remaining trials\n{'='*80}")

                def objective(trial: optuna.Trial) -> float:
                    if model_name in ("gatide", "ga-tide", "tide"):
                        params = sample_tide_gatide(trial, model_name)
                    elif model_name == "dlinear":
                        params = sample_dlinear(trial)
                    elif model_name == "patchtst":
                        params = sample_patchtst(trial)
                    else:
                        params = sample_naive(trial)

                    # Set trial seed for determinism per trial
                    trial_seed = args.seed + trial.number
                    torch.manual_seed(trial_seed)
                    np.random.seed(trial_seed)

                    try:
                        val_mse_norm = run_one_trial(
                            dataset=dataset,
                            horizon=horizon,
                            model_name=model_name if model_name != "ga-tide" else "gatide",
                            params=dict(params),  # copy
                            csv_dir=args.csv_dir,
                            lookback=args.lookback,
                            n_epochs=args.n_epochs,
                            patience=args.patience,
                            split_convention=args.split_convention,
                            device=args.device,
                        )
                    except Exception as e:
                        print(f"  trial {trial.number} failed: {e}")
                        # Return large value to prune
                        return float("inf")

                    # Report intermediate
                    trial.set_user_attr("lr", params.get("_lr", 1e-3))
                    trial.set_user_attr("batch_size", params.get("_batch_size", 512))
                    return val_mse_norm

                # Optimize remaining trials
                n_remaining = args.n_trials - len(study.trials)
                study.optimize(objective, n_trials=n_remaining, show_progress_bar=True)

                # Save best
                best_trial = study.best_trial
                best_params_raw = best_trial.params
                # Clean private keys into final config
                best_params = {k: v for k, v in best_params_raw.items() if not k.startswith("_")}
                if "_lr" in best_params_raw:
                    best_params["lr"] = best_params_raw["_lr"]
                best_params["batch_size"] = best_params_raw.get("_batch_size", 512)

                out_path = os.path.join(args.out_dir, f"{study_name}_best.json")
                with open(out_path, "w") as f:
                    json.dump({
                        "best_params": best_params,
                        "best_value": study.best_value,
                        "best_trial_number": best_trial.number,
                        "n_trials": len(study.trials),
                        "study_name": study_name,
                        "dataset": dataset,
                        "horizon": horizon,
                        "model": model_name,
                    }, f, indent=2)

                print(f"[done] {study_name} best val mse_norm {study.best_value:.4f} params {best_params}")
                print(f"  saved {out_path}")

                # Also append to aggregated tuned_best.yaml
                tuned_yaml = os.path.join(args.out_dir, "tuned_best.yaml")
                # Load existing or create
                if os.path.exists(tuned_yaml):
                    with open(tuned_yaml) as f:
                        agg = yaml.safe_load(f) or {}
                else:
                    agg = {}
                # Structure: {dataset: {horizon: {model: params}}}
                agg.setdefault(dataset, {}).setdefault(str(horizon), {})[model_name] = best_params
                with open(tuned_yaml, "w") as f:
                    yaml.dump(agg, f, sort_keys=False)

    print(f"\n[done] All tuning complete. Best configs in {args.out_dir}/*.json and aggregated {args.out_dir}/tuned_best.yaml")
    print("Next: python run_benchmark.py --config tuned_configs/tuned_best.yaml --seeds 0 1 2 3 4 --split-convention tide")

if __name__ == "__main__":
    main()

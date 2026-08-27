"""Optuna hyperparameter search for GA-TiDE.

One constraint distinguishes this from a generic search and it matters:
`hidden_size` must be a multiple of `num_attn_heads`. `GATiDEModel.__init__`
validates this unconditionally, so sampling the two independently would raise
at construction on roughly half the trials. Here `hidden_size` is sampled as
`num_attn_heads * k`, making every point in the space valid by construction.

Tune the baseline with the SAME budget before comparing. A model that happens
to receive a better learning rate will appear to win on architecture, so either
run this script for both `--model tide` and `--model ga-tide`, or tune on one
and reuse the configuration for the other -- and state in the paper which you
did.

Usage
-----
    python scripts/tune_optuna.py --dataset ETTh1 --horizon 96 \
        --model ga-tide --n-trials 50

    python scripts/tune_optuna.py --dataset ETTh1 --horizon 96 \
        --model tide --n-trials 50

Study state persists to `optuna_studies/<name>.db`, so a search can be
resumed after an interrupted session.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import optuna
import torch
# Optuna 4.x moved this into the separate `optuna-integration` package.
try:
    from optuna.integration import PyTorchLightningPruningCallback
except ImportError:  # pragma: no cover
    try:
        from optuna_integration import PyTorchLightningPruningCallback
    except ImportError as exc:
        raise ImportError(
            "PyTorchLightningPruningCallback not found. On Optuna 4.x install "
            "the integration package: pip install optuna-integration[pytorch_lightning]"
        ) from exc
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from darts.dataprocessing.transformers import Scaler
from darts.metrics import mse

from benchmark import (  # reuse the harness so tuning and evaluation agree
    build_encoders,
    load_series,
    split_series,
    to_channel_list,
)
from darts.models import TiDEModel

from ga_tide import GATiDEModel

MODELS = {"tide": TiDEModel, "ga-tide": GATiDEModel}

STUDY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "optuna_studies")


def objective(trial: optuna.Trial, args, data) -> float:
    train_s, val_s, encoders = data

    cls = MODELS[args.model]

    # --- architecture -------------------------------------------------------
    num_attn_heads = trial.suggest_categorical("num_attn_heads", [2, 4, 8])
    # Sample hidden_size as a MULTIPLE of the head count so that every point in
    # the search space is constructible. Sampling the two independently would
    # discard roughly half the trials.
    hidden_mult = trial.suggest_int("hidden_size_mult", 8, 64, step=8)
    hidden_size = num_attn_heads * hidden_mult

    params = dict(
        input_chunk_length=args.lookback,
        output_chunk_length=args.horizon,
        hidden_size=hidden_size,
        num_encoder_layers=trial.suggest_int("num_encoder_layers", 1, 3),
        num_decoder_layers=trial.suggest_int("num_decoder_layers", 1, 3),
        decoder_output_dim=trial.suggest_categorical("decoder_output_dim", [4, 8, 16, 32]),
        temporal_decoder_hidden=trial.suggest_categorical(
            "temporal_decoder_hidden", [32, 64, 128]),
        temporal_width_past=trial.suggest_int("temporal_width_past", 0, 8),
        temporal_width_future=trial.suggest_int("temporal_width_future", 0, 8),
        dropout=trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        use_layer_norm=trial.suggest_categorical("use_layer_norm", [True, False]),
    )
    if cls is GATiDEModel:
        params["num_attn_heads"] = num_attn_heads

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.patience,
                      min_delta=1e-4, mode="min"),
        PyTorchLightningPruningCallback(trial, monitor="val_loss"),
    ]

    model = cls(
        **params,
        loss_fn=torch.nn.MSELoss(),
        optimizer_kwargs={"lr": lr},
        batch_size=batch_size,
        n_epochs=args.n_epochs,
        random_state=args.seed,
        add_encoders=encoders,
        use_reversible_instance_norm=True,
        pl_trainer_kwargs={
            "callbacks": callbacks,
            "accelerator": args.accelerator,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "gradient_clip_val": 1.0,
        },
        force_reset=True,
        save_checkpoints=False,
    )
    model.fit(series=train_s, val_series=val_s, verbose=False)

    # Objective: one-shot MSE on the validation period. The full rolling-origin
    # protocol is reserved for the test set in benchmark.py; using it here would
    # multiply the search cost for no gain in ranking quality.
    preds = model.predict(n=args.horizon, series=train_s)
    scores = [
        mse(v[: args.horizon], p) for v, p in zip(val_s, preds)
        if len(v) >= args.horizon
    ]
    score = float(np.mean(scores))

    trial.set_user_attr("hidden_size", hidden_size)
    trial.set_user_attr(
        "params",
        sum(p.numel() for p in model.model.parameters() if p.requires_grad),
    )
    return score


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--lookback", type=int, default=720)
    p.add_argument("--model", default="ga-tide", choices=list(MODELS))
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source", default="darts", choices=["darts", "csv"])
    p.add_argument("--csv-dir", default=None)
    p.add_argument("--split-convention", default="prior-work",
                   choices=["prior-work", "tide"])
    p.add_argument("--accelerator", default="auto")
    args = p.parse_args()

    # --- data (loaded once and reused across trials) ------------------------
    series = load_series(args.dataset, args.source, args.csv_dir)
    train, val, _ = split_series(series, args.dataset, args.split_convention)
    scaler = Scaler(StandardScaler(), global_fit=True)
    train_s = scaler.fit_transform(to_channel_list(train))
    val_s = scaler.transform(to_channel_list(val))
    encoders = build_encoders(freq_is_subhourly=args.dataset.startswith("ETTm"))

    os.makedirs(STUDY_DIR, exist_ok=True)
    name = f"{args.dataset}_H{args.horizon}_{args.model}"
    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{os.path.join(STUDY_DIR, name)}.db",
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(
        lambda t: objective(t, args, (train_s, val_s, encoders)),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    print("\n" + "=" * 70)
    print(f"study      : {name}")
    print(f"best value : {study.best_value:.6f}")
    print(f"parameters : {study.best_trial.user_attrs.get('params')}")
    print("best params:")
    print(json.dumps(study.best_params, indent=2))

    out = os.path.join(STUDY_DIR, f"{name}_best.json")
    with open(out, "w") as fh:
        json.dump(
            {"best_value": study.best_value,
             "best_params": study.best_params,
             "user_attrs": study.best_trial.user_attrs,
             "model": args.model},
            fh, indent=2,
        )
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()

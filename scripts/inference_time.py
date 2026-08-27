"""
Long-term forecasting benchmark for TiDE / GA-TiDE
==================================================

Compares Darts' `TiDEModel` against `GATiDEModel` on the standard long-term
forecasting datasets (ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity) under
the protocol used by Informer / Autoformer / DLinear / PatchTST / TiDE, so
that the resulting numbers are comparable to published tables.

Four protocol choices are made explicit because getting any of them wrong
silently produces numbers that cannot be compared to the literature:

1. CHANNEL INDEPENDENCE. TiDE is a *channel-independent* (global
   univariate) model: one shared set of weights is trained across all
   channels, each channel being forecast from its own past plus
   covariates. Darts' `TiDEModel` fitted on a single multivariate
   `TimeSeries` is NOT that model -- it flattens all channels into one
   encoder input and learns a joint multivariate map, which has far more
   parameters and different inductive bias. This harness therefore splits
   each multivariate dataset into a *list* of univariate series and fits
   globally over the list (`--channel-independent`, the default).

   This choice also determines whether the Layer Normalization degeneracy
   is reachable: under channel independence `output_dim == 1`, so the
   temporal decoder's normalization is over a unit-width axis. See
   `diagnostics.py`.

2. SPLITS. Prior work uses 6:2:2 for the ETT datasets (12/4/4 months) and
   7:1:2 for Weather / Electricity / Traffic. Das et al. (2023) describe
   7:1:2 for all datasets. The two conventions give different numbers on
   ETT. The default here follows prior work; `--split-convention tide`
   selects 7:1:2 everywhere. Whichever is used must be stated in the paper.

3. NORMALIZATION. Standardization (zero mean, unit variance) with
   statistics computed on the *training period only*, applied per channel.
   Metrics are reported on the standardized scale, as in all prior work.
   Note that Darts' `Scaler()` defaults to MinMaxScaler, not
   StandardScaler; the default would silently change the metric scale.

4. DATA REPAIR. The LTSF CSVs are not always on a regular time grid. The
   Weather file carries one duplicated timestamp and one 100-minute gap in an
   otherwise 10-minute series. Darts regularises the index by inserting NaN
   rows, and a single NaN inside a training window yields a NaN loss and a
   model destroyed on the first optimiser step -- surfacing downstream as NaN
   metrics rather than as an error. `load_series` therefore drops duplicate
   timestamps and interpolates absent ones, reporting both on stdout. This is
   preprocessing and belongs in the paper's experimental setup.

5. DATA SOURCE. Darts' bundled loaders are not always the same series as
   the LTSF benchmark CSVs. In particular Darts' `ElectricityDataset` is
   the 370-client LD2011_2014 set, whereas the benchmark "Electricity"
   (ECL) is a 321-client preprocessed variant; the two are not comparable.
   Prefer `--source csv` with the CSVs from the Autoformer repository for
   any number that will appear in a results table against published work.

Covariates
----------
Das et al. use time-derived features as global dynamic covariates. These
are generated here via Darts' `add_encoders`. They are not optional for
GA-TiDE: Segment Attention Fusion needs at least two input segments, and
with a univariate target and no covariates only the lookback segment
exists, in which case the fusion falls back to plain concatenation and the
`attention` arm is identical to `concat`.

Usage
-----
    # single run
    python benchmark.py --dataset ETTh1 --horizon 96 --model ga-tide --seed 0

    # both models, three seeds
    python benchmark.py --dataset ETTh1 --horizon 96 --model all --seeds 3

Results are appended to `results.csv`; re-running the same
(dataset, horizon, model, seed) overwrites that row.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, mse
from darts.utils.missing_values import fill_missing_values
from darts.models import TiDEModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from ga_tide import GATiDEModel

# --------------------------------------------------------------------------- #
# Models under comparison
# --------------------------------------------------------------------------- #
# `GATiDEModel` applies gating and segment-attention fusion together and
# exposes no switch to enable either in isolation, so the comparison here is
# between two models rather than across an ablation grid.
#
# CAVEAT for the paper: `tide` and `ga-tide` differ in THREE respects at once
# -- the gate, the segment-attention fusion (which also narrows the encoder's
# input interface), and the dropout position inside the residual block. A
# difference in accuracy is therefore not attributable to any one of them.
# Separating them requires constructor switches the model does not have.
MODELS: dict[str, type] = {
    "tide":    TiDEModel,      # baseline
    "ga-tide": GATiDEModel,    # proposed
}

# --------------------------------------------------------------------------- #
# Dataset registry
# --------------------------------------------------------------------------- #
# `n_train/n_val` are absolute row counts where prior work fixes them
# (the ETT family); `None` means use the fractional 7:1:2 split.
@dataclass(frozen=True)
class DatasetSpec:
    darts_loader: str
    csv_name: str
    date_col: str
    n_train: Optional[int]
    n_val: Optional[int]
    n_test: Optional[int]


DATASETS: dict[str, DatasetSpec] = {
    # ETTh*: 12 / 4 / 4 months at hourly resolution
    "ETTh1": DatasetSpec("ETTh1Dataset", "ETTh1.csv", "date", 8640, 2880, 2880),
    "ETTh2": DatasetSpec("ETTh2Dataset", "ETTh2.csv", "date", 8640, 2880, 2880),
    # ETTm*: same months at 15-minute resolution
    "ETTm1": DatasetSpec("ETTm1Dataset", "ETTm1.csv", "date", 34560, 11520, 11520),
    "ETTm2": DatasetSpec("ETTm2Dataset", "ETTm2.csv", "date", 34560, 11520, 11520),
    # Fractional 7:1:2
    "Weather":     DatasetSpec("WeatherDataset", "weather.csv", "date", None, None, None),
    "Electricity": DatasetSpec("ElectricityDataset", "electricity.csv", "date", None, None, None),
}

HORIZONS = (96, 192, 336, 720)


# --------------------------------------------------------------------------- #
# Data loading and splitting
# --------------------------------------------------------------------------- #
def load_series(name: str, source: str, csv_dir: Optional[str]) -> TimeSeries:
    """Return the full multivariate series for `name`."""
    spec = DATASETS[name]

    if source == "csv":
        if csv_dir is None:
            raise ValueError("--csv-dir is required when --source csv")
        path = os.path.join(csv_dir, spec.csv_name)
        df = pd.read_csv(path)
        df[spec.date_col] = pd.to_datetime(df[spec.date_col])

        # The LTSF benchmark CSVs are not always on a perfectly regular grid.
        # The Weather file, for instance, contains one duplicated timestamp and
        # one 100-minute gap in an otherwise 10-minute series. Both must be
        # handled BEFORE the series reaches the model:
        #
        #   * a duplicated timestamp prevents Darts from inferring a frequency;
        #   * `fill_missing_dates=True` regularises the index by inserting NaN
        #     rows, and a single NaN inside a training window produces a NaN
        #     loss, NaN gradients, and a model destroyed on the first optimiser
        #     step -- which surfaces downstream as NaN metrics rather than as
        #     an error.
        #
        # Both repairs are reported on stdout so that any preprocessing applied
        # to a dataset is visible in the run log and can be stated in writing.
        n_dupes = int(df[spec.date_col].duplicated().sum())
        if n_dupes:
            df = df.drop_duplicates(subset=spec.date_col, keep="first")
            print(f"[data] {spec.csv_name}: dropped {n_dupes} duplicated "
                  f"timestamp(s), keeping the first occurrence.")

        df = df.sort_values(spec.date_col)
        value_cols = [c for c in df.columns if c != spec.date_col]
        series = TimeSeries.from_dataframe(
            df, time_col=spec.date_col, value_cols=value_cols,
            fill_missing_dates=True, freq=None,
        ).astype(np.float32)

        n_missing = int(np.isnan(series.values()).sum())
        if n_missing:
            n_rows = int(np.isnan(series.values()).any(axis=1).sum())
            series = fill_missing_values(series, fill="auto")
            remaining = int(np.isnan(series.values()).sum())
            print(f"[data] {spec.csv_name}: {n_rows} timestamp(s) were absent "
                  f"from the grid ({n_missing} values); filled by linear "
                  f"interpolation.")
            if remaining:
                raise ValueError(
                    f"{spec.csv_name}: {remaining} values are still NaN after "
                    "interpolation (missing values at a series boundary cannot "
                    "be interpolated). Inspect the file before proceeding."
                )
        return series

    # Darts bundled loader
    import darts.datasets as dd
    try:
        series = getattr(dd, spec.darts_loader)().load().astype(np.float32)
    except ValueError as exc:
        # Darts' own loaders call `TimeSeries.from_dataframe(...,
        # fill_missing_dates=True)`, which reindexes onto a regular grid. If the
        # underlying file carries a duplicated timestamp -- as the 10-minute
        # Weather series does -- pandas refuses to reindex and raises
        # "cannot reindex on an axis with duplicate labels" from inside the
        # library, where this script cannot repair it. The CSV path below does
        # the repair itself, so route the user there rather than leaving them
        # with a pandas error and no context.
        raise RuntimeError(
            f"Darts' bundled loader for '{name}' failed while regularising the "
            f"time index ({type(exc).__name__}: {exc}).\n"
            "  This is a property of the source file, not of the model. The "
            "usual cause is a duplicated timestamp, which the `--source csv` "
            "path in this script detects, repairs and reports.\n"
            "  Fix: download the LTSF benchmark CSVs (see data/README.md) and "
            f"re-run with:\n"
            f"      --source csv --csv-dir <directory containing {spec.csv_name}>\n"
            "  This is also the recommended source for any number reported "
            "against published results."
        ) from exc
    if name == "Electricity":
        print(
            "[warn] Darts' ElectricityDataset is LD2011_2014 (370 clients). The "
            "LTSF benchmark 'Electricity' (ECL) is a 321-client preprocessed "
            "variant; results will NOT be comparable to published ECL numbers. "
            "Use --source csv for comparability."
        )
    return series


def split_series(
    series: TimeSeries, name: str, convention: str
) -> tuple[TimeSeries, TimeSeries, TimeSeries]:
    """Chronological train/validation/test split.

    `convention='prior-work'` uses the fixed month counts for the ETT family
    and 7:1:2 elsewhere; `convention='tide'` uses 7:1:2 throughout.
    """
    spec = DATASETS[name]
    n = len(series)

    if convention == "prior-work" and spec.n_train is not None:
        n_tr, n_va, n_te = spec.n_train, spec.n_val, spec.n_test
        if n < n_tr + n_va + n_te:
            raise ValueError(
                f"{name}: expected at least {n_tr + n_va + n_te} rows, found {n}."
            )
    else:
        n_tr = int(0.7 * n)
        n_va = int(0.1 * n)
        n_te = n - n_tr - n_va

    train = series[:n_tr]
    val = series[n_tr : n_tr + n_va]
    test = series[n_tr + n_va : n_tr + n_va + n_te]
    return train, val, test


def to_channel_list(series: TimeSeries) -> list[TimeSeries]:
    """Split a multivariate series into one univariate series per component.

    This is what makes the model channel-independent: Darts trains a single
    global model over the list, sharing weights across channels.
    """
    return [series[c] for c in series.components]


# --------------------------------------------------------------------------- #
# Covariate encoders
# --------------------------------------------------------------------------- #
def build_encoders(freq_is_subhourly: bool) -> dict:
    """Time-derived global covariates, following Das et al.

    Produces both future covariates (calendar features, known in advance) and
    a past covariate (relative position), so that Segment Attention Fusion has
    three tokens to attend over: lookback, past covariates, future covariates.
    The encoder outputs are standardized with training-period statistics.
    """
    future_cyclic = ["hour", "dayofweek", "month"]
    if freq_is_subhourly:
        future_cyclic = ["minute"] + future_cyclic
    return {
        "cyclic": {"future": future_cyclic},
        "datetime_attribute": {"future": ["dayofyear"]},
        "position": {"past": ["relative"]},
        "transformer": Scaler(StandardScaler()),
    }


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def build_model(
    model_name: str,
    lookback: int,
    horizon: int,
    seed: int,
    encoders: dict,
    hidden_size: int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    decoder_output_dim: int,
    temporal_decoder_hidden: int,
    temporal_width_past: int,
    temporal_width_future: int,
    dropout: float,
    use_layer_norm: bool,
    lr: float,
    batch_size: int,
    n_epochs: int,
    patience: int,
    num_attn_heads: int,
    use_rin: bool,
    accelerator: str,
    precision: str = "32-true",
    max_train_batches: Optional[int] = None,
) -> GATiDEModel:
    cls = MODELS[model_name]

    # GATiDEModel validates this itself, but failing here keeps an invalid
    # sweep configuration from consuming a full data-loading cycle first.
    if cls is GATiDEModel and hidden_size % num_attn_heads != 0:
        raise ValueError(
            f"hidden_size={hidden_size} is not divisible by "
            f"num_attn_heads={num_attn_heads}."
        )

    stopper = EarlyStopping(
        monitor="val_loss", patience=patience, min_delta=1e-4, mode="min"
    )

    extra = {"num_attn_heads": num_attn_heads} if cls is GATiDEModel else {}

    return cls(
        input_chunk_length=lookback,
        output_chunk_length=horizon,
        # --- architecture (held fixed across arms) ---
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        decoder_output_dim=decoder_output_dim,
        hidden_size=hidden_size,
        temporal_width_past=temporal_width_past,
        temporal_width_future=temporal_width_future,
        temporal_decoder_hidden=temporal_decoder_hidden,
        use_layer_norm=use_layer_norm,
        dropout=dropout,
        use_static_covariates=False,
        **extra,
        # --- training ---
        loss_fn=torch.nn.MSELoss(),
        optimizer_kwargs={"lr": lr},
        batch_size=batch_size,
        n_epochs=n_epochs,
        random_state=seed,
        add_encoders=encoders,
        use_reversible_instance_norm=use_rin,
        pl_trainer_kwargs={
            "callbacks": [stopper],
            "accelerator": accelerator,
            "enable_progress_bar": True,
            "enable_model_summary": False,
            "gradient_clip_val": 1.0,
            # Mixed precision roughly halves the time per step on any GPU with
            # tensor cores, at negligible cost in forecast accuracy for these
            # models. Apply it to BOTH arms or not at all: a precision
            # difference between arms would confound the comparison.
            "precision": precision,
            # Capping optimiser steps per epoch makes a very large dataset
            # tractable, but it REDEFINES the epoch: the model no longer sees
            # the training set once per epoch. The cap is written into the
            # results row so the definition travels with the numbers.
            **({} if max_train_batches is None
               else {"limit_train_batches": max_train_batches}),
        },
        force_reset=True,
        save_checkpoints=False,
    )


# Keys in an Optuna `*_best.json` that map onto build_model() arguments.
TUNABLE = (
    "num_encoder_layers", "num_decoder_layers", "decoder_output_dim",
    "temporal_decoder_hidden", "temporal_width_past", "temporal_width_future",
    "dropout", "use_layer_norm", "lr", "batch_size", "num_attn_heads",
)


def load_config(path: str, expect_model: str) -> dict:
    """Read a tuned configuration written by tune_optuna.py.

    Transcribing a dozen hyperparameters by hand across every dataset and
    horizon is the kind of error that is invisible until someone tries to
    reproduce the table, so the search output is consumed directly.

    `hidden_size` is not sampled directly: the search samples
    `num_attn_heads` and `hidden_size_mult` and takes their product, so that
    every point satisfies `hidden_size % num_attn_heads == 0`. The realised
    value is recorded under `user_attrs`; it is recomputed here as a
    cross-check.
    """
    with open(path) as fh:
        blob = json.load(fh)

    if blob.get("model") and blob["model"] != expect_model:
        raise ValueError(
            f"{path} is a configuration for '{blob['model']}' but was passed "
            f"for '{expect_model}'. Using the wrong model's hyperparameters "
            "would silently invalidate the comparison."
        )

    best = blob.get("best_params", {})
    cfg = {k: best[k] for k in TUNABLE if k in best}

    heads = best.get("num_attn_heads")
    mult = best.get("hidden_size_mult")
    if heads is not None and mult is not None:
        cfg["hidden_size"] = heads * mult
    recorded = blob.get("user_attrs", {}).get("hidden_size")
    if recorded is not None:
        if "hidden_size" in cfg and cfg["hidden_size"] != recorded:
            raise ValueError(
                f"{path}: hidden_size mismatch -- num_attn_heads * "
                f"hidden_size_mult = {cfg['hidden_size']} but user_attrs "
                f"records {recorded}."
            )
        cfg["hidden_size"] = recorded

    if "hidden_size" not in cfg:
        raise ValueError(f"{path}: could not determine hidden_size.")
    return cfg


def count_parameters(model) -> int:
    """Trainable parameter count. Only valid after `fit()`, since Darts
    instantiates the network lazily from the first training sample."""
    if model.model is None:
        return -1
    return sum(p.numel() for p in model.model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    model,
    train_scaled: list[TimeSeries],
    val_scaled: list[TimeSeries],
    test_scaled: list[TimeSeries],
    horizon: int,
    lookback: int,
    stride: int,
) -> tuple[float, float, float, int]:
    """Rolling-origin evaluation over the test period.

    Prior work slides a window of length `lookback` with stride 1 across the
    test set, forecasts `horizon` steps from each origin, and averages the
    error over all origins and channels. `stride > 1` subsamples the origins;
    use it for quick iteration only, never for reported numbers.

    The series passed to `historical_forecasts` is train+val+test so that the
    first test origin has a full lookback available; `start` places the first
    forecast at the beginning of the test period.

    Returns (mse, mae, inference_seconds, n_origins_scored). `inference_seconds`
    times ONLY `historical_forecasts` -- the actual forward passes -- not the
    numpy metric reduction below, which is comparatively negligible and would
    otherwise dilute a wall-clock number meant to describe model inference.
    Note this is total time over ALL scored origins at the given `stride`, not
    a per-forecast figure; `stride` changes the origin count, so a run's
    `inference_seconds` is only comparable to another run at the same stride
    unless you also divide by `n_origins_scored`.
    """
    full = [
        tr.append(va).append(te)
        for tr, va, te in zip(train_scaled, val_scaled, test_scaled)
    ]
    start = len(train_scaled[0]) + len(val_scaled[0])

    t0 = time.time()
    forecasts = model.historical_forecasts(
        series=full,
        start=start,
        start_format="position",
        forecast_horizon=horizon,
        stride=stride,
        retrain=False,
        last_points_only=False,
        verbose=True,
        show_warnings=False,
    )
    inference_seconds = time.time() - t0

    # Metrics are computed on raw arrays rather than per forecast through
    # Darts' metric functions. The two are numerically identical -- every
    # forecast has the same horizon and component count, so averaging
    # per-forecast means equals the overall mean -- but the array form avoids
    # building two TimeSeries objects and running two xarray reductions for
    # each origin. On Weather at stride 1 that is ~219k origins per run, where
    # the Python-level loop costs more than the forward passes it scores.
    mse_tot = mae_tot = 0.0
    n_scored = 0

    for series, per_origin in zip(full, forecasts):
        vals = series.values()                      # (T, C)
        # Position lookup for the first timestamp of each forecast. Built once
        # per series so that mapping a forecast back onto the target is O(1)
        # and does not depend on assuming a particular origin spacing.
        pos_of = {t: i for i, t in enumerate(series.time_index)}

        preds, actuals = [], []
        for fc in per_origin:
            p0 = pos_of.get(fc.time_index[0])
            if p0 is None or p0 + horizon > len(vals):
                continue                            # forecast runs off the end
            preds.append(fc.values())
            actuals.append(vals[p0 : p0 + horizon])

        if not preds:
            continue

        pred_arr = np.asarray(preds, dtype=np.float64)     # (n, H, C)
        act_arr = np.asarray(actuals, dtype=np.float64)
        err = act_arr - pred_arr
        # Sum of per-forecast means, accumulated so that origins are weighted
        # uniformly across series regardless of how many each contributes.
        mse_tot += float(np.mean(err ** 2, axis=(1, 2)).sum())
        mae_tot += float(np.mean(np.abs(err), axis=(1, 2)).sum())
        n_scored += len(preds)

    if n_scored == 0:
        raise ValueError(
            "No forecast origin could be scored. Check that the test split is "
            "longer than the forecast horizon."
        )
    return mse_tot / n_scored, mae_tot / n_scored, inference_seconds, n_scored


# --------------------------------------------------------------------------- #
# One experiment
# --------------------------------------------------------------------------- #
def run_one(args, dataset: str, horizon: int, model_name: str, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    series = load_series(dataset, args.source, args.csv_dir)
    train, val, test = split_series(series, dataset, args.split_convention)

    # Standardize on the training period only. Darts' Scaler defaults to
    # MinMaxScaler, so StandardScaler is passed explicitly.
    scaler = Scaler(StandardScaler(), global_fit=True)

    if args.channel_independent:
        train_l, val_l, test_l = (to_channel_list(s) for s in (train, val, test))
    else:
        train_l, val_l, test_l = [train], [val], [test]

    train_s = scaler.fit_transform(train_l)
    val_s = scaler.transform(val_l)
    test_s = scaler.transform(test_l)

    subhourly = dataset.startswith("ETTm")
    encoders = build_encoders(freq_is_subhourly=subhourly)

    # Defaults from the command line, then the tuned configuration for THIS
    # model on top. Each model keeps its own architecture, which is the point
    # of tuning per model; report parameter counts alongside the metrics.
    arch = dict(
        hidden_size=args.hidden_size,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        decoder_output_dim=args.decoder_output_dim,
        temporal_decoder_hidden=args.temporal_decoder_hidden,
        temporal_width_past=args.temporal_width_past,
        temporal_width_future=args.temporal_width_future,
        dropout=args.dropout,
        use_layer_norm=args.use_layer_norm,
        lr=args.lr,
        batch_size=args.batch_size,
        num_attn_heads=args.num_attn_heads,
    )
    cfg_path = getattr(args, "configs", {}).get(model_name)
    if cfg_path:
        arch.update(load_config(cfg_path, model_name))

    model = build_model(
        model_name=model_name,
        lookback=args.lookback,
        horizon=horizon,
        seed=seed,
        encoders=encoders,
        n_epochs=args.n_epochs,
        patience=args.patience,
        use_rin=args.use_rin,
        accelerator=args.accelerator,
        precision=args.precision,
        max_train_batches=args.max_train_batches,
        **arch,
    )

    # Darts moved dataloader options behind `dataloader_kwargs`; pass them only
    # if this version accepts them, rather than assuming the signature.
    fit_kwargs = {}
    if args.num_workers and "dataloader_kwargs" in inspect.signature(model.fit).parameters:
        fit_kwargs["dataloader_kwargs"] = {
            "num_workers": args.num_workers, "persistent_workers": True
        }

    t0 = time.time()
    model.fit(series=train_s, val_series=val_s, verbose=False, **fit_kwargs)
    fit_seconds = time.time() - t0
    epochs_run = int(model.trainer.current_epoch) if model.trainer else args.n_epochs

    test_mse, test_mae, inference_seconds, n_origins = evaluate(
        model, train_s, val_s, test_s, horizon, args.lookback, args.stride
    )

    return {
        "dataset": dataset,
        "horizon": horizon,
        "model": model_name,
        "seed": seed,
        "mse": test_mse,
        "mae": test_mae,
        "params": count_parameters(model),
        "epochs": epochs_run,
        "fit_seconds": round(fit_seconds, 1),
        "sec_per_epoch": round(fit_seconds / max(epochs_run, 1), 2),
        "inference_seconds": round(inference_seconds, 2),
        "n_forecast_origins": n_origins,
        "ms_per_origin": round(1000 * inference_seconds / max(n_origins, 1), 4),
        "lookback": args.lookback,
        "hidden_size": arch["hidden_size"],
        "use_layer_norm": arch["use_layer_norm"],
        "dropout": arch["dropout"],
        "lr": arch["lr"],
        "batch_size": arch["batch_size"],
        "num_attn_heads": arch["num_attn_heads"] if model_name == "ga-tide" else None,
        "num_encoder_layers": arch["num_encoder_layers"],
        "num_decoder_layers": arch["num_decoder_layers"],
        "decoder_output_dim": arch["decoder_output_dim"],
        "config_source": os.path.basename(cfg_path) if cfg_path else "cli-defaults",
        "max_train_batches": args.max_train_batches,
        "channel_independent": args.channel_independent,
        "split_convention": args.split_convention,
        "stride": args.stride,
    }


# --------------------------------------------------------------------------- #
# Results table
# --------------------------------------------------------------------------- #
KEY_COLS = ["dataset", "horizon", "model", "seed"]

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "results.csv",
)


def already_done(key: dict, path: str) -> bool:
    """True if a row with this (dataset, horizon, model, seed) is already on
    disk. Used by --skip-existing so an interrupted sweep can be resumed
    without repeating completed runs -- essential on Colab/Kaggle, where
    sessions disconnect long before a full grid finishes."""
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if df.empty or not all(c in df.columns for c in KEY_COLS):
        return False
    mask = np.ones(len(df), dtype=bool)
    for c in KEY_COLS:
        mask &= df[c].astype(str) == str(key[c])
    return bool(mask.any())


def append_result(row: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df_new = pd.DataFrame([row])
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Overwrite any existing row with the same key rather than duplicating.
        mask = np.ones(len(df), dtype=bool)
        for c in KEY_COLS:
            mask &= df[c].astype(str) == str(row[c])
        df = pd.concat([df[~mask], df_new], ignore_index=True)
    else:
        df = df_new
    df.sort_values(KEY_COLS).to_csv(path, index=False)


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # what to run
    p.add_argument("--dataset", default="ETTh1",
                   choices=list(DATASETS) + ["all"])
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--all-horizons", action="store_true")
    p.add_argument("--model", default="ga-tide",
                   help="model name, a comma-separated list, or 'all' "
                        f"(available: {list(MODELS)})")
    p.add_argument("--seeds", type=int, default=1,
                   help="number of seeds, run as 0..seeds-1")
    p.add_argument("--seed", type=int, default=None,
                   help="run a single specific seed instead")

    # protocol
    p.add_argument("--source", default="darts", choices=["darts", "csv"])
    p.add_argument("--csv-dir", default=None,
                   help="directory of LTSF benchmark CSVs (Autoformer repo)")
    p.add_argument("--split-convention", default="prior-work",
                   choices=["prior-work", "tide"])
    p.add_argument("--channel-independent", action="store_true", default=True)
    p.add_argument("--multivariate", dest="channel_independent",
                   action="store_false",
                   help="fit one joint multivariate model (NOT the TiDE protocol)")
    p.add_argument("--stride", type=int, default=1,
                   help="evaluation origin stride; keep at 1 for reported numbers")

    # architecture (held fixed across arms)
    p.add_argument("--lookback", type=int, default=720)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-encoder-layers", type=int, default=2)
    p.add_argument("--num-decoder-layers", type=int, default=2)
    p.add_argument("--decoder-output-dim", type=int, default=8)
    p.add_argument("--temporal-decoder-hidden", type=int, default=128)
    p.add_argument("--temporal-width-past", type=int, default=4)
    p.add_argument("--temporal-width-future", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--use-layer-norm", action="store_true", default=True)
    p.add_argument("--no-layer-norm", dest="use_layer_norm", action="store_false")
    p.add_argument("--num-attn-heads", type=int, default=4)
    p.add_argument("--config-tide", default=None,
                   help="path to a tune_optuna.py *_best.json for the TiDE arm")
    p.add_argument("--config-ga-tide", default=None,
                   help="path to a tune_optuna.py *_best.json for the GA-TiDE arm")
    p.add_argument("--use-rin", action="store_true", default=True)

    # training
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-train-batches", type=int, default=None,
                   help="cap optimiser steps per epoch. Makes a large dataset "
                        "tractable, but an epoch becomes a fixed number of "
                        "steps rather than a pass over the training set. "
                        "Raise --n-epochs to compensate, and state the cap in "
                        "any write-up.")
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--precision", default="32-true",
                   choices=["32-true", "16-mixed", "bf16-mixed"],
                   help="Lightning precision. '16-mixed' is typically ~1.5-2x "
                        "faster on a tensor-core GPU. Use the SAME value for "
                        "every arm of a comparison.")
    p.add_argument("--num-workers", type=int, default=0,
                   help="dataloader worker processes (Kaggle/Colab have ~2-4 "
                        "CPU cores; 2 is usually enough to keep the GPU fed)")

    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--skip-existing", action="store_true",
                   help="skip (dataset, horizon, model, seed) combinations "
                        "already present in --out; use to resume an "
                        "interrupted sweep")
    args = p.parse_args()

    args.configs = {}
    if args.config_tide:
        args.configs["tide"] = args.config_tide
    if args.config_ga_tide:
        args.configs["ga-tide"] = args.config_ga_tide
    for m, path in args.configs.items():
        if not os.path.exists(path):
            p.error(f"--config for '{m}': no such file: {path}")
        print(f"{m:>8}: {load_config(path, m)}")

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    horizons = list(HORIZONS) if args.all_horizons else [args.horizon]
    if args.model == "all":
        models = list(MODELS)
    else:
        models = [m.strip() for m in args.model.split(",") if m.strip()]
        unknown = [m for m in models if m not in MODELS]
        if unknown:
            p.error(f"unknown model(s) {unknown}; choose from {list(MODELS)} or 'all'")
    seeds = [args.seed] if args.seed is not None else list(range(args.seeds))

    total = len(datasets) * len(horizons) * len(models) * len(seeds)
    print(f"Planned runs: {total}\n" + "-" * 60)

    i = 0
    for ds in datasets:
        for h in horizons:
            for model_name in models:
                for seed in seeds:
                    i += 1
                    tag = f"[{i}/{total}] {ds} H={h} model={model_name} seed={seed}"
                    key = {"dataset": ds, "horizon": h,
                           "model": model_name, "seed": seed}
                    if args.skip_existing and already_done(key, args.out):
                        print(tag + "  -- already done, skipped", flush=True)
                        continue
                    print(tag, flush=True)
                    try:
                        row = run_one(args, ds, h, model_name, seed)
                    except Exception as exc:  # keep the sweep alive
                        print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
                        continue
                    print("  " + json.dumps(
                        {k: row[k] for k in
                         ("mse", "mae", "params", "epochs", "sec_per_epoch",
                          "inference_seconds", "ms_per_origin")}
                    ), flush=True)
                    append_result(row, args.out)

    print("-" * 60)
    if os.path.exists(args.out):
        df = pd.read_csv(args.out)
        summary = (
            df.groupby(["dataset", "horizon", "model"])
              .agg(mse_mean=("mse", "mean"), mse_std=("mse", "std"),
                   mae_mean=("mae", "mean"), mae_std=("mae", "std"),
                   params=("params", "first"),
                   sec_per_epoch=("sec_per_epoch", "mean"),
                   inference_seconds=("inference_seconds", "mean"),
                   ms_per_origin=("ms_per_origin", "mean"))
              .round(4)
        )
        print(summary.to_string())


if __name__ == "__main__":
    main()

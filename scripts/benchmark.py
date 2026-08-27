"""
Long-term forecasting benchmark for TiDE / GA-TiDE
==================================================

Evaluates the ablation grid of `ga_tide.py` on the standard long-term
forecasting datasets (ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity)
under the protocol used by Informer / Autoformer / DLinear / PatchTST /
TiDE, so that the resulting numbers are comparable to published tables.

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

4. DATA SOURCE. Darts' bundled loaders are not always the same series as
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
        value_cols = [c for c in df.columns if c != spec.date_col]
        return TimeSeries.from_dataframe(
            df, time_col=spec.date_col, value_cols=value_cols,
            fill_missing_dates=True, freq=None,
        ).astype(np.float32)

    # Darts bundled loader
    import darts.datasets as dd
    series = getattr(dd, spec.darts_loader)().load().astype(np.float32)
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
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "gradient_clip_val": 1.0,
        },
        force_reset=True,
        save_checkpoints=False,
    )


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
) -> tuple[float, float]:
    """Rolling-origin evaluation over the test period.

    Prior work slides a window of length `lookback` with stride 1 across the
    test set, forecasts `horizon` steps from each origin, and averages the
    error over all origins and channels. `stride > 1` subsamples the origins;
    use it for quick iteration only, never for reported numbers.

    The series passed to `historical_forecasts` is train+val+test so that the
    first test origin has a full lookback available; `start` places the first
    forecast at the beginning of the test period.
    """
    full = [
        tr.append(va).append(te)
        for tr, va, te in zip(train_scaled, val_scaled, test_scaled)
    ]
    start = len(train_scaled[0]) + len(val_scaled[0])

    forecasts = model.historical_forecasts(
        series=full,
        start=start,
        start_format="position",
        forecast_horizon=horizon,
        stride=stride,
        retrain=False,
        last_points_only=False,
        verbose=False,
        show_warnings=False,
    )

    # `forecasts` is a list (one entry per series) of lists (one per origin).
    # Metrics are averaged uniformly over origins and channels, matching the
    # convention in which published MSE/MAE tables are computed.
    mse_vals, mae_vals = [], []
    for series, per_origin in zip(full, forecasts):
        for fc in per_origin:
            actual = series.slice_intersect(fc)
            if len(actual) < horizon:
                continue
            mse_vals.append(mse(actual, fc))
            mae_vals.append(mae(actual, fc))

    return float(np.mean(mse_vals)), float(np.mean(mae_vals))


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

    model = build_model(
        model_name=model_name,
        lookback=args.lookback,
        horizon=horizon,
        seed=seed,
        encoders=encoders,
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
        n_epochs=args.n_epochs,
        patience=args.patience,
        num_attn_heads=args.num_attn_heads,
        use_rin=args.use_rin,
        accelerator=args.accelerator,
    )

    t0 = time.time()
    model.fit(series=train_s, val_series=val_s, verbose=False)
    fit_seconds = time.time() - t0
    epochs_run = int(model.trainer.current_epoch) if model.trainer else args.n_epochs

    test_mse, test_mae = evaluate(
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
        "lookback": args.lookback,
        "hidden_size": args.hidden_size,
        "use_layer_norm": args.use_layer_norm,
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
    p.add_argument("--use-rin", action="store_true", default=True)

    # training
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--accelerator", default="auto")

    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

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
                    print(tag, flush=True)
                    try:
                        row = run_one(args, ds, h, model_name, seed)
                    except Exception as exc:  # keep the sweep alive
                        print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
                        continue
                    print("  " + json.dumps(
                        {k: row[k] for k in
                         ("mse", "mae", "params", "epochs", "sec_per_epoch")}
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
                   sec_per_epoch=("sec_per_epoch", "mean"))
              .round(4)
        )
        print(summary.to_string())


if __name__ == "__main__":
    main()

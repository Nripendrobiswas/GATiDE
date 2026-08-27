"""
Data ingestion & preprocessing
============================
- Loads time-series CSVs from data/ (ETTh1, ETTh2, ETTm1, ETTm2, weather, electricity, traffic).
- Auto-discovers value columns (all except 'date' / 'Date' / 'date_col').
- Sequential split with TiDE-paper faithful options (Das et al. 2023, TMLR, §5.1):
    * --split-convention tide (default, paper): 7:1:2 for all 7 datasets (70%/10%/20%)
      -> matches TiDE Table 2 protocol: "train:validation:test ratio is 7:1:2 as dictated by prior work"
    * --split-convention prior-work: 6:2:2 for ETT family (12/4/4 months) and 7:1:2 elsewhere
      -> matches Informer/Autoformer/DLinear convention used in prior GATiDE scripts/benchmark.py
  Requirement prompt specifies 70/10/20, which equals tide (7:1:2). Both are supported.
- Standardize with train-only statistics (StandardScaler) to prevent leakage.
  TiDE paper: "all the experiments are performed on standard normalized datasets (using the mean and
  the standard deviations in the training period)" – i.e. metrics in Table 2 are on normalized scale.
  Requirement asks for inverse-scaled metrics – benchmark now reports BOTH (see trainer.py).
- Creates sliding-window Dataset with fixed lookback L=720 (TiDE always 720, §5.1) and horizon H.
- Optional time-derived covariates (hour, dayofweek, month, etc.) as in TiDE §5.1:
  "As global dynamic covariates, we use simple time-derived features like minute of the hour, hour of
  the day, day of the week etc which are normalized similar to Alexandrov et al. 2020". When enabled,
  GATiDE's SegmentAttentionFusion receives ≥2 segments and attention is active; without covariates it
  falls back to single-segment (still valid, but attention arm is identical to concat).

Supports both generic CSVs (date,col1,col2,...) and the LTSF benchmark layout.
"""
from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from benchmark.utils.scaler import StandardScaler


# ---------------------------------------------------------------------------
# CSV discovery helpers
# ---------------------------------------------------------------------------

CSV_CANDIDATES = ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv",
                  "weather.csv", "electricity.csv", "traffic.csv", "Weather.csv"]


def discover_datasets(csv_dir: str) -> List[str]:
    """Return dataset names (stem without .csv) found in csv_dir."""
    if not os.path.isdir(csv_dir):
        raise FileNotFoundError(f"csv_dir not found: {csv_dir}")
    csvs = glob.glob(os.path.join(csv_dir, "*.csv"))
    names = [os.path.splitext(os.path.basename(p))[0] for p in csvs]
    # normalize lower-case traffic vs Traffic etc, but keep original stem
    return sorted(names)


def _resolve_csv(csv_dir: str, dataset: str) -> str:
    """Find CSV file for dataset name (case-insensitive stem)."""
    # exact
    p = os.path.join(csv_dir, f"{dataset}.csv")
    if os.path.exists(p):
        return p
    # case-insensitive
    for f in os.listdir(csv_dir):
        if f.lower() == f"{dataset.lower()}.csv":
            return os.path.join(csv_dir, f)
    raise FileNotFoundError(f"CSV for dataset '{dataset}' not found in {csv_dir}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TimeSeriesWindowDataset(Dataset):
    """Sliding window dataset.

    Given a 2-D array data of shape (T, C) already scaled,
    yields (x, y) where x: (L, C), y: (H, C).
    """

    def __init__(self, data: np.ndarray, lookback: int, horizon: int):
        assert data.ndim == 2, f"data must be (T,C), got {data.shape}"
        self.data = data.astype(np.float32)
        self.L = lookback
        self.H = horizon
        self.n_samples = len(data) - lookback - horizon + 1
        if self.n_samples <= 0:
            raise ValueError(
                f"Series too short for L={lookback}, H={horizon}: T={len(data)}, "
                f"need >= {lookback+horizon}, got {self.n_samples} samples"
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx: idx + self.L]            # (L, C)
        y = self.data[idx + self.L: idx + self.L + self.H]  # (H, C)
        return torch.from_numpy(x), torch.from_numpy(y)


# TiDE-paper dataset registry for split convention handling
# ETT family uses fixed month counts in prior-work convention (6:2:2), otherwise 7:1:2
ETT_FAMILY = {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}

def _split_indices(n: int, dataset: str, convention: str) -> tuple[int, int, int]:
    """Return (n_train, n_val, n_test) per convention.
    tide: 7:1:2 for all datasets (TiDE paper §5.1)
    prior-work: 6:2:2 for ETT (12/4/4 months at hourly / 15-min), 7:1:2 elsewhere
    """
    if convention == "tide":
        n_train = int(0.7 * n)
        n_val = int(0.1 * n)
        n_test = n - n_train - n_val
        return n_train, n_val, n_test
    # prior-work
    if dataset in ETT_FAMILY:
        # Fixed counts from GATiDE scripts/benchmark.py DATASETS spec
        # ETTh*: 8640/2880/2880 at hourly, ETTm*: 34560/11520/11520 at 15-min
        if dataset.startswith("ETTh"):
            return 8640, 2880, 2880
        else:  # ETTm
            return 34560, 11520, 11520
    n_train = int(0.7 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def _make_time_covariates(dates: pd.Series, freq_is_subhourly: bool = False) -> Optional[np.ndarray]:
    """Generate TiDE-style global dynamic time covariates (hour, dayofweek, month, dayofyear, + minute if subhourly).
    Returns (T, F) standardized features or None if dates missing. Mirrors GATiDE scripts/benchmark.py build_encoders.
    """
    if dates is None or dates.isna().all():
        return None
    df = pd.DataFrame({"date": pd.to_datetime(dates)})
    feats = []
    # cyclic encodings not strictly needed for simple linear – use raw normalized ints for transparency
    # TiDE uses cyclic + datetime_attribute + position; here we generate straightforward normalized time features
    feats.append(df["date"].dt.hour.values[:, None] / 23.0)
    feats.append(df["date"].dt.dayofweek.values[:, None] / 6.0)
    feats.append(df["date"].dt.month.values[:, None] / 12.0)
    feats.append(df["date"].dt.dayofyear.values[:, None] / 366.0)
    if freq_is_subhourly:
        feats.append(df["date"].dt.minute.values[:, None] / 59.0)
    cov = np.concatenate(feats, axis=1).astype(np.float32)
    # Fill NaT rows with 0
    cov = np.nan_to_num(cov, nan=0.0)
    return cov


@dataclass
class SplitData:
    dataset: str
    csv_path: str
    n_features: int
    feature_names: List[str]
    T_total: int
    T_train: int
    T_val: int
    T_test: int
    scaler: StandardScaler
    train_data_scaled: np.ndarray  # (T_train, C)
    val_data_scaled: np.ndarray
    test_data_scaled: np.ndarray
    train_raw: np.ndarray
    val_raw: np.ndarray
    test_raw: np.ndarray
    # TiDE covariates (optional, not used by default pure loop but available for GATiDE attention)
    dates: Optional[pd.Series] = None
    cov_train: Optional[np.ndarray] = None  # (T_train, F)
    cov_val: Optional[np.ndarray] = None
    cov_test: Optional[np.ndarray] = None


def load_and_split(
    csv_dir: str,
    dataset: str,
    lookback: int = 720,
    horizon: int = 96,
    split_ratios: Tuple[float, float, float] = (0.70, 0.10, 0.20),
    split_convention: str = "tide",
    use_covariates: bool = False,
) -> SplitData:
    """Load CSV, split sequentially, fit scaler on train only.

    Args:
        split_convention: "tide" (7:1:2 for all, TiDE paper §5.1) or "prior-work" (6:2:2 for ETT)
        use_covariates: if True, also generate time-derived covariates for GATiDE segment fusion
    Returns SplitData with both scaled and raw arrays.
    """
    csv_path = _resolve_csv(csv_dir, dataset)
    df = pd.read_csv(csv_path)

    # Identify date column: common names, otherwise first column if parsable
    date_col_candidates = ["date", "Date", "datetime", "timestamp", "time"]
    date_col = None
    for c in date_col_candidates:
        if c in df.columns:
            date_col = c
            break
    # Fallback: if first column can be parsed as date and name is not numeric
    if date_col is None:
        # Use first column as date if it looks like datetime
        try:
            pd.to_datetime(df.iloc[:, 0], errors="raise")
            date_col = df.columns[0]
        except Exception:
            date_col = None

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        value_cols = [c for c in df.columns if c != date_col]
        # Drop rows where date parsing failed?
        # Sort by date to ensure chronological order
        df = df.sort_values(date_col)
    else:
        # No date column: treat all columns as values, assume already sorted
        value_cols = list(df.columns)

    # Coerce to numeric, drop non-numeric columns (e.g., stray strings)
    data_raw = df[value_cols].apply(pd.to_numeric, errors="coerce")
    # Forward fill then drop remaining NaNs (some weather cols have NaNs)
    data_raw = data_raw.ffill().bfill().fillna(0)

    values = data_raw.values.astype(np.float32)  # (T, C)
    feature_names = value_cols
    T_total = len(values)

    # Split per convention – TiDE paper uses 7:1:2 for all datasets
    if split_convention in ("tide", "prior-work"):
        n_train, n_val, n_test = _split_indices(T_total, dataset, split_convention)
        # Validate fixed counts for ETT prior-work
        if n_train + n_val + n_test > T_total:
            # Fallback to ratio if file shorter than expected (e.g., truncated traffic)
            n_train = int(T_total * split_ratios[0])
            n_val = int(T_total * split_ratios[1])
            n_test = T_total - n_train - n_val
    else:
        n_train = int(T_total * split_ratios[0])
        n_val = int(T_total * split_ratios[1])
        n_test = T_total - n_train - n_val

    # Ensure at least lookback + horizon available in each split for windowing
    # (Splits are sequential, not windowed yet; windowing is done per split)
    train_raw = values[:n_train]
    val_raw = values[n_train: n_train + n_val]
    test_raw = values[n_train + n_val:]

    scaler = StandardScaler()
    scaler.fit(train_raw)
    train_scaled = scaler.transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    # Time covariates (optional) – generate from date column if requested
    dates = df[date_col] if date_col is not None else None
    cov_train = cov_val = cov_test = None
    if use_covariates and dates is not None:
        freq_is_subhourly = dataset.startswith("ETTm") or dataset.lower() == "weather"
        cov_all = _make_time_covariates(dates, freq_is_subhourly=freq_is_subhourly)
        if cov_all is not None:
            cov_train = cov_all[:n_train]
            cov_val = cov_all[n_train: n_train + n_val]
            cov_test = cov_all[n_train + n_val:]
            # Standardize covariates with train stats as well
            cov_scaler = StandardScaler()
            cov_scaler.fit(cov_train)
            cov_train = cov_scaler.transform(cov_train)
            cov_val = cov_scaler.transform(cov_val)
            cov_test = cov_scaler.transform(cov_test)

    return SplitData(
        dataset=dataset,
        csv_path=csv_path,
        n_features=len(feature_names),
        feature_names=feature_names,
        T_total=T_total,
        T_train=n_train,
        T_val=n_val,
        T_test=n_test,
        scaler=scaler,
        train_data_scaled=train_scaled,
        val_data_scaled=val_scaled,
        test_data_scaled=test_scaled,
        train_raw=train_raw,
        val_raw=val_raw,
        test_raw=test_raw,
        dates=dates,
        cov_train=cov_train,
        cov_val=cov_val,
        cov_test=cov_test,
    )


def make_loaders(
    split: SplitData,
    lookback: int,
    horizon: int,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders from SplitData."""
    train_ds = TimeSeriesWindowDataset(split.train_data_scaled, lookback, horizon)
    val_ds = TimeSeriesWindowDataset(split.val_data_scaled, lookback, horizon)
    test_ds = TimeSeriesWindowDataset(split.test_data_scaled, lookback, horizon)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)
    return train_loader, val_loader, test_loader

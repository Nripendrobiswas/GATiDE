"""Metrics computed on inverse-scaled predictions."""
from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Both inputs shape (N, H, C) in original scale."""
    return {"mse": mse(y_true, y_pred), "mae": mae(y_true, y_pred)}


def batch_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Convenience returning mse, mae."""
    return mse(y_true, y_pred), mae(y_true, y_pred)

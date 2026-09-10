"""
StandardScaler utilities – fit on train only to prevent leakage.

Replicates the preprocessing described in the requirements:
  Standardize inputs (zero-mean, unit-variance) using training split
  parameters. Inverse transform is required before computing MSE/MAE on
  the test set so metrics are in original scale (or report both).
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler as SkStandardScaler


class StandardScaler:
    """Thin wrapper over sklearn StandardScaler with inverse support.

    Fitted only on train data. Handles (T, C) arrays.
    """

    def __init__(self):
        self.scaler = SkStandardScaler()
        self._fitted = False

    def fit(self, data: np.ndarray) -> "StandardScaler":
        """Fit on 2-D array (T, C)."""
        assert data.ndim == 2, f"expected (T, C), got {data.shape}"
        self.scaler.fit(data)
        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        assert self._fitted, "scaler not fitted"
        assert data.ndim == 2
        return self.scaler.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform 2-D or 3-D arrays.

        For 3-D (N, H, C) we reshape to (-1, C), inverse, then reshape back.
        """
        assert self._fitted
        if data.ndim == 2:
            return self.scaler.inverse_transform(data)
        elif data.ndim == 3:
            n, h, c = data.shape
            flat = data.reshape(-1, c)
            inv = self.scaler.inverse_transform(flat)
            return inv.reshape(n, h, c)
        else:
            raise ValueError(f"unsupported ndim {data.ndim}")

    @property
    def mean(self) -> np.ndarray:
        return self.scaler.mean_

    @property
    def scale(self) -> np.ndarray:
        return self.scaler.scale_

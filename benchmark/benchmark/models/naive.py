"""
Naive persistence baselines.

- last: repeat last observed value across horizon (classic persistence)
- mean: repeat mean of lookback window
- drift: linear drift from first to last observation (naive drift)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NaiveBaseline(nn.Module):
    """No trainable parameters – pure persistence.

    Args:
        num_features: C
        lookback: L (unused but kept for API compat)
        horizon: H
        strategy: last | mean | drift
    """

    def __init__(
        self,
        num_features: int,
        lookback: int = 720,
        horizon: int = 96,
        strategy: str = "last",
        **kwargs,
    ):
        super().__init__()
        self.C = num_features
        self.L = lookback
        self.H = horizon
        assert strategy in ("last", "mean", "drift"), f"unknown strategy {strategy}"
        self.strategy = strategy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C)
        returns: (B, H, C)
        """
        B, L, C = x.shape
        if self.strategy == "last":
            last = x[:, -1:, :]  # (B, 1, C)
            return last.repeat(1, self.H, 1)
        elif self.strategy == "mean":
            mean = x.mean(dim=1, keepdim=True)  # (B, 1, C)
            return mean.repeat(1, self.H, 1)
        elif self.strategy == "drift":
            # Linear extrapolation from first to last
            first = x[:, 0:1, :]  # (B, 1, C)
            last = x[:, -1:, :]   # (B, 1, C)
            slope = (last - first) / max(L - 1, 1)  # (B, 1, C)
            # For horizon steps 1..H beyond last
            steps = torch.arange(1, self.H + 1, device=x.device, dtype=x.dtype).view(1, self.H, 1)
            return last + slope * steps
        else:
            raise ValueError(self.strategy)

    # No parameters, but trainer expects .parameters()
    # Return empty iterator

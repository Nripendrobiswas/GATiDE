"""
DLinear – Decomposition Linear baseline.

Reference: Zeng et al. 2023 "Are Transformers Effective for Time Series Forecasting?"
DLinear decomposes input into trend (moving average) and seasonal (residual) then
applies linear projection per component, summing outputs.

Two modes:
  individual=False (default): single Linear(L -> H) shared? Actually DLinear uses
    per-channel? We implement channel-independent linear via nn.Linear that is
    applied as Linear over time dimension per channel (weight shared across channels
    unless individual=True which uses per-channel weights).

We implement both variants via nn.Linear with appropriate handling.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """Moving average for trend extraction."""

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> need (B, C, L) for AvgPool1d
        # padding to keep length L
        padding = (self.kernel_size - 1) // 2
        # Pad front/back with replicate of endpoints to keep length
        # Simpler: use F.pad then avg
        import torch.nn.functional as F
        # x (B, L, C) -> (B, C, L)
        x_perm = x.permute(0, 2, 1)
        # Pad
        front = x_perm[:, :, 0:1].repeat(1, 1, padding)
        end = x_perm[:, :, -1:].repeat(1, 1, padding)
        x_padded = torch.cat([front, x_perm, end], dim=2)
        # AvgPool sliding with kernel
        # Use 1D conv avg via AvgPool1d over padded
        # Kernel will produce L outputs if correctly padded
        # We need to handle even kernel padding – ensure output L
        x_smooth = self.avg(x_padded)  # (B, C, L) if padding correct
        # If length mismatch (off by 1 due to even kernel), trim/pad
        if x_smooth.shape[-1] != x.shape[1]:
            # Trim or interpolate
            if x_smooth.shape[-1] > x.shape[1]:
                x_smooth = x_smooth[:, :, :x.shape[1]]
            else:
                # pad last value
                pad_len = x.shape[1] - x_smooth.shape[-1]
                x_smooth = torch.cat([x_smooth, x_smooth[:, :, -1:].repeat(1, 1, pad_len)], dim=2)
        return x_smooth.permute(0, 2, 1)  # (B, L, C)


class DLinear(nn.Module):
    """DLinear model.

    Args:
        num_features: C
        lookback: L
        horizon: H
        kernel_size: moving average kernel for decomposition
        individual: if True, per-feature linear layers; else shared
    """

    def __init__(
        self,
        num_features: int,
        lookback: int = 720,
        horizon: int = 96,
        kernel_size: int = 25,
        individual: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.C = num_features
        self.L = lookback
        self.H = horizon
        self.individual = individual
        self.decomposition = MovingAvg(kernel_size)

        if individual:
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()
            for _ in range(num_features):
                self.linear_seasonal.append(nn.Linear(lookback, horizon))
                self.linear_trend.append(nn.Linear(lookback, horizon))
        else:
            self.linear_seasonal = nn.Linear(lookback, horizon)
            self.linear_trend = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C)
        returns: (B, H, C)
        """
        # Decompose
        trend = self.decomposition(x)  # (B, L, C)
        seasonal = x - trend

        if self.individual:
            out_seasonal = torch.zeros(x.shape[0], self.H, self.C, device=x.device, dtype=x.dtype)
            out_trend = torch.zeros_like(out_seasonal)
            for c in range(self.C):
                out_seasonal[:, :, c] = self.linear_seasonal[c](seasonal[:, :, c])
                out_trend[:, :, c] = self.linear_trend[c](trend[:, :, c])
            out = out_seasonal + out_trend
        else:
            # Shared linear: apply to per-channel via transpose
            # nn.Linear is last-dim; we need to map L -> H so transpose
            # x (B, L, C) -> (B, C, L) -> linear per (B,C) batch -> (B,C,H) -> (B,H,C)
            s = seasonal.permute(0, 2, 1)  # (B, C, L)
            t = trend.permute(0, 2, 1)
            # Apply linear: we can reshape (B*C, L) -> (B*C, H)
            B = x.shape[0]
            s_flat = s.reshape(B * self.C, self.L)
            t_flat = t.reshape(B * self.C, self.L)
            s_out = self.linear_seasonal(s_flat).view(B, self.C, self.H)
            t_out = self.linear_trend(t_flat).view(B, self.C, self.H)
            out = (s_out + t_out).permute(0, 2, 1)  # (B, H, C)

        return out

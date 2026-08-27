"""
TiDE (Time-series Dense Encoder) – standalone PyTorch implementation.

Reference: Das et al. 2023 "Long-term Forecasting with TiDE: Time-series Dense Encoder"
Architecture mirrors Darts' _TideModule but decoupled from Darts, so it can be
trained with the unified PyTorch loop (MSE, AdamW, CosineAnnealing).

Input:  (B, L, C)   lookback window
Output: (B, H, C)   horizon forecast
Channel-independent by default but can be used multivariate (C is n_features).

Simplifications vs. Darts TiDE:
- No past/future covariate projections in this pure version (covariates are
  optional; the benchmark pipeline currently supplies only target series).
- Static covariates not used.
- ResidualBlock is plain MLP+skip+optional LayerNorm (gating/attention are GATiDE).

This baseline guarantees a fair comparison under the same scaling / windowing /
optimisation as GATiDE.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """TiDE residual block: Linear -> ReLU -> Dropout -> Linear + skip -> LayerNorm."""

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int,
                 dropout: float = 0.1, use_layer_norm: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)
        self.skip = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        # Degenerate LayerNorm(1) zeroes gradient – guard as in GATiDE
        self.layer_norm = nn.LayerNorm(output_dim) if (use_layer_norm and output_dim > 1) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.dropout(self.act(self.fc1(x))))
        out = self.skip(x) + h
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        return out


class TiDE(nn.Module):
    """Vanilla TiDE for multivariate direct multi-step forecasting.

    Args:
        num_features: C (n channels)
        lookback: L
        horizon: H
        hidden_size: width of encoder/decoder residual stack
        num_encoder_layers, num_decoder_layers
        decoder_output_dim: width after decoding before temporal decoder
        temporal_decoder_hidden
        dropout, use_layer_norm
        temporal_width_*: kept for API compat, not used without covariates
    """

    def __init__(
        self,
        num_features: int,
        lookback: int = 720,
        horizon: int = 96,
        hidden_size: int = 256,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        decoder_output_dim: int = 16,
        temporal_decoder_hidden: int = 32,
        temporal_width_past: int = 4,
        temporal_width_future: int = 4,
        dropout: float = 0.1,
        use_layer_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.C = num_features
        self.L = lookback
        self.H = horizon
        self.hidden_size = hidden_size
        self.nr_params = 1  # point forecast

        # Encoder input is flattened lookback: L*C
        encoder_input_dim = lookback * num_features

        # Encoder stack
        layers = []
        in_dim = encoder_input_dim
        for _ in range(num_encoder_layers):
            layers.append(ResidualBlock(in_dim, hidden_size, hidden_size, dropout, use_layer_norm))
            in_dim = hidden_size
        self.encoders = nn.Sequential(*layers)

        # Decoder stack
        decoder_layers = []
        for _ in range(num_decoder_layers - 1):
            decoder_layers.append(ResidualBlock(hidden_size, hidden_size, hidden_size, dropout, use_layer_norm))
        # Last decoder maps to H * decoder_output_dim
        decoder_layers.append(
            ResidualBlock(hidden_size, decoder_output_dim * horizon * self.nr_params,
                          hidden_size, dropout, use_layer_norm)
        )
        self.decoders = nn.Sequential(*decoder_layers)

        # Temporal decoder: maps decoder_output_dim -> C per timestep
        self.temporal_decoder = ResidualBlock(
            decoder_output_dim, num_features * self.nr_params,
            temporal_decoder_hidden, dropout, use_layer_norm
        )
        # Lookback skip connection (per-channel linear from L -> H)
        self.lookback_skip = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C)
        returns: (B, H, C)
        """
        B = x.shape[0]
        # Flatten
        x_flat = x.reshape(B, -1)  # (B, L*C)
        encoded = self.encoders(x_flat)  # (B, hidden)
        decoded = self.decoders(encoded)  # (B, H*decoder_output_dim)
        decoded = decoded.view(B, self.H, -1)  # (B, H, decoder_output_dim)

        # Temporal decoder per step
        # Apply residual block per timestep – share weights across H by reshaping
        # temporal_decoder expects (B*H, decoder_output_dim)
        td_in = decoded.reshape(-1, decoded.shape[-1])
        td_out = self.temporal_decoder(td_in)  # (B*H, C)
        td_out = td_out.view(B, self.H, self.C)

        # Skip connection: linear over time axis per channel
        # x: (B, L, C) -> transpose to (B, C, L) -> linear L->H per channel -> transpose back
        # Reshape to (B*C, L), apply Linear(L->H), then view as (B, C, H) -> (B, H, C)
        skip = self.lookback_skip(x.transpose(1, 2).reshape(B * self.C, self.L))
        skip = skip.view(B, self.C, self.H).transpose(1, 2)  # (B, H, C)

        out = td_out + skip.view_as(td_out)
        # nr_params ==1 so squeeze last dim if needed
        return out

"""
PatchTST – Patch Time Series Transformer.

Reference: Nie et al. 2023 "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
Channel-independent PatchTST: each channel is treated independently, shared weights.

Simplified implementation:
- Patch creation via unfold (patch_len, stride)
- Linear projection patch_len -> d_model
- Learnable positional embedding
- Transformer encoder stack
- Head: flatten patches * d_model -> horizon

Input: (B, L, C) -> output (B, H, C)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class PatchTST(nn.Module):
    def __init__(
        self,
        num_features: int,
        lookback: int = 720,
        horizon: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        n_layers: int = 2,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        act: str = "gelu",
        **kwargs,
    ):
        super().__init__()
        self.C = num_features
        self.L = lookback
        self.H = horizon
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        # Number of patches
        self.num_patches = (lookback - patch_len) // stride + 1
        assert self.num_patches > 0, f"patch_len {patch_len} stride {stride} too large for L={lookback}"

        # Patch projection: patch_len -> d_model
        self.patch_proj = nn.Linear(patch_len, d_model)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=act,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Prediction head: flatten patches
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),  # (B*C, num_patches*d_model)
            nn.Linear(self.num_patches * d_model, horizon),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C)
        returns: (B, H, C)
        """
        B, L, C = x.shape
        assert L == self.L, f"expected L={self.L}, got {L}"
        # Channel independent: reshape to (B*C, L)
        x = x.permute(0, 2, 1).reshape(B * C, L)  # (B*C, L)

        # Create patches via unfold: (B*C, num_patches, patch_len)
        # Use unfold on last dim
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # (B*C, num_patches, patch_len)
        # Project
        patches = self.patch_proj(patches)  # (B*C, num_patches, d_model)
        patches = patches + self.pos_embed  # broadcast
        patches = self.dropout(patches)

        # Transformer
        enc_out = self.encoder(patches)  # (B*C, num_patches, d_model)

        # Head
        out = self.head(enc_out)  # (B*C, H)
        out = out.view(B, C, self.H).permute(0, 2, 1)  # (B, H, C)
        return out

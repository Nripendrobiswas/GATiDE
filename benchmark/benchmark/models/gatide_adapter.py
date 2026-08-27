"""
GATiDE adapter – bridges src/ga_tide/model.py into the unified PyTorch benchmark.

Two integration paths:

1) Darts path (optional): wraps `GATiDEModel` from `src/ga_tide/model.py` (darts.models.TiDEModel subclass).
   This path is used when running via `benchmark_darts.py` or when --use-darts flag is set.
   It inherits Darts' PLForecastingModule plumbing, add_encoders, historical_forecasts, etc.
   See scripts/benchmark.py in the sibling GATiDE repo for the canonical Darts usage.

2) Pure PyTorch path (default for this benchmark): re-implements the GATiDE
   architectural deltas (GatedResidualBlock + SegmentAttentionFusion) as a
   standalone nn.Module that accepts (B, L, C) tensors, so it can be trained
   with the same Engine (MSE, AdamW, CosineAnnealing) as TiDE/DLinear/PatchTST
   for_apples-to-apples throughput and accuracy comparison.

The pure path does NOT require Darts at runtime, only torch. It faithfully
re-implements the paper description:
  - GatedResidualBlock: fc1->ReLU->dropout->fc2 + sigmoid gate on skip branch
  - SegmentAttentionFusion: project each input segment (here single segment = flattened lookback)
    to hidden_size and run MultiheadAttention across segments. When only one segment
    is present (no covariates, the typical benchmark case without add_encoders), the
    fusion is bypassed – matching the fallback in the original code
    (segment_fusion = None when len(segment_dims) < 2).

For the benchmark's covariate-free setting (lookback only), GATiDE reduces to:
  Gated encoder/decoder stacks + skip connection, which is still meaningfully
  different from vanilla TiDE (gating + dropout placement + optional LayerNorm fix).

If src/ga_tide/model.py is importable, we also expose the original classes for
advanced users (e.g., running Darts historical_forecasts with covariates).
"""
from __future__ import annotations

import os
import sys
import importlib.util
from typing import Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Try to import original GATiDE for reference / Darts path
# ---------------------------------------------------------------------------

def _try_import_original():
    """Attempt to import GATiDEModel from sibling repo src/ga_tide/model.py.

    Returns (GATiDEModel, GatedResidualBlock, SegmentAttentionFusion, _GATideModule) or (None, ...)
    """
    candidates = [
        # When pip install -e . was run from GATiDE repo
        "ga_tide.model",
        # Relative path from OPEN CODE to GATiDE
        None,
    ]
    # Try pip-installed
    try:
        import ga_tide.model as m  # type: ignore
        return m.GATiDEModel, m.GatedResidualBlock, m.SegmentAttentionFusion, m._GATideModule
    except Exception:
        pass

    # Try filesystem sibling
    sibling_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "GATiDE", "src", "ga_tide", "model.py"),
        os.path.join(os.path.dirname(__file__), "..", "GATiDE", "src", "ga_tide", "model.py"),
        "E:/Machine Learning Research/GATiDE Final Verse/GATiDE/src/ga_tide/model.py",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../GATiDE/src/ga_tide/model.py")),
    ]
    for p in sibling_paths:
        p = os.path.abspath(p)
        if os.path.exists(p):
            try:
                spec = importlib.util.spec_from_file_location("ga_tide_original", p)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                return mod.GATiDEModel, mod.GatedResidualBlock, mod.SegmentAttentionFusion, mod._GATideModule
            except Exception as e:
                # Report but continue
                print(f"[gatide_adapter] failed to load original from {p}: {e}")
                continue
    return None, None, None, None


GATiDEModel_orig, GatedResidualBlock_orig, SegmentAttentionFusion_orig, _GATideModule_orig = _try_import_original()

if GATiDEModel_orig is not None:
    print("[gatide_adapter] Original GATiDEModel imported successfully (Darts path available).")
else:
    print("[gatide_adapter] Original GATiDEModel not found – pure PyTorch GATiDE will be used only.")


# ---------------------------------------------------------------------------
# Pure PyTorch reimplementation (faithful to paper + original file)
# ---------------------------------------------------------------------------

class GatedResidualBlock(nn.Module):
    """Gated residual block – identical to src/ga_tide/model.py:GatedResidualBlock."""

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int,
                 dropout: float = 0.1, use_layer_norm: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(input_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim)
        self.act = nn.ReLU()
        self.layer_norm = (
            nn.LayerNorm(output_dim) if (use_layer_norm and output_dim > 1) else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.dropout(self.act(self.fc1(x))))
        g = torch.sigmoid(self.gate(x))
        out = self.skip(x) + g * h
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        return out


class SegmentAttentionFusion(nn.Module):
    """Segment attention fusion – same as original but decoupled."""

    def __init__(self, segment_dims: list[int], hidden_size: int,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if len(segment_dims) < 2:
            raise ValueError("Need >=2 segments for attention fusion")
        self.projections = nn.ModuleList([nn.Linear(d, hidden_size) for d in segment_dims])
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.output_dim = len(segment_dims) * hidden_size

    def forward(self, segments: list[torch.Tensor]) -> torch.Tensor:
        tokens = torch.stack([proj(seg) for proj, seg in zip(self.projections, segments)], dim=1)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        fused = self.norm(tokens + attn_out)
        return fused.flatten(start_dim=1)


class GATiDEPure(nn.Module):
    """
    Pure PyTorch GATiDE – standalone for benchmark's unified training loop.

    Mirrors _GATideModule architecture but simplified for the covariate-free
    case (input is only flattened lookback). If covariates were supplied,
    segment_dims would be expanded and fusion enabled.

    Input:  (B, L, C)
    Output: (B, H, C)
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
        num_attn_heads: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.C = num_features
        self.L = lookback
        self.H = horizon
        self.hidden_size = hidden_size
        self.num_attn_heads = num_attn_heads
        if hidden_size % num_attn_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_attn_heads {num_attn_heads}")

        # Segment dims – in covariate-free benchmark only lookback segment exists
        segment_dims = [lookback * num_features]
        # If covariates were added, append here – e.g., past_cov_flat etc.
        # For now single segment -> no fusion
        if len(segment_dims) >= 2:
            self.segment_fusion = SegmentAttentionFusion(segment_dims, hidden_size, num_attn_heads, dropout)
            fused_dim = self.segment_fusion.output_dim
        else:
            self.segment_fusion = None
            fused_dim = segment_dims[0]

        # Encoder stack: first block maps fused_dim -> hidden_size, rest hidden->hidden
        self.encoders = nn.Sequential(
            GatedResidualBlock(fused_dim, hidden_size, hidden_size, dropout, use_layer_norm),
            *[GatedResidualBlock(hidden_size, hidden_size, hidden_size, dropout, use_layer_norm)
              for _ in range(num_encoder_layers - 1)]
        )

        # Decoder stack: hidden -> hidden ... -> H*decoder_output_dim
        dec_layers = []
        for _ in range(num_decoder_layers - 1):
            dec_layers.append(GatedResidualBlock(hidden_size, hidden_size, hidden_size, dropout, use_layer_norm))
        dec_layers.append(GatedResidualBlock(hidden_size, decoder_output_dim * horizon,
                                             hidden_size, dropout, use_layer_norm))
        self.decoders = nn.Sequential(*dec_layers)

        # Temporal decoder: decoder_output_dim -> C
        self.temporal_decoder = GatedResidualBlock(
            decoder_output_dim, num_features,
            temporal_decoder_hidden, dropout, use_layer_norm
        )
        self.lookback_skip = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C)
        returns: (B, H, C)
        """
        B, L, C = x.shape
        segments = [x.reshape(B, -1)]  # single flattened segment
        if self.segment_fusion is not None:
            fused = self.segment_fusion(segments)
        else:
            fused = segments[0]
        encoded = self.encoders(fused)  # (B, hidden)
        decoded = self.decoders(encoded)  # (B, H*decoder_output_dim)
        decoded = decoded.view(B, self.H, -1)  # (B, H, decoder_output_dim)

        # Temporal decoder per step
        td_in = decoded.reshape(B * self.H, -1)
        td_out = self.temporal_decoder(td_in).view(B, self.H, self.C)

        # Skip: per-channel Linear(L -> H)
        skip = self.lookback_skip(x.transpose(1, 2).reshape(B * self.C, self.L))
        skip = skip.view(B, self.C, self.H).transpose(1, 2)  # (B, H, C)

        return td_out + skip


# Alias for benchmark factory
GATiDE = GATiDEPure

def get_gatide_model(*args, use_darts: bool = False, **kwargs):
    """
    Factory that returns either pure PyTorch GATiDE or the Darts GATiDEModel.

    By default returns pure PyTorch (fair throughput comparison). Set
    use_darts=True to get the original Darts model (requires darts & compatible env).
    """
    if use_darts and GATiDEModel_orig is not None:
        # GATiDEModel expects input_chunk_length etc. – map args
        # Caller must supply via kwargs with Darts naming
        return GATiDEModel_orig(*args, **kwargs)
    return GATiDEPure(*args, **kwargs)

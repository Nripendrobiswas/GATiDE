"""
GATiDE Ablation Variants - standalone
======================================
Does NOT modify benchmark/models/gatide_adapter.py or src/ga_tide/model.py.
Duplicates the exact blocks for isolation, with line anchors to originals.

- GatedResidualBlock : src/ga_tide/model.py:51 + benchmark/models/gatide_adapter.py:99
  forward: h = W2 Dropout(ReLU(W1 x)), g = sigmoid(Wg x), y = Ln*( Ws x + g*h )
- ResidualBlock      : benchmark/models/tide.py:27
  forward: h = W2 Dropout(ReLU(W1 x)), y = Ln*( Ws x + h )
- SegmentAttentionFusion : src/ga_tide/model.py:82 + gatide_adapter.py:124
  Bypass when |D|<2 : gatide_adapter.py:187

One configurable class GATiDEAblation with flags:
  use_gating: True -> GatedResidualBlock else ResidualBlock
  use_layer_norm: bool (conditional Ln* when d_out>1)
  use_skip: bool (global Linear(L->H) skip)

Thin aliases GATiDE_Gated / GATiDE_NoGate / GATiDE_NoLN / GATiDE_NoSkip set flags
for CLI convenience: --ablations gatide gatide_nogate gatide_no_ln

Usage in run_ablation.py:
  from ablation_study.models.gatide_ablation import GATiDEAblation, GATiDE_Gated, GATiDE_NoGate
  model = GATiDEAblation(num_features=7, lookback=720, horizon=96, hidden_size=256,
                         use_gating=True, use_layer_norm=True, use_skip=True)
"""
from __future__ import annotations
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Blocks - exact copies with anchors
# ---------------------------------------------------------------------------

class GatedResidualBlock(nn.Module):
    """GatedResidualBlock - src/ga_tide/model.py:51, gatide_adapter.py:99"""
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int,
                 dropout: float = 0.1, use_layer_norm: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(input_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim)
        self.act = nn.ReLU()
        self.layer_norm = nn.LayerNorm(output_dim) if (use_layer_norm and output_dim > 1) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.dropout(self.act(self.fc1(x))))
        g = torch.sigmoid(self.gate(x))
        out = self.skip(x) + g * h
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        return out


class ResidualBlock(nn.Module):
    """ResidualBlock - benchmark/models/tide.py:27 (no gate)"""
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int,
                 dropout: float = 0.1, use_layer_norm: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)
        self.skip = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.layer_norm = nn.LayerNorm(output_dim) if (use_layer_norm and output_dim > 1) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.dropout(self.act(self.fc1(x))))
        out = self.skip(x) + h
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        return out


class SegmentAttentionFusion(nn.Module):
    """SegmentAttentionFusion - gatide_adapter.py:124"""
    def __init__(self, segment_dims: list[int], hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if len(segment_dims) < 2:
            raise ValueError("Need >=2 segments for attention fusion")
        self.projections = nn.ModuleList([nn.Linear(d, hidden_size) for d in segment_dims])
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.output_dim = len(segment_dims) * hidden_size

    def forward(self, segments: list[torch.Tensor]) -> torch.Tensor:
        tokens = torch.stack([proj(seg) for proj, seg in zip(self.projections, segments)], dim=1)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        fused = self.norm(tokens + attn_out)
        return fused.flatten(start_dim=1)


# ---------------------------------------------------------------------------
# Configurable GATiDE Ablation
# ---------------------------------------------------------------------------

class GATiDEAblation(nn.Module):
    """
    Configurable GATiDE for ablations.
    Mirrors benchmark/models/gatide_adapter.py:145 GATiDEPure but adds flags.

    Args:
        num_features, lookback L, horizon H
        hidden_size, num_encoder_layers, num_decoder_layers, decoder_output_dim, temporal_decoder_hidden
        dropout, use_layer_norm, num_attn_heads
        use_gating: True -> GatedResidualBlock else ResidualBlock
        use_skip: True -> Linear(L->H) global skip else no skip
    Input (B,L,C) -> Output (B,H,C)
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
        use_layer_norm: bool = True,
        num_attn_heads: int = 4,
        use_gating: bool = True,
        use_skip: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.C = num_features
        self.L = lookback
        self.H = horizon
        self.hidden_size = hidden_size
        self.num_attn_heads = num_attn_heads
        self.use_gating = use_gating
        self.use_skip = use_skip
        self.use_layer_norm = use_layer_norm
        if hidden_size % num_attn_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_attn_heads {num_attn_heads}")
        Block = GatedResidualBlock if use_gating else ResidualBlock

        # In covariate-free mode only one segment -> SAF bypassed (gatide_adapter.py:187)
        segment_dims = [lookback * num_features]
        if len(segment_dims) >= 2:
            self.segment_fusion = SegmentAttentionFusion(segment_dims, hidden_size, num_attn_heads, dropout)
            fused_dim = self.segment_fusion.output_dim
        else:
            self.segment_fusion = None
            fused_dim = segment_dims[0]

        # Encoder: fused_dim -> hidden, rest hidden->hidden
        self.encoders = nn.Sequential(
            Block(fused_dim, hidden_size, hidden_size, dropout, use_layer_norm),
            *[Block(hidden_size, hidden_size, hidden_size, dropout, use_layer_norm) for _ in range(num_encoder_layers - 1)]
        )
        # Decoder: hidden -> ... -> H*decoder_output_dim
        dec_layers = []
        for _ in range(num_decoder_layers - 1):
            dec_layers.append(Block(hidden_size, hidden_size, hidden_size, dropout, use_layer_norm))
        dec_layers.append(Block(hidden_size, decoder_output_dim * horizon, hidden_size, dropout, use_layer_norm))
        self.decoders = nn.Sequential(*dec_layers)

        # Temporal decoder per step
        self.temporal_decoder = Block(decoder_output_dim, num_features, temporal_decoder_hidden, dropout, use_layer_norm)
        self.lookback_skip = nn.Linear(lookback, horizon) if use_skip else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        segments = [x.reshape(B, -1)]
        if self.segment_fusion is not None:
            fused = self.segment_fusion(segments)
        else:
            fused = segments[0]
        encoded = self.encoders(fused)
        decoded = self.decoders(encoded)
        decoded = decoded.view(B, self.H, -1)
        td_in = decoded.reshape(B * self.H, -1)
        td_out = self.temporal_decoder(td_in).view(B, self.H, self.C)
        if self.lookback_skip is not None:
            skip = self.lookback_skip(x.transpose(1, 2).reshape(B * self.C, self.L))
            skip = skip.view(B, self.C, self.H).transpose(1, 2)
            return td_out + skip
        return td_out


# ---------------------------------------------------------------------------
# Thin aliases for registry
# ---------------------------------------------------------------------------

class GATiDE_Gated(GATiDEAblation):
    """Full GATiDE: GRB + LN + Skip"""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_gating", True)
        kwargs.setdefault("use_layer_norm", True)
        kwargs.setdefault("use_skip", True)
        super().__init__(*args, **kwargs)

class GATiDE_NoGate(GATiDEAblation):
    """Ablation: No gating (RB), LN+Skip kept -> isolates gate"""
    def __init__(self, *args, **kwargs):
        kwargs["use_gating"] = False
        kwargs.setdefault("use_layer_norm", True)
        kwargs.setdefault("use_skip", True)
        super().__init__(*args, **kwargs)

class GATiDE_NoLN(GATiDEAblation):
    """Ablation: No LayerNorm, GRB+Skip kept"""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_gating", True)
        kwargs["use_layer_norm"] = False
        kwargs.setdefault("use_skip", True)
        super().__init__(*args, **kwargs)

class GATiDE_NoSkip(GATiDEAblation):
    """Ablation: No global skip, GRB+LN kept"""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_gating", True)
        kwargs.setdefault("use_layer_norm", True)
        kwargs["use_skip"] = False
        super().__init__(*args, **kwargs)

# Backwards compat aliases
GATiDE = GATiDE_Gated

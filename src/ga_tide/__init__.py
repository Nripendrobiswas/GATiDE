"""GA-TiDE: Gated-Attention Time-series Dense Encoder.

A drop-in subclass of the Darts ``TiDEModel`` adding a gated residual block and
segment-attention fusion at the encoder input.

    from ga_tide import GATiDEModel

    model = GATiDEModel(
        input_chunk_length=720,
        output_chunk_length=96,
        hidden_size=256,
        num_attn_heads=4,       # must divide hidden_size
    )
"""

from ga_tide.model import (
    GATiDEModel,
    GatedResidualBlock,
    SegmentAttentionFusion,
    _GATideModule,
)

__all__ = [
    "GATiDEModel",
    "GatedResidualBlock",
    "SegmentAttentionFusion",
    "_GATideModule",
]
__version__ = "0.1.0"

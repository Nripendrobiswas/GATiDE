"""Ablation variant definitions.

Registers the GA-TiDE ablation variants into benchmark.models.MODEL_REGISTRY:

  gatide-gate : gate kept, segment attention REPLACED by vanilla-TiDE-style
                concat pre-mixing of projected covariates (isolates the
                gating contribution)
  gatide-attn : segment attention kept, gate REMOVED (all residual blocks are
                plain TiDE ResidualBlocks; isolates the attention contribution)

The other two corners of the 2x2 factorial already exist in the base registry:
  tide   : vanilla TiDE (no gate, no attention) -- accepts the same covariate
           information via the vanilla concat path (benchmark injects
           num_time_covariates when --use-covariates is set)
  gatide : full GA-TiDE (gate + segment attention)

All four share the identical data protocol, trainer, and (via run_ablation.py)
hyperparameters -- only the architecture differs.
"""
from __future__ import annotations

from typing import Dict, Type
import torch.nn as nn

from benchmark.models import MODEL_REGISTRY
from benchmark.models.gatide_adapter import GATiDEPure


class GATiDEGate(GATiDEPure):
    """Ablation: gating kept, attention replaced by vanilla concat pre-mixing.

    num_time_covariates is declared explicitly so benchmark.py's signature-based
    covariate injection can see it through the wrapper.
    """

    def __init__(self, *args, num_time_covariates: int = 0, **kwargs):
        kwargs.setdefault("fusion", "concat")
        super().__init__(*args, num_time_covariates=num_time_covariates, **kwargs)


class GATiDEAttn(GATiDEPure):
    """Ablation: segment attention kept, gating removed (plain residual blocks).

    num_time_covariates is declared explicitly so benchmark.py's signature-based
    covariate injection can see it through the wrapper.
    """

    def __init__(self, *args, num_time_covariates: int = 0, **kwargs):
        kwargs.setdefault("use_gate", False)
        super().__init__(*args, num_time_covariates=num_time_covariates, **kwargs)


ABLATION_VARIANTS: Dict[str, Type[nn.Module]] = {
    "gatide-gate": GATiDEGate,
    "gatide-attn": GATiDEAttn,
}

# canonical 2x2 factorial ordering used for tables
ABLATION_ORDER = ["tide", "gatide-gate", "gatide-attn", "gatide"]


def register_ablation_variants() -> list:
    """Idempotently register the ablation variants into MODEL_REGISTRY."""
    for name, cls in ABLATION_VARIANTS.items():
        MODEL_REGISTRY[name] = cls
    return list(ABLATION_VARIANTS.keys())

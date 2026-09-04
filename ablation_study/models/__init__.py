"""Ablation study model registry - standalone, does not patch benchmark/models/__init__.py"""
from .gatide_ablation import (
    GATiDEAblation,
    GATiDE_Gated,
    GATiDE_NoGate,
    GATiDE_NoLN,
    GATiDE_NoSkip,
)

ABLATION_REGISTRY = {
    "gatide": GATiDE_Gated,
    "gatide_gated": GATiDE_Gated,
    "gatide_gated_ln": GATiDE_Gated,
    "gatide_nogate": GATiDE_NoGate,
    "gatide_no_gate": GATiDE_NoGate,
    "gatide-rb": GATiDE_NoGate,
    "gatide_no_ln": GATiDE_NoLN,
    "gatide_noln": GATiDE_NoLN,
    "gatide_no_skip": GATiDE_NoSkip,
    "gatide_noskip": GATiDE_NoSkip,
}

def get_ablation_model(name: str):
    key = name.lower().strip().replace("-", "_")
    if key not in ABLATION_REGISTRY:
        raise ValueError(f"Unknown ablation '{name}'. Available: {list(ABLATION_REGISTRY)}")
    return ABLATION_REGISTRY[key]

def list_ablations():
    return sorted(set(ABLATION_REGISTRY.keys()))

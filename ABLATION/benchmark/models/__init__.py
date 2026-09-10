"""Model factory for benchmark."""
from __future__ import annotations

from typing import Dict, Type
import torch.nn as nn

from benchmark.models.tide import TiDE
from benchmark.models.dlinear import DLinear
from benchmark.models.patchtst import PatchTST
from benchmark.models.naive import NaiveBaseline
from benchmark.models.gatide_adapter import GATiDEPure

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "gatide": GATiDEPure,
    "ga-tide": GATiDEPure,
    "gatide-pure": GATiDEPure,
    "tide": TiDE,
    "dlinear": DLinear,
    "patchtst": PatchTST,
    "naive": NaiveBaseline,
    # alias for persistence baseline
    "persistence": NaiveBaseline,
}

def get_model(name: str):
    key = name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]

def list_models():
    return list(MODEL_REGISTRY)

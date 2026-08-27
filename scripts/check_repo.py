"""Repository health check.

Verifies that the repository is correctly installed and that every script is
importable and internally consistent, WITHOUT downloading any dataset or
training anything substantial. Run this first; it isolates packaging and
import problems from modelling problems.

    python scripts/check_repo.py

Exit code 0 means every check passed.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import traceback

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    """Decorator running one check and recording PASS/FAIL rather than raising,
    so a single failure does not hide the ones after it."""
    def wrap(fn):
        try:
            detail = fn() or ""
            RESULTS.append((name, True, str(detail)))
            print(f"  [PASS] {name}" + (f" -- {detail}" if detail else ""))
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            RESULTS.append((name, False, msg))
            print(f"  [FAIL] {name} -- {msg}")
            if os.environ.get("CHECK_REPO_TRACEBACK"):
                traceback.print_exc()
        return fn
    return wrap


print("=" * 70)
print("GA-TiDE repository health check")
print("=" * 70)


# --------------------------------------------------------------------------- #
print("\n1. Package installation")
# --------------------------------------------------------------------------- #
@check("ga_tide is importable")
def _():
    import ga_tide
    return f"version {ga_tide.__version__}"


@check("ga_tide resolves to the installed package, not the CWD")
def _():
    import ga_tide
    path = os.path.abspath(ga_tide.__file__)
    assert "ga_tide" in path, path
    # A src/ layout means the package must NOT resolve to the repo root.
    assert os.path.basename(os.path.dirname(os.path.dirname(path))) == "src", (
        f"resolved to {path}; expected .../src/ga_tide/__init__.py. Run "
        "`pip install -e .` from the repository root."
    )
    return path


@check("public API is exported")
def _():
    from ga_tide import (
        GATiDEModel, GatedResidualBlock, SegmentAttentionFusion, _GATideModule,
    )
    return "GATiDEModel, GatedResidualBlock, SegmentAttentionFusion, _GATideModule"


# --------------------------------------------------------------------------- #
print("\n2. Dependencies")
# --------------------------------------------------------------------------- #
@check("darts / torch / lightning import at the pinned versions")
def _():
    import darts, torch, pytorch_lightning as pl, numpy as np
    return (f"darts {darts.__version__}, torch {torch.__version__}, "
            f"pl {pl.__version__}, numpy {np.__version__}")


@check("Darts private internals GA-TiDE depends on still exist")
def _():
    from darts.models.forecasting.tide_model import TiDEModel, _TideModule, _ResidualBlock
    from darts.models.forecasting.pl_forecasting_module import io_processor
    from darts.utils.data.torch_datasets.utils import PLModuleInput, TorchTrainingSample
    from darts.utils.torch import MonteCarloDropout
    return "all present"


# --------------------------------------------------------------------------- #
print("\n3. Model construction (no training)")
# --------------------------------------------------------------------------- #
@check("GATiDEModel constructs")
def _():
    from ga_tide import GATiDEModel
    m = GATiDEModel(input_chunk_length=48, output_chunk_length=12,
                    hidden_size=32, num_attn_heads=4)
    return f"hidden_size=32, num_attn_heads=4"


@check("indivisible hidden_size / num_attn_heads is rejected")
def _():
    from ga_tide import GATiDEModel
    try:
        GATiDEModel(input_chunk_length=48, output_chunk_length=12,
                    hidden_size=30, num_attn_heads=4)
    except ValueError as exc:
        assert "divisible" in str(exc).lower(), str(exc)
        return "ValueError raised as expected"
    raise AssertionError("no error raised for hidden_size=30, num_attn_heads=4")


@check("constructor parameters are captured for save/load")
def _():
    from ga_tide import GATiDEModel
    m = GATiDEModel(input_chunk_length=48, output_chunk_length=12,
                    hidden_size=32, num_attn_heads=8)
    assert m.model_params.get("num_attn_heads") == 8, m.model_params
    return "num_attn_heads survives ModelMeta capture"


@check("GatedResidualBlock forward runs and guards unit-width LayerNorm")
def _():
    import torch
    from ga_tide import GatedResidualBlock
    b = GatedResidualBlock(20, 1, 32, 0.0, True)
    assert b.layer_norm is None, "LayerNorm should be suppressed for output_dim=1"
    out = b(torch.randn(8, 20))
    assert out.shape == (8, 1), out.shape
    b2 = GatedResidualBlock(20, 4, 32, 0.0, True)
    assert b2.layer_norm is not None, "LayerNorm should be active for output_dim=4"
    return "guard active at output_dim=1, inactive at output_dim=4"


@check("SegmentAttentionFusion forward produces the declared width")
def _():
    import torch
    from ga_tide import SegmentAttentionFusion
    f = SegmentAttentionFusion([48, 144, 120], hidden_size=32,
                               num_heads=4, dropout=0.0)
    out = f([torch.randn(8, 48), torch.randn(8, 144), torch.randn(8, 120)])
    assert out.shape == (8, f.output_dim), (out.shape, f.output_dim)
    return f"3 tokens -> {f.output_dim}"


# --------------------------------------------------------------------------- #
print("\n4. Scripts are importable and self-consistent")
# --------------------------------------------------------------------------- #
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


@check("benchmark.py imports")
def _():
    m = importlib.import_module("benchmark")
    assert set(m.MODELS) == {"tide", "ga-tide"}, m.MODELS
    for fn in ("load_series", "split_series", "to_channel_list",
               "build_encoders", "build_model", "evaluate", "run_one"):
        assert hasattr(m, fn), fn
    return f"models: {list(m.MODELS)}; datasets: {list(m.DATASETS)}"


@check("run_diagnostics.py imports")
def _():
    m = importlib.import_module("run_diagnostics")
    for fn in ("run_layout", "run_layernorm", "run_precision", "run_params"):
        assert hasattr(m, fn), fn
    return "layout, layernorm, precision, params"


@check("tune_optuna.py imports (needs optuna)")
def _():
    m = importlib.import_module("tune_optuna")
    assert set(m.MODELS) == {"tide", "ga-tide"}, m.MODELS
    return "objective + MODELS present"


@check("build_model accepts both model names")
def _():
    m = importlib.import_module("benchmark")
    sig = inspect.signature(m.build_model)
    assert "model_name" in sig.parameters, list(sig.parameters)
    return "signature uses model_name"


# --------------------------------------------------------------------------- #
print("\n5. Repository layout")
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(SCRIPTS_DIR)


@check("expected files are present")
def _():
    expected = [
        "README.md", "NOTICE", "LICENSE", "requirements.txt", "pyproject.toml",
        "src/ga_tide/__init__.py", "src/ga_tide/model.py",
        "scripts/benchmark.py", "scripts/run_diagnostics.py",
        "scripts/tune_optuna.py", "tests/test_segment_layout.py",
        "data/README.md",
    ]
    missing = [f for f in expected if not os.path.exists(os.path.join(ROOT, f))]
    assert not missing, f"missing: {missing}"
    return f"{len(expected)} files"


@check(".gitignore has been renamed from gitignore.txt")
def _():
    assert os.path.exists(os.path.join(ROOT, ".gitignore")), (
        "rename gitignore.txt to .gitignore before committing"
    )
    return "present"


@check("LICENSE contains the full Apache 2.0 text")
def _():
    text = open(os.path.join(ROOT, "LICENSE"), encoding="utf-8").read()
    assert "ACTION REQUIRED" not in text, (
        "LICENSE is still the placeholder; replace it with the full text from "
        "https://www.apache.org/licenses/LICENSE-2.0.txt"
    )
    assert "TERMS AND CONDITIONS" in text, "LICENSE does not look like Apache 2.0"
    return f"{len(text)} bytes"


# --------------------------------------------------------------------------- #
print("\n" + "=" * 70)
n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
print(f"{len(RESULTS) - n_fail} passed, {n_fail} failed")
if n_fail:
    print("\nFailures:")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  - {name}: {detail}")
    print("\nRe-run with CHECK_REPO_TRACEBACK=1 for full tracebacks.")
print("=" * 70)
sys.exit(1 if n_fail else 0)

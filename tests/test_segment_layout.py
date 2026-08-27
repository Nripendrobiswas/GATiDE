"""
Segment-layout verification for GA-TiDE
=======================================

Section 4.4 of the manuscript asserts that GA-TiDE's independent re-derivation
of the flattened input-segment widths "is currently consistent with Darts'
internal convention (verified empirically across the no-covariate, static-only,
past-and-future-covariate, and multivariate-target configurations)".

This file is that verification. Run it before submission so the claim is
backed by an executable test rather than by inspection:

    pytest -v test_segment_layout.py
    # or, without pytest:
    python test_segment_layout.py

What is checked, per configuration
----------------------------------
1. The model builds and completes a forward/backward pass.
2. `_segment_dims` computed in `__init__` equals the flattened widths that
   `forward` actually produces (this is what the runtime assertion guards).
3. The channel layout of `x` is `[target | past covariates | historic future
   covariates]` -- the assumption behind the slicing in `forward`. This is
   verified by *content*, not by width, using covariates with disjoint,
   recognizable value ranges. A width-only check cannot detect a reordering
   when two covariate groups happen to have the same number of features,
   which is the failure mode the runtime assertion in `forward` does not
   cover.
4. Prediction shape is `(H, output_dim)`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from darts import TimeSeries
from ga_tide import GATiDEModel, _GATideModule

L, H = 48, 12
N = 400


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _index(n: int = N) -> pd.DatetimeIndex:
    return pd.date_range("2021-01-01", periods=n, freq="h")


def _series(n_components: int = 1, offset: float = 0.0, n: int = N) -> TimeSeries:
    """A deterministic series whose values sit in a recognizable band.

    Each component of the k-th group occupies [offset + k, offset + k + 1),
    so that a reordering of channels inside `x` is detectable by value.
    """
    idx = _index(n)
    t = np.arange(n)
    cols = {}
    for c in range(n_components):
        base = offset + c
        cols[f"c{int(base)}"] = base + 0.5 + 0.4 * np.sin(2 * np.pi * t / 24)
    return TimeSeries.from_dataframe(
        pd.DataFrame(cols, index=idx)
    ).astype(np.float32)


def _model(**overrides) -> GATiDEModel:
    kwargs = dict(
        input_chunk_length=L,
        output_chunk_length=H,
        hidden_size=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        decoder_output_dim=4,
        temporal_decoder_hidden=16,
        temporal_width_past=3,
        temporal_width_future=2,
        dropout=0.0,
        use_layer_norm=False,
        num_attn_heads=4,
        n_epochs=1,
        batch_size=16,
        random_state=0,
        pl_trainer_kwargs={
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
            "limit_train_batches": 2,
        },
        force_reset=True,
        save_checkpoints=False,
    )
    kwargs.update(overrides)
    return GATiDEModel(**kwargs)


# --------------------------------------------------------------------------- #
# The four configurations named in Section 4.4
# --------------------------------------------------------------------------- #
CONFIGS = {
    # name: (n_target_components, n_past_cov, n_future_cov, use_static)
    "no_covariate":            (1, 0, 0, False),
    "static_only":             (1, 0, 0, True),
    "past_and_future_covs":    (1, 2, 2, False),
    "multivariate_target":     (3, 2, 2, False),
    # Additional case not named in the manuscript but worth covering: equal
    # past and future covariate widths, where a channel reordering inside `x`
    # would leave all segment *widths* unchanged and therefore slip past the
    # runtime assertion in `forward`.
    "equal_width_covs":        (1, 2, 2, False),
}


def _build(cfg: tuple[int, int, int, bool]):
    n_target, n_past, n_future, use_static = cfg

    target = _series(n_target, offset=0.0)
    if use_static:
        target = target.with_static_covariates(
            pd.DataFrame({"s0": [7.5], "s1": [8.5]})
        )

    # Disjoint value bands: target in [0, n_target), past covariates starting
    # at 100, future covariates starting at 200.
    past_cov = _series(n_past, offset=100.0) if n_past else None
    # Future covariates must extend H steps beyond the target.
    future_cov = _series(n_future, offset=200.0, n=N + H) if n_future else None

    return target, past_cov, future_cov


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(CONFIGS))
def test_builds_and_predicts(name: str) -> None:
    """The model trains and produces a forecast of the expected shape."""
    target, past_cov, future_cov = _build(CONFIGS[name])
    n_target = CONFIGS[name][0]

    model = _model()
    model.fit(target, past_covariates=past_cov,
              future_covariates=future_cov, verbose=False)

    pred = model.predict(n=H, past_covariates=past_cov,
                         future_covariates=future_cov)
    assert len(pred) == H, f"{name}: expected {H} steps, got {len(pred)}"
    assert pred.width == n_target, (
        f"{name}: expected {n_target} components, got {pred.width}"
    )


@pytest.mark.parametrize("name", list(CONFIGS))
def test_segment_dims_match_forward(name: str) -> None:
    """`_segment_dims` from __init__ equals the widths `forward` produces.

    This is the assumption the runtime assertion in `forward` guards. Here it
    is checked directly by instrumenting the fusion module, so a failure is
    attributed precisely rather than surfacing as a shape error deeper in the
    network.
    """
    target, past_cov, future_cov = _build(CONFIGS[name])

    model = _model()
    model.fit(target, past_covariates=past_cov,
              future_covariates=future_cov, verbose=False)

    net: _GATideModule = model.model
    if net.segment_fusion is None:
        pytest.skip(f"{name}: single segment, fusion disabled by design")

    declared = [p.in_features for p in net.segment_fusion.projections]
    assert declared == net._segment_dims, (
        f"{name}: projections sized {declared} but _segment_dims is "
        f"{net._segment_dims}"
    )

    # Capture the widths actually seen at run time.
    seen: list[list[int]] = []
    original = net.segment_fusion.forward

    def spy(segments):
        seen.append([s.shape[-1] for s in segments])
        return original(segments)

    net.segment_fusion.forward = spy
    model.predict(n=H, past_covariates=past_cov, future_covariates=future_cov)
    net.segment_fusion.forward = original

    assert seen, f"{name}: fusion was never invoked"
    assert seen[0] == declared, (
        f"{name}: forward produced segment widths {seen[0]} but the fusion "
        f"was built for {declared}"
    )


def test_channel_layout_of_x() -> None:
    """Verify the `[target | past covs | historic future covs]` layout of `x`.

    The runtime assertion in `forward` compares *widths* only. When the past
    and future covariate groups have the same number of features, a
    reordering by a future Darts release would leave every width unchanged and
    the assertion would pass while the model silently trained on mis-assigned
    features. This test closes that gap by checking values, not widths.
    """
    target, past_cov, future_cov = _build(CONFIGS["equal_width_covs"])
    assert past_cov.width == future_cov.width, "test presupposes equal widths"

    model = _model()
    model.fit(target, past_covariates=past_cov,
              future_covariates=future_cov, verbose=False)

    net: _GATideModule = model.model
    captured: dict[str, torch.Tensor] = {}

    original_forward = net.forward

    def spy(x_in):
        captured["x"] = x_in[0].detach().clone()
        return original_forward(x_in)

    net.forward = spy
    model.predict(n=H, past_covariates=past_cov, future_covariates=future_cov)
    net.forward = original_forward

    x = captured["x"]
    d_y, d_p, d_f = net.output_dim, net.past_cov_dim, net.future_cov_dim
    assert x.shape[-1] == d_y + d_p + d_f, (
        f"channel count {x.shape[-1]} != {d_y} + {d_p} + {d_f}"
    )

    # Value bands: target < 100, past covariates in [100, 200), historic
    # future covariates >= 200.
    tgt = x[..., :d_y]
    pst = x[..., d_y:d_y + d_p]
    fut = x[..., -d_f:]

    assert float(tgt.max()) < 100.0, (
        "channels [0:output_dim] are not the target -- Darts' layout for `x` "
        "has changed; the slicing in _GATideModule.forward is invalid."
    )
    assert 100.0 <= float(pst.min()) and float(pst.max()) < 200.0, (
        "channels [output_dim:output_dim+past_cov_dim] are not the past "
        "covariates -- Darts' layout for `x` has changed."
    )
    assert float(fut.min()) >= 200.0, (
        "trailing channels are not the historic future covariates -- Darts' "
        "layout for `x` has changed."
    )


def test_head_divisibility_is_validated_at_construction() -> None:
    """An indivisible hidden_size / num_attn_heads pair fails immediately."""
    with pytest.raises(ValueError, match="divisible"):
        _model(hidden_size=30, num_attn_heads=4)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

"""GA-TiDE diagnostics.

Consolidates the verification and measurement experiments the paper reports.
Each flag maps to one artefact:

    --layout      segment-width and channel-layout verification (Section 4.4)
    --layernorm   unit-width LayerNorm gradient diagnostic (Section 4.3)
    --precision   float32 vs float64, deciding whether the residual gradient
                  is numerical or analytic (Section 4.3)
    --params      parameter counts, TiDE vs GA-TiDE (Section 5)
    --all         all of the above

Everything runs on CPU on a small synthetic series in about a minute. No
dataset download, no network access. Results are written to `results/` as CSV.

    python scripts/run_diagnostics.py --all
    python scripts/run_diagnostics.py --params --lookback 720 --hidden-size 256
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from darts import TimeSeries
from darts.models import TiDEModel
from darts.models.forecasting.tide_model import _ResidualBlock

from ga_tide import GATiDEModel, GatedResidualBlock

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

TRAINER = {
    "accelerator": "cpu",
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "logger": False,
    "limit_train_batches": 2,
}

# GATiDEModel applies gating and fusion together with no switch to isolate
# either, so the parameter comparison is between two models.
MODELS = {
    "TiDE (baseline)": TiDEModel,
    "GA-TiDE":         GATiDEModel,
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def banner(text: str) -> None:
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


def make_series(n_components: int, offset: float, n: int) -> TimeSeries:
    """Series whose components occupy disjoint value bands starting at `offset`.

    The bands make the channel-layout check possible: a reordering inside `x`
    is then detectable by value even when every width is unchanged.
    """
    idx = pd.date_range("2021-01-01", periods=n, freq="h")
    t = np.arange(n)
    cols = {
        f"c{int(offset) + c}": offset + c + 0.5 + 0.4 * np.sin(2 * np.pi * t / 24)
        for c in range(n_components)
    }
    return TimeSeries.from_dataframe(pd.DataFrame(cols, index=idx)).astype(np.float32)


def base_kwargs(lookback: int, horizon: int, hidden: int, **over) -> dict:
    kw = dict(
        input_chunk_length=lookback,
        output_chunk_length=horizon,
        hidden_size=hidden,
        num_encoder_layers=1,
        num_decoder_layers=1,
        decoder_output_dim=16,
        temporal_decoder_hidden=32,
        temporal_width_past=3,
        temporal_width_future=2,
        dropout=0.0,
        use_layer_norm=False,
        n_epochs=1,
        batch_size=16,
        random_state=0,
        pl_trainer_kwargs=TRAINER,
        force_reset=True,
        save_checkpoints=False,
    )
    kw.update(over)
    return kw


def save(df: pd.DataFrame, name: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    df.to_csv(path, index=False)
    print(f"\n  written: {path}")


def report_env() -> None:
    import darts
    import pytorch_lightning as pl

    banner("Environment (record these in the paper)")
    print(f"  darts              {darts.__version__}")
    print(f"  torch              {torch.__version__}")
    print(f"  pytorch-lightning  {pl.__version__}")
    print(f"  numpy              {np.__version__}")


# --------------------------------------------------------------------------- #
# --layout
# --------------------------------------------------------------------------- #
LAYOUT_CONFIGS = [
    # name, n_target, n_past_cov, n_future_cov, static
    ("no_covariate",         1, 0, 0, False),
    ("static_only",          1, 0, 0, True),
    ("past_and_future_covs", 1, 2, 3, False),
    ("multivariate_target",  3, 2, 3, False),
    # Equal covariate widths: the case a width-only assertion cannot catch.
    ("equal_width_covs",     1, 2, 2, False),
]


def build_inputs(n_target, n_past, n_future, static, n=400, horizon=12):
    target = make_series(n_target, 0.0, n)
    if static:
        target = target.with_static_covariates(pd.DataFrame({"s0": [7.5], "s1": [8.5]}))
    past = make_series(n_past, 100.0, n) if n_past else None
    fut = make_series(n_future, 200.0, n + horizon) if n_future else None
    return target, past, fut


def run_layout(lookback: int, horizon: int, hidden: int) -> pd.DataFrame:
    banner("LAYOUT -- segment widths and channel layout (Section 4.4)")
    rows = []

    for name, n_t, n_p, n_f, static in LAYOUT_CONFIGS:
        target, past, fut = build_inputs(n_t, n_p, n_f, static, horizon=horizon)
        try:
            m = GATiDEModel(**base_kwargs(lookback, horizon, hidden,
                                          use_static_covariates=static),
                            num_attn_heads=4)
            m.fit(target, past_covariates=past, future_covariates=fut, verbose=False)
            pred = m.predict(n=horizon, past_covariates=past, future_covariates=fut)
            ok = len(pred) == horizon and pred.width == n_t and \
                bool(np.isfinite(pred.values()).all())
        except Exception as exc:
            rows.append({"config": name, "check": "build+predict",
                         "status": "FAIL", "detail": f"{type(exc).__name__}: {exc}"})
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
            continue

        rows.append({"config": name, "check": "build+predict",
                     "status": "PASS" if ok else "FAIL",
                     "detail": f"pred ({len(pred)}, {pred.width})"})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} build+predict")

        net = m.model
        fusion = getattr(net, "segment_fusion", None)
        if fusion is None:
            rows.append({"config": name, "check": "segment_dims", "status": "PASS",
                         "detail": "single segment; fusion disabled by design"})
            print(f"  [PASS] {name} segment_dims (fusion disabled)")
            continue

        declared = [p.in_features for p in fusion.projections]
        seen = []
        original = fusion.forward

        def spy(segments, _orig=original, _seen=seen):
            _seen.append([int(s.shape[-1]) for s in segments])
            return _orig(segments)

        fusion.forward = spy
        try:
            m.predict(n=horizon, past_covariates=past, future_covariates=fut)
        finally:
            fusion.forward = original

        agree = bool(seen) and seen[0] == declared == list(net._segment_dims)
        rows.append({"config": name, "check": "segment_dims",
                     "status": "PASS" if agree else "FAIL",
                     "detail": f"init={net._segment_dims}, forward={seen[0] if seen else None}"})
        print(f"  [{'PASS' if agree else 'FAIL'}] {name} segment_dims {declared}")

    # -- channel layout, by value ------------------------------------------
    target, past, fut = build_inputs(1, 2, 2, False, horizon=horizon)
    m = GATiDEModel(**base_kwargs(lookback, horizon, hidden), num_attn_heads=4)
    m.fit(target, past_covariates=past, future_covariates=fut, verbose=False)
    net = m.model

    captured = {}
    original_fwd = net.forward

    def spy_fwd(x_in, _orig=original_fwd, _cap=captured):
        _cap["x"] = x_in[0].detach().clone()
        return _orig(x_in)

    net.forward = spy_fwd
    try:
        m.predict(n=horizon, past_covariates=past, future_covariates=fut)
    finally:
        net.forward = original_fwd

    x = captured["x"]
    d_y, d_p, d_f = net.output_dim, net.past_cov_dim, net.future_cov_dim
    tgt, pst, fth = x[..., :d_y], x[..., d_y:d_y + d_p], x[..., -d_f:]
    layout_ok = (float(tgt.max()) < 100.0
                 and 100.0 <= float(pst.min()) and float(pst.max()) < 200.0
                 and float(fth.min()) >= 200.0)

    print(f"\n  channel layout of `x` (output_dim={d_y}, past={d_p}, future={d_f}):")
    print(f"    target   [{float(tgt.min()):8.2f}, {float(tgt.max()):8.2f}]  expected < 100")
    print(f"    past     [{float(pst.min()):8.2f}, {float(pst.max()):8.2f}]  expected [100, 200)")
    print(f"    hist-fut [{float(fth.min()):8.2f}, {float(fth.max()):8.2f}]  expected >= 200")
    print(f"  [{'PASS' if layout_ok else 'FAIL'}] channel layout verified by value")

    rows.append({"config": "equal_width_covs", "check": "channel_layout",
                 "status": "PASS" if layout_ok else "FAIL",
                 "detail": "[target | past | historic future]"})

    df = pd.DataFrame(rows)
    save(df, "layout.csv")
    return df


# --------------------------------------------------------------------------- #
# --layernorm
# --------------------------------------------------------------------------- #
def synthetic_batch(net, batch_size: int = 64, seed: int = 0):
    """One PLModuleInput batch built directly from the network's own dims."""
    g = torch.Generator().manual_seed(seed)
    dt = next(net.parameters()).dtype
    d_y, d_p, d_f = net.output_dim, net.past_cov_dim, net.future_cov_dim
    Lc, Hc = net.input_chunk_length, net.output_chunk_length

    x = torch.randn(batch_size, Lc, d_y + d_p + d_f, generator=g).to(dt)
    x_future = torch.randn(batch_size, Hc, d_f, generator=g).to(dt)
    x_static = (
        torch.randn(batch_size, net.static_cov_dim, generator=g).to(dt)
        if net.static_cov_dim else None
    )
    return x, x_future, x_static, None


def _grad_sum(module) -> float:
    return float(sum(p.grad.abs().sum() for p in module.parameters()
                     if p.grad is not None))


def run_layernorm(lookback: int, horizon: int, hidden: int) -> pd.DataFrame:
    banner("LAYERNORM -- unit-width degeneracy (Section 4.3)")
    rows = []

    # -- block level --------------------------------------------------------
    print("\n  Block level")
    print(f"  {'block':<44}{'out var':>12}{'|d/dx|':>12}{'|d/dW|':>12}")
    torch.manual_seed(0)
    x0 = torch.randn(64, 20)

    for label, blk in [
        ("_ResidualBlock  out=1  LN=True",
         _ResidualBlock(20, 1, 32, 0.0, True)),
        ("_ResidualBlock  out=1  LN=False",
         _ResidualBlock(20, 1, 32, 0.0, False)),
        ("GatedResidualBlock  out=1  LN=True (guarded)",
         GatedResidualBlock(20, 1, 32, 0.0, True)),
    ]:
        xi = x0.clone().requires_grad_(True)
        out = blk(xi)
        out.sum().backward()
        ov, gi, gw = (float(out.detach().var()), float(xi.grad.abs().sum()),
                      _grad_sum(blk))
        print(f"  {label:<44}{ov:>12.3e}{gi:>12.3e}{gw:>12.3e}")
        rows.append({"level": "block", "arm": label, "out_var": ov,
                     "grad_input": gi, "grad_params": gw})

    print("\n  Under the degeneracy the ONLY parameter receiving gradient is the\n"
          "  LayerNorm bias, whose gradient equals the batch size (one per sample).")

    # -- network level ------------------------------------------------------
    print("\n  Network level (univariate target, use_layer_norm=True)")
    target, past, fut = build_inputs(1, 2, 3, False, horizon=horizon)

    # NOTE: GA-TiDE differs from the baseline in the LayerNorm guard, the gate
    # AND the segment-attention fusion. This comparison therefore shows that
    # gradient flow is restored, not that the guard alone restores it.
    for label, cls in [("TiDE (unguarded)", TiDEModel),
                       ("GA-TiDE (guarded)", GATiDEModel)]:
        kw = base_kwargs(lookback, horizon, hidden, use_layer_norm=True)
        if cls is GATiDEModel:
            kw["num_attn_heads"] = 4
        m = cls(**kw)
        m.fit(target, past_covariates=past, future_covariates=fut, verbose=False)

        # An explicit forward/backward. Reading `p.grad` after `fit()` returns
        # measures nothing: Lightning calls zero_grad(set_to_none=True) around
        # the optimizer step, so every gradient is None by then.
        net = m.model
        net.eval()
        net.zero_grad(set_to_none=True)
        net(synthetic_batch(net)).pow(2).mean().backward()

        enc = _grad_sum(net.encoders[0])
        skp = _grad_sum(net.lookback_skip)
        td = _grad_sum(net.temporal_decoder)
        ln = getattr(net.temporal_decoder, "layer_norm", None)
        ln_b = float(ln.bias.grad.abs().sum()) if ln is not None and ln.bias.grad is not None else float("nan")

        print(f"\n  {label}")
        print(f"    temporal_decoder LayerNorm : {'None (skipped)' if ln is None else ln}")
        print(f"    grad, first encoder block  : {enc:.6e}")
        print(f"    grad, lookback skip        : {skp:.6e}")
        print(f"    ratio, skip / encoder      : {skp / enc if enc else float('inf'):.4e}")
        rows.append({"level": "network", "arm": label, "grad_encoder0": enc,
                     "grad_lookback_skip": skp, "grad_temporal_decoder": td,
                     "grad_layernorm_bias": ln_b,
                     "ratio_skip_over_encoder": (skp / enc if enc else float("inf"))})

    df = pd.DataFrame(rows)
    save(df, "layernorm.csv")
    return df


# --------------------------------------------------------------------------- #
# --precision
# --------------------------------------------------------------------------- #
def run_precision(lookback: int, horizon: int, hidden: int) -> pd.DataFrame:
    banner("PRECISION -- is the residual gradient numerical or analytic?")

    print("\n  Part 1 -- nn.LayerNorm(1) alone")
    print(f"  {'dtype':<10}{'shape':<16}{'|x_hat| max':>14}{'|d out/d x|':>16}")
    rows = []
    for dtype in (torch.float32, torch.float64):
        for shape in ((64, 1), (64, horizon, 1)):
            ln = torch.nn.LayerNorm(1).to(dtype)
            x = torch.randn(*shape, dtype=dtype, requires_grad=True)
            y = ln(x)
            y.sum().backward()
            with torch.no_grad():
                x_hat = (y - ln.bias) / ln.weight
            xh, gx = float(x_hat.abs().max()), float(x.grad.abs().sum())
            print(f"  {str(dtype).split('.')[-1]:<10}{str(tuple(shape)):<16}{xh:>14.3e}{gx:>16.3e}")
            rows.append({"part": "operator", "dtype": str(dtype).split(".")[-1],
                         "shape": str(tuple(shape)), "x_hat_max": xh, "grad_input": gx})

    print("\n  Part 2 -- vanilla TiDE network, float32 vs float64")
    target, past, fut = build_inputs(1, 2, 3, False, horizon=horizon)
    m = TiDEModel(**base_kwargs(lookback, horizon, hidden, use_layer_norm=True))
    m.fit(target, past_covariates=past, future_covariates=fut, verbose=False)
    net = m.model

    for dtype in (torch.float32, torch.float64):
        nd = net.to(dtype)
        nd.eval()
        nd.zero_grad(set_to_none=True)
        nd(synthetic_batch(nd)).pow(2).mean().backward()

        enc, skp = _grad_sum(nd.encoders[0]), _grad_sum(nd.lookback_skip)
        td = _grad_sum(nd.temporal_decoder)
        lnw = float(nd.temporal_decoder.layer_norm.weight.grad.abs().sum())
        lnb = float(nd.temporal_decoder.layer_norm.bias.grad.abs().sum())
        rows.append({"part": "network", "dtype": str(dtype).split(".")[-1],
                     "grad_encoder0": enc, "grad_lookback_skip": skp,
                     "grad_temporal_decoder": td, "ln_weight": lnw, "ln_bias": lnb,
                     "ratio_skip_over_encoder": skp / enc if enc else float("inf"),
                     "ln_bias_share": lnb / td if td else float("nan")})

    df = pd.DataFrame(rows)
    net_rows = df[df["part"] == "network"].set_index("dtype")
    e32 = float(net_rows.loc["float32", "grad_encoder0"])
    e64 = float(net_rows.loc["float64", "grad_encoder0"])
    ratio = e32 / e64 if e64 else float("inf")
    eps_ratio = float(np.finfo(np.float32).eps / np.finfo(np.float64).eps)

    print(f"\n  encoder gradient, float32 : {e32:.4e}")
    print(f"  encoder gradient, float64 : {e64:.4e}")
    print(f"  ratio                     : {ratio:.3e}")
    print(f"  ratio of machine epsilons : {eps_ratio:.3e}")
    if ratio > 1e3:
        print("\n  VERDICT: the surviving gradient is floating-point residue. The block\n"
              "  is analytically constant. Say 'suppressed to numerical residue',\n"
              "  never 'blocked', and report the encoder-to-skip RATIO, which is\n"
              "  what determines whether the stack trains.")
    else:
        print("\n  VERDICT: the gradient survives in float64 -- a real path reaches the\n"
              "  encoder that the Section 4.3 argument does not account for. Locate it\n"
              "  before making any claim.")

    save(df, "precision.csv")
    return df


# --------------------------------------------------------------------------- #
# --params
# --------------------------------------------------------------------------- #
def run_params(lookback: int, horizon: int, hidden: int) -> pd.DataFrame:
    banner("PARAMS -- parameter count, TiDE vs GA-TiDE (Section 5)")
    target, past, fut = build_inputs(1, 2, 3, False, n=lookback + horizon + 600,
                                     horizon=horizon)

    rows = []
    for label, cls in MODELS.items():
        kw = base_kwargs(lookback, horizon, hidden)
        if cls is GATiDEModel:
            kw["num_attn_heads"] = 4
        m = cls(**kw)
        m.fit(target, past_covariates=past, future_covariates=fut, verbose=False)

        total = sum(p.numel() for p in m.model.parameters() if p.requires_grad)
        fusion = getattr(m.model, "segment_fusion", None)
        n_fusion = sum(p.numel() for p in fusion.parameters()) if fusion else 0
        rows.append({"model": label, "params": total, "fusion_params": n_fusion})

    df = pd.DataFrame(rows)
    base = int(df.loc[df["model"] == "TiDE (baseline)", "params"].iloc[0])
    df["delta"] = df["params"] - base
    df["delta_pct"] = ((df["params"] / base - 1) * 100).round(1)
    # The remainder after removing the fusion module. It can be NEGATIVE:
    # fusion narrows the encoder's first-layer input from the concatenated
    # segment width to n_segments * hidden_size, and that saving can exceed
    # what gating adds. It is therefore gating AND the narrowing combined, not
    # gating alone -- separating them needs a gated-without-fusion variant.
    df["non_fusion_delta"] = df["delta"] - df["fusion_params"]
    df["lookback"] = lookback
    df["hidden_size"] = hidden

    print(f"\n  L={lookback}, H={horizon}, hidden_size={hidden}, univariate target\n")
    print(df.to_string(index=False))
    print("\n  `non_fusion_delta` combines gating with the encoder-input narrowing\n"
          "  that fusion also causes; it is not the cost of gating alone. Recompute\n"
          "  at the production configuration before quoting numbers in the paper.")

    save(df, "parameters.csv")
    return df


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layout", action="store_true")
    p.add_argument("--layernorm", action="store_true")
    p.add_argument("--precision", action="store_true")
    p.add_argument("--params", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--lookback", type=int, default=48)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--hidden-size", type=int, default=32)
    args = p.parse_args()

    if not any([args.layout, args.layernorm, args.precision, args.params, args.all]):
        p.error("choose at least one of --layout --layernorm --precision --params --all")

    report_env()
    L, H, W = args.lookback, args.horizon, args.hidden_size

    if args.all or args.layout:
        run_layout(L, H, W)
    if args.all or args.layernorm:
        run_layernorm(L, H, W)
    if args.all or args.precision:
        run_precision(L, H, W)
    if args.all or args.params:
        run_params(L, H, W)


if __name__ == "__main__":
    main()

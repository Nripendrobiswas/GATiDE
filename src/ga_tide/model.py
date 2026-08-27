"""
GA-TiDE wired into Darts
=========================

Drop-in subclass of Darts' `TiDEModel` / `_TideModule`. Two changes vs.
vanilla TiDE, both informed by reading Darts' actual source
(darts/models/forecasting/tide_model.py):

1. `_ResidualBlock` -> `GatedResidualBlock`: every residual MLP block
   becomes a gated (GRN-style) block. Also fixes a latent bug present in
   Darts' own `_ResidualBlock` when `use_layer_norm=True` and an
   `output_dim == 1` block is used (e.g. `temporal_decoder` when
   `output_dim * nr_params == 1`): `nn.LayerNorm(1)` normalizes any input
   to exactly 0, silently killing all upstream gradient. We only apply
   LayerNorm when output_dim > 1. (Darts defaults `use_layer_norm=False`,
   so this doesn't bite by default -- but it will the moment someone
   turns layer norm on with a univariate target, which is exactly our
   electricity-load setup.)

2. Segment-attention fusion at the encoder's input. Darts' encoder input
   is NOT a sequence -- x_lookback, past-covariate features, future
   covariate features, and static covariates are each flattened to a
   vector and concatenated (`torch.cat` of flattened segments) before
   the residual stack ever sees them. There is no timestep axis left at
   that point, so a per-timestep attention layer (my first draft) does
   not apply here. Instead we project each present segment to a common
   `hidden_size`, treat them as a handful of tokens, and run one
   self-attention layer across segments before flattening back down into
   the same residual stack Darts already uses. This lets e.g. the
   future-covariate segment (temperature/humidity/holiday) attend to the
   lookback-target segment before the encoder compresses everything,
   instead of the model only ever seeing them pre-mixed by concatenation.

Everything else (decoder stack, temporal decoder, lookback skip
connection, PLForecastingModule plumbing, fit/predict/historical
forecasts, Optuna-friendliness) is untouched -- inherited straight from
Darts' `TiDEModel` / `MixedCovariatesTorchModel`.
"""

from typing import Optional

import torch
import torch.nn as nn

from darts.models.forecasting.pl_forecasting_module import io_processor
from darts.models.forecasting.tide_model import TiDEModel, _TideModule
from darts.utils.data.torch_datasets.utils import PLModuleInput, TorchTrainingSample
from darts.utils.torch import MonteCarloDropout


class GatedResidualBlock(nn.Module):
    """Drop-in replacement for Darts' `_ResidualBlock`: same constructor
    signature, gated (GRN-style) internals instead of plain MLP + skip."""

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int,
                 dropout: float, use_layer_norm: bool):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_dim)
        self.dropout = MonteCarloDropout(dropout)
        self.gate = nn.Linear(input_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim)
        self.act = nn.ReLU()

        # see module docstring point (1): LayerNorm(1) is degenerate and
        # would zero out all upstream gradient, so only enable it when
        # output_dim > 1, regardless of what the caller asked for.
        self.layer_norm = (
            nn.LayerNorm(output_dim) if (use_layer_norm and output_dim > 1)
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.dropout(self.act(self.fc1(x))))
        g = torch.sigmoid(self.gate(x))
        out = self.skip(x) + g * h
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        return out


class SegmentAttentionFusion(nn.Module):
    """Projects each present input segment (lookback target, past-cov
    features, future-cov features, static covariates) to a common width
    and runs one self-attention layer across them as tokens, before
    flattening back down to feed the existing residual encoder stack."""

    def __init__(self, segment_dims: list[int], hidden_size: int,
                 num_heads: int, dropout: float):
        super().__init__()
        if len(segment_dims) < 2:
            raise ValueError(
                "SegmentAttentionFusion needs >= 2 segments to attend "
                "over (e.g. target + at least one covariate/static group); "
                "got a config with only one input segment."
            )
        self.projections = nn.ModuleList(
            [nn.Linear(d, hidden_size) for d in segment_dims]
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.output_dim = len(segment_dims) * hidden_size

    def forward(self, segments: list[torch.Tensor]) -> torch.Tensor:
        tokens = torch.stack(
            [proj(seg) for proj, seg in zip(self.projections, segments)],
            dim=1,
        )  # (batch, n_segments, hidden_size)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        fused = self.norm(tokens + attn_out)
        return fused.flatten(start_dim=1)  # (batch, n_segments * hidden_size)


class _GATideModule(_TideModule):
    """Same as Darts' `_TideModule`, with GatedResidualBlock throughout
    and segment-attention fusion at the encoder input."""

    def __init__(self, *args, num_attn_heads: int = 4, **kwargs):
        # let the parent build everything exactly as vanilla TiDE would
        # (decoders, temporal_decoder, lookback_skip, past/future cov
        # projections, and a self.encoders stack we're about to replace)
        super().__init__(*args, **kwargs)
        self.num_attn_heads = num_attn_heads

        # --- rebuild past/future covariate projections and encoder as
        # gated blocks (parent already built these as _ResidualBlock;
        # swap them 1:1 so shapes stay identical to what the rest of
        # forward() expects) ---
        if self.past_cov_projection is not None:
            self.past_cov_projection = GatedResidualBlock(
                input_dim=self.past_cov_dim,
                output_dim=self.temporal_width_past,
                hidden_size=self.temporal_hidden_size_past,
                use_layer_norm=self.use_layer_norm,
                dropout=self.dropout,
            )
        if self.future_cov_projection is not None:
            self.future_cov_projection = GatedResidualBlock(
                input_dim=self.future_cov_dim,
                output_dim=self.temporal_width_future,
                hidden_size=self.temporal_hidden_size_future,
                use_layer_norm=self.use_layer_norm,
                dropout=self.dropout,
            )

        # figure out segment dims exactly as parent's __init__ did, so
        # SegmentAttentionFusion's projections match what forward() will
        # actually hand it
        past_covariates_flat_dim = 0
        if self.past_cov_dim and self.temporal_width_past:
            past_covariates_flat_dim = self.input_chunk_length * self.temporal_width_past
        elif self.past_cov_dim:
            past_covariates_flat_dim = self.input_chunk_length * self.past_cov_dim

        future_flat_dim = 0
        if self.future_cov_dim and self.temporal_width_future:
            future_flat_dim = (
                (self.input_chunk_length + self.output_chunk_length)
                * self.temporal_width_future
            )
        elif self.future_cov_dim:
            future_flat_dim = (
                (self.input_chunk_length + self.output_chunk_length)
                * self.future_cov_dim
            )

        segment_dims = [self.input_chunk_length * self.output_dim]  # x_lookback, always present
        if past_covariates_flat_dim:
            segment_dims.append(past_covariates_flat_dim)
        if future_flat_dim:
            segment_dims.append(future_flat_dim)
        if self.static_cov_dim:
            segment_dims.append(self.static_cov_dim)
        self._segment_dims = segment_dims

        if len(segment_dims) >= 2:
            self.segment_fusion = SegmentAttentionFusion(
                segment_dims=segment_dims, hidden_size=self.hidden_size,
                num_heads=num_attn_heads, dropout=self.dropout,
            )
            fused_dim = self.segment_fusion.output_dim
        else:
            # only the target lookback is present (no covariates, no
            # static): nothing to attend over, fall back to a plain
            # linear projection so the rest of the pipeline is unaffected
            self.segment_fusion = None
            fused_dim = segment_dims[0]

        self.encoders = nn.Sequential(
            GatedResidualBlock(
                input_dim=fused_dim, output_dim=self.hidden_size,
                hidden_size=self.hidden_size,
                use_layer_norm=self.use_layer_norm, dropout=self.dropout,
            ),
            *[
                GatedResidualBlock(
                    input_dim=self.hidden_size, output_dim=self.hidden_size,
                    hidden_size=self.hidden_size,
                    use_layer_norm=self.use_layer_norm, dropout=self.dropout,
                )
                for _ in range(self.num_encoder_layers - 1)
            ],
        )

        self.decoders = nn.Sequential(
            *[
                GatedResidualBlock(
                    input_dim=self.hidden_size, output_dim=self.hidden_size,
                    hidden_size=self.hidden_size,
                    use_layer_norm=self.use_layer_norm, dropout=self.dropout,
                )
                for _ in range(self.num_decoder_layers - 1)
            ],
            GatedResidualBlock(
                input_dim=self.hidden_size,
                output_dim=self.decoder_output_dim * self.output_chunk_length * self.nr_params,
                hidden_size=self.hidden_size,
                use_layer_norm=self.use_layer_norm, dropout=self.dropout,
            ),
        )

        decoder_input_dim = self.decoder_output_dim * self.nr_params
        if self.temporal_width_future and self.future_cov_dim:
            decoder_input_dim += self.temporal_width_future
        elif self.future_cov_dim:
            decoder_input_dim += self.future_cov_dim

        self.temporal_decoder = GatedResidualBlock(
            input_dim=decoder_input_dim,
            output_dim=self.output_dim * self.nr_params,
            hidden_size=self.temporal_decoder_hidden,
            use_layer_norm=self.use_layer_norm, dropout=self.dropout,
        )

    @io_processor
    def forward(self, x_in: PLModuleInput) -> torch.Tensor:
        x, x_future_covariates, x_static_covariates, _ = x_in
        x_lookback = x[:, :, : self.output_dim]

        if self.future_cov_dim:
            x_dynamic_future_covariates = torch.cat(
                [
                    x[:, :, -self.future_cov_dim:],
                    x_future_covariates,
                ],
                dim=1,
            )
            if self.temporal_width_future:
                x_dynamic_future_covariates = self.future_cov_projection(
                    x_dynamic_future_covariates
                )
        else:
            x_dynamic_future_covariates = None

        if self.past_cov_dim:
            x_dynamic_past_covariates = x[
                :, :, self.output_dim: self.output_dim + self.past_cov_dim,
            ]
            if self.temporal_width_past:
                x_dynamic_past_covariates = self.past_cov_projection(
                    x_dynamic_past_covariates
                )
        else:
            x_dynamic_past_covariates = None

        segments = [
            x_lookback, x_dynamic_past_covariates,
            x_dynamic_future_covariates, x_static_covariates,
        ]
        segments = [t.flatten(start_dim=1) for t in segments if t is not None]

        if self.segment_fusion is not None:
            expected_dims = [p.in_features for p in self.segment_fusion.projections]
            actual_dims = [s.shape[-1] for s in segments]
            if actual_dims != expected_dims:
                raise RuntimeError(
                    f"GA-TiDE segment shape mismatch: forward() produced flattened "
                    f"segment dims {actual_dims} but SegmentAttentionFusion was built "
                    f"expecting {expected_dims}. This usually means the installed "
                    f"Darts version has changed its internal tensor layout for `x` / "
                    f"covariates since this GA-TiDE subclass was written -- check "
                    f"darts.models.forecasting.tide_model._TideModule against the "
                    f"assumptions documented in this file's module docstring."
                )

        fused = self.segment_fusion(segments) if self.segment_fusion is not None else segments[0]

        encoded = self.encoders(fused)
        decoded = self.decoders(encoded)
        decoded = decoded.view(x.shape[0], self.output_chunk_length, -1)

        temporal_decoder_input = [
            decoded,
            (x_dynamic_future_covariates[:, -self.output_chunk_length:, :]
             if self.future_cov_dim > 0 else None),
        ]
        temporal_decoder_input = [t for t in temporal_decoder_input if t is not None]
        temporal_decoder_input = torch.cat(temporal_decoder_input, dim=2)
        temporal_decoded = self.temporal_decoder(temporal_decoder_input)

        skip = self.lookback_skip(x_lookback.transpose(1, 2)).transpose(1, 2)
        y = temporal_decoded + skip.reshape_as(temporal_decoded)
        y = y.view(-1, self.output_chunk_length, self.output_dim, self.nr_params)
        return y


class GATiDEModel(TiDEModel):
    """Same public API as `darts.models.TiDEModel` (fit / predict /
    historical_forecasts / gridsearch / your Optuna objective all work
    unchanged) with one extra constructor arg: `num_attn_heads`.

    NOTE: Darts' `ModelMeta` metaclass captures constructor parameters by
    introspecting `cls.__init__`'s signature directly (see
    forecasting_model.py: `ModelMeta.__call__`), so this signature is
    written out explicitly (mirroring `TiDEModel.__init__`) rather than
    as a `*args, **kwargs` passthrough -- the latter breaks parameter
    capture (positional args get bound to the wrong names) and silently
    corrupts things like `output_chunk_shift`.
    """

    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        decoder_output_dim: int = 16,
        hidden_size: int = 128,
        temporal_width_past: int = 4,
        temporal_width_future: int = 4,
        temporal_hidden_size_past: Optional[int] = None,
        temporal_hidden_size_future: Optional[int] = None,
        temporal_decoder_hidden: int = 32,
        use_layer_norm: bool = False,
        dropout: float = 0.1,
        use_static_covariates: bool = True,
        num_attn_heads: int = 4,
        **kwargs,
    ):
        if hidden_size % num_attn_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attn_heads "
                f"({num_attn_heads}) -- the Segment Attention Fusion module's "
                f"nn.MultiheadAttention requires embed_dim % num_heads == 0. "
                f"Choose a hidden_size that is a multiple of num_attn_heads, or "
                f"vice versa."
            )
        self.num_attn_heads = num_attn_heads
        super().__init__(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            decoder_output_dim=decoder_output_dim,
            hidden_size=hidden_size,
            temporal_width_past=temporal_width_past,
            temporal_width_future=temporal_width_future,
            temporal_hidden_size_past=temporal_hidden_size_past,
            temporal_hidden_size_future=temporal_hidden_size_future,
            temporal_decoder_hidden=temporal_decoder_hidden,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            use_static_covariates=use_static_covariates,
            **kwargs,
        )

    def _create_model(self, train_sample: TorchTrainingSample) -> torch.nn.Module:
        (
            past_target, past_covariates, historic_future_covariates,
            future_covariates, static_covariates, future_target,
        ) = train_sample

        input_dim = (
            past_target.shape[1]
            + (past_covariates.shape[1] if past_covariates is not None else 0)
            + (historic_future_covariates.shape[1]
               if historic_future_covariates is not None else 0)
        )
        output_dim = future_target.shape[1]
        future_cov_dim = (
            future_covariates.shape[1] if future_covariates is not None else 0
        )
        static_cov_dim = (
            static_covariates.shape[0] * static_covariates.shape[1]
            if static_covariates is not None else 0
        )
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        return _GATideModule(
            input_dim=input_dim,
            output_dim=output_dim,
            future_cov_dim=future_cov_dim,
            static_cov_dim=static_cov_dim,
            nr_params=nr_params,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            decoder_output_dim=self.decoder_output_dim,
            hidden_size=self.hidden_size,
            temporal_width_past=self.temporal_width_past,
            temporal_width_future=self.temporal_width_future,
            temporal_hidden_size_past=self.temporal_hidden_size_past,
            temporal_hidden_size_future=self.temporal_hidden_size_future,
            temporal_decoder_hidden=self.temporal_decoder_hidden,
            use_layer_norm=self.use_layer_norm,
            dropout=self.dropout,
            num_attn_heads=self.num_attn_heads,
            **self.pl_module_params,
        )

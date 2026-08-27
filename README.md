# GA-TiDE: Gated-Attention Time-series Dense Encoder

Reference implementation and benchmark code for *[paper title]*.

GA-TiDE extends TiDE (Das et al., 2023) with two modifications:

1. a **gated residual block** — a sigmoid gate on the nonlinear branch of
   TiDE's residual block, leaving the activation, dropout placement and linear
   skip unchanged;
2. **Segment Attention Fusion** — the flattened input segments (target
   lookback, past covariates, future covariates, static attributes) are
   projected to a common width, treated as tokens, and passed through one
   multi-head self-attention layer before entering the encoder, instead of
   being concatenated directly.

It is implemented as a drop-in subclass of the Darts `TiDEModel`: `fit`,
`predict`, `historical_forecasts` and `save`/`load` behave identically.

> **Status:** [pre-review / under review / published]. Results in this
> repository correspond to [commit or tag].

---

## Installation

```bash
git clone https://github.com/<user>/ga-tide.git
cd ga-tide
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

The `darts` version is pinned deliberately — GA-TiDE subclasses private Darts
internals. See `requirements.txt`.

## Quick start

```python
from ga_tide import GATiDEModel

model = GATiDEModel(
    input_chunk_length=720,
    output_chunk_length=96,
    hidden_size=256,
    num_attn_heads=4,        # must divide hidden_size
)
model.fit(series, future_covariates=covariates)
pred = model.predict(n=96, future_covariates=covariates)
```

Both modifications are always active; there is no switch to enable either in
isolation. The baseline is therefore Darts' stock `TiDEModel`, and the
comparison is between two models:

| Name | Class | Blocks | Input fusion |
|---|---|---|---|
| `tide` | `darts.models.TiDEModel` | `_ResidualBlock` | concatenation |
| `ga-tide` | `ga_tide.GATiDEModel` | `GatedResidualBlock` | segment attention |

### Known limitation of this comparison

`tide` and `ga-tide` differ in **three** respects simultaneously:

1. the sigmoid gate on the nonlinear branch;
2. segment-attention fusion, which also narrows the encoder's first-layer input
   from the concatenated segment width to `n_segments × hidden_size`;
3. the dropout position inside the residual block — `GatedResidualBlock` applies
   dropout to the hidden activation, whereas `_ResidualBlock` applies it after
   the second linear map.

A difference in accuracy is consequently not attributable to any single one of
them. Isolating them would require constructor switches for the block type and
the fusion mode, which this implementation does not expose. Report the
comparison as between two models, not as an ablation.

## Reproducing the paper

```bash
# 0. Verify the environment and produce the diagnostic tables (~1 min, CPU).
python scripts/run_diagnostics.py --all

# 1. Both models on one dataset/horizon.
python scripts/benchmark.py --dataset ETTh1 --horizon 96 --model all --seeds 3

# 2. Main results across all datasets and horizons.
python scripts/benchmark.py --dataset all --all-horizons --model all --seeds 3
```

Results append to `results/results.csv`, keyed on
`(dataset, horizon, model, seed)`; re-running a key overwrites its row.

| Paper artefact | Command | Output |
|---|---|---|
| Main results table | `benchmark.py --dataset all --all-horizons --model all` | `results/results.csv` |
| Parameter-cost table | `run_diagnostics.py --params` | `results/parameters.csv` |
| LayerNorm diagnostic | `run_diagnostics.py --layernorm` | `results/layernorm.csv` |
| Precision check | `run_diagnostics.py --precision` | `results/precision.csv` |
| Layout verification | `run_diagnostics.py --layout` | `results/layout.csv` |

Hyperparameters are searched separately per model with an equal trial budget:

```bash
python scripts/tune_optuna.py --dataset ETTh1 --horizon 96 --model tide    --n-trials 50
python scripts/tune_optuna.py --dataset ETTh1 --horizon 96 --model ga-tide --n-trials 50
```

## Benchmark protocol

Three choices determine comparability with published TiDE numbers. All are
command-line flags and all are stated in the paper.

- **Channel independence** (`--channel-independent`, default). Each dataset is
  split into univariate series and one global model is fitted across them, as
  in TiDE. `--multivariate` fits a single joint model instead; that is a
  different model and is not comparable to published numbers.
- **Splits** (`--split-convention`, default `prior-work`). 6:2:2 for the ETT
  family, 7:1:2 elsewhere. `tide` selects 7:1:2 throughout.
- **Data source** (`--source`, default `darts`). Use `--source csv` with the
  LTSF benchmark CSVs for any number reported against prior work; Darts'
  bundled Electricity loader is a different series from the ECL benchmark. See
  `data/README.md`.

Evaluation is rolling-origin with stride 1 over the test period, metrics
averaged over origins and channels on the standardized scale.

## Tests

```bash
pytest -v
```

`tests/test_segment_layout.py` verifies that GA-TiDE's re-derivation of the
input segment widths matches what Darts produces, and checks the channel
layout of the covariate tensor **by value** rather than by width — a width
check alone cannot detect a reordering when the past and future covariate
groups have the same number of features. It fails loudly if a Darts upgrade
changes either.

## Environment used for the reported results

darts 0.46.1 · torch 2.10.0+cu128 · pytorch-lightning 2.6.5 · numpy 2.0.2

## Citation

```bibtex
@article{biswas2026gatide,
  title  = {[paper title]},
  author = {Biswas, Nripendro},
  year   = {2026}
}
```

## License

Apache License 2.0. Contains code derived from
[Darts](https://github.com/unit8co/darts) (Unit8 SA, Apache 2.0); see
[`NOTICE`](NOTICE) for the specific files and modifications.

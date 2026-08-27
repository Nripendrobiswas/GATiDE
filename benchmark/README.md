# GATiDE Benchmark – Production-Ready Forecasting Evaluation

Modular PyTorch benchmark for evaluating **GATiDE** (`src/ga_tide/model.py`) against standard baselines
**TiDE, DLinear, PatchTST, Naive (persistence)** on the LTSF datasets available in `data/` (ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Weather, Traffic).

Implements the exact protocol requested:

- **Lookback** `L=720`, **Horizons** `H ∈ {96,192,336,720}`
- **Sequential split** `70% train / 10% val / 20% test` with **train-only StandardScaler** (zero-mean, unit-variance) to prevent leakage
- **MSE loss + AdamW + CosineAnnealing/StepLR + EarlyStopping (val_loss)**
- **Inverse-scaled MSE/MAE** on the Test set
- **Throughput:** training time per epoch (s) and **GPU peak memory (MB)** via `torch.cuda.max_memory_allocated`
- **Tabular summary** (pandas + tabulate) across models × datasets × horizons, `.npy` predictions, `benchmark_results.csv`

---

## Installation

```bash
# From OPEN CODE directory
pip install -r requirements.txt
# Optional: install GATiDE as editable so `from ga_tide import GATiDEModel` works for Darts path
pip install -e "E:/Machine Learning Research/GATiDE Final Verse/GATiDE"
# OR ensure PYTHONPATH includes GATiDE/src
export PYTHONPATH="E:/Machine Learning Research/GATiDE Final Verse/GATiDE/src:$PYTHONPATH"
```

Required: `torch`, `pandas`, `scikit-learn`, `pyyaml`, `tabulate`, `tqdm`.  
Optional: `darts`, `pytorch-lightning` (only for the Darts-based GATiDE wrapper / original scripts).

---

## Data

Place the LTSF CSVs in `data/` (already present in the sibling `GATiDE/data/`):

```
GATiDE/data/
  ETTh1.csv   ETTh2.csv   ETTm1.csv   ETTm2.csv
  weather.csv electricity.csv traffic.csv
```

Each CSV is `date, <features...>, OT` (or 321/370 columns for Electricity).  
The loader auto-detects the `date` column, sorts chronologically, coerces to numeric, and forward-fills NaNs.

> **Note on Electricity:** the CSV (`electricity.csv`, 321 clients) is the LTSF/ECL benchmark variant; Darts' bundled `ElectricityDataset` (370 clients) is a different series – use `--csv-dir` for comparability.

---

## Quick Start

### Minimal run (CPU, 2 epochs, 1 dataset/horizon, smoke test)

```bash
python run_benchmark.py \
  --csv-dir "E:/Machine Learning Research/GATiDE Final Verse/GATiDE/data" \
  --datasets ETTh1 \
  --horizons 96 \
  --models gatide tide dlinear naive \
  --epochs 2 --batch-size 32 --device cpu --save-dir ./benchmark_outputs_test
```

### Full protocol (L=720, H={96,192,336,720}, all datasets, 100 epochs)

```bash
python run_benchmark.py \
  --csv-dir "E:/Machine Learning Research/GATiDE Final Verse/GATiDE/data" \
  --all-datasets --all-horizons \
  --models all \
  --epochs 100 --batch-size 32 --lr 1e-3 --scheduler cosine --patience 10 \
  --device auto --save-dir ./benchmark_outputs
```

`--models all` expands to `gatide tide dlinear patchtst naive`.

### With YAML config

```bash
python run_benchmark.py --config configs/default.yaml --csv-dir ./GATiDE/data
```

Override any YAML field from CLI, e.g.:

```bash
python run_benchmark.py --config configs/default.yaml --epochs 50 --lr 5e-4 --horizons 96 192
```

---

## Configuration

**`configs/default.yaml`** is the canonical hyperparameter set:

```yaml
data:
  lookback: 720
  horizons: [96,192,336,720]
  split: [0.70,0.10,0.20]

training:
  batch_size: 32
  n_epochs: 100
  lr: 1.0e-3
  optimizer: adamw
  scheduler: cosine
  early_stopping: {patience: 10, min_delta: 1e-4}

models:
  gatide: {hidden_size:256, num_attn_heads:4, ...}
  tide:   {hidden_size:256, ...}
  dlinear: {kernel_size:25}
  patchtst: {patch_len:16, stride:8, d_model:128, n_layers:2, ...}
```

All fields are overridable via CLI flags (`--lr`, `--batch-size`, `--scheduler step`, etc.).

---

## Model Integration

- **GATiDE** – `benchmark/models/gatide_adapter.py` imports `GATiDEModel`, `GatedResidualBlock`, `SegmentAttentionFusion` from `src/ga_tide/model.py` (tries pip-installed `ga_tide` then filesystem sibling).  
  The default benchmark uses the **pure PyTorch reimplementation** `GATiDEPure` (same architecture, no Darts dependency) so all baselines share the identical training loop for fair throughput comparison. Set `use_darts=True` via `get_gatide_model(..., use_darts=True)` to use the original Darts model with `historical_forecasts`.

- **TiDE** – `benchmark/models/tide.py`: vanilla residual blocks, encoder/decoder stack, temporal decoder, lookback skip (Darts-faithful but Darts-free).
- **DLinear** – `benchmark/models/dlinear.py`: moving-average decomposition + per-channel/ shared linear.
- **PatchTST** – `benchmark/models/patchtst.py`: patching + TransformerEncoder (channel-independent).
- **Naive** – `benchmark/models/naive.py`: persistence (last / mean / drift), no training.

Factory: `benchmark/models/__init__.py:MODEL_REGISTRY` and `get_model(name)`.

---

## Training & Evaluation Pipeline

`benchmark/trainer.py:train_one_model`:

- **Loss:** `nn.MSELoss` on scaled windows
- **Optimizer:** `AdamW(lr, weight_decay)` (or `Adam`)
- **Scheduler:** `CosineAnnealingLR(T_max=n_epochs)` or `StepLR(step_size, gamma)` – selectable via `training.scheduler`
- **EarlyStopping:** patience on `val_loss` (MSE scaled); restores best weights
- **Metrics:** inverse-transform predictions via `StandardScaler` then compute `MSE`/`MAE` on original scale
- **Throughput:** `time.time()` per epoch, `torch.cuda.max_memory_allocated()/1e6` for peak MB (0 on CPU)

`benchmark/datasets.py`:

- `load_and_split`: sequential split, `StandardScaler.fit(train_raw)` then `transform` all splits
- `TimeSeriesWindowDataset`: sliding windows `(L,C) -> (H,C)` ; length `T-L-H+1`
- `make_loaders`: DataLoaders

---

## Reporting & Output

After each run (`dataset × horizon × model`) incremental save:

```
benchmark_outputs/
  benchmark_results.csv          # one row per (dataset, horizon, model)
  predictions/
    ETTh1_H96_gatide_pred.npy    # (N_test_windows, H, C) in original scale
    ETTh1_H96_gatide_true.npy
    ETTh1_H96_tide_pred.npy
    ...
```

**CSV columns:**

| dataset | horizon | model | lookback | mse | mae | val_mse | val_mae | train_time_per_epoch_s | peak_memory_mb | epochs_run | n_params | batch_size | lr | status |
|---------|---------|-------|----------|-----|-----|---------|---------|------------------------|----------------|------------|----------|------------|----|--------|

**Console:**

- Per-run: `MSE | MAE | val MSE | s/epoch | peak MB | epochs`
- Final tables via `tabulate` (psql):
  - Full grid `dataset | horizon | model | mse | mae | s/epoch | peak MB`
  - Mean per model `mse, mae` aggregated over datasets/horizons

For post-hoc analysis:

```python
import numpy as np
pred = np.load("benchmark_outputs/predictions/ETTh1_H96_gatide_pred.npy")  # (N,H,C)
true = np.load("benchmark_outputs/predictions/ETTh1_H96_gatide_true.npy")
# e.g., plot first window, channel 0
import matplotlib.pyplot as plt
plt.plot(true[0,:,0], label="true"); plt.plot(pred[0,:,0], label="pred"); plt.legend(); plt.show()
```

---

## Programmatic Use

```python
from benchmark.benchmark import run_benchmark

df = run_benchmark(
    csv_dir="E:/Machine Learning Research/GATiDE Final Verse/GATiDE/data",
    datasets=["ETTh1","weather"],
    horizons=[96,192],
    models=["gatide","tide","dlinear"],
    lookback=720, n_epochs=50, batch_size=32,
    save_dir="./my_results"
)
print(df.groupby(["model","horizon"])[["mse","mae"]].mean())
```

---

## Benchmark Protocol Notes (for paper)

- **Channel independence** is not yet forced – this pipeline trains multivariate (all channels jointly). To reproduce the channel-independent protocol (one global univariate model, as in TiDE paper), split multivariate DataFrame into per-channel series externally or set `num_features=1` and loop over channels. The current pipeline is multivariate joint; state which was used.
- **Splits** are strictly `70/10/20` sequential regardless of dataset (requirement). Prior work uses `6:2:2` for ETT; use `--split` override if exact comparability needed.
- **Normalization** is `StandardScaler` fit on train only, metrics on inverse-scaled (original) values – consistent within this harness; prior tables often report standardized metrics – note the scale in the paper.
- **Covariates:** time-derived covariates (`add_encoders` in `scripts/benchmark.py`) are not used in this pure pipeline (single flattened segment). For Segment Attention Fusion to have ≥2 tokens, extend `benchmark/models/gatide_adapter.py:segment_dims` to include past/future covariate flat dims derived from calendar features.

---

## Extending

Add a new baseline:

1. Create `benchmark/models/my_model.py` with `class MyModel(nn.Module): def __init__(self, num_features, lookback, horizon, ...) / def forward(self, x): -> (B,H,C)`
2. Register in `benchmark/models/__init__.py:MODEL_REGISTRY["my_model"] = MyModel`
3. Run with `--models my_model`

Hyperparameter search: integrate Optuna by wrapping `run_benchmark` or use `scripts/tune_optuna.py` in the sibling repo (per-model equal trial budget).

---

## Citation

GATiDE extends TiDE (Das et al., 2023). When reporting, cite both and note the three simultaneous differences (gate, segment-attention/narrowed encoder input, dropout placement) as documented in `src/ga_tide/model.py:1-38` and `README.md` Known limitation.

```
@inproceedings{das2023tide, title={Long-term Forecasting with TiDE...}, ...}
@article{biswas2026gatide, title={[paper title]}, author={Biswas, Nripendro}, year={2026}}
```

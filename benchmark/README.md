# GATiDE Benchmark

Production-ready PyTorch benchmark: **GATiDE** vs **TiDE, DLinear, PatchTST, Naive** on LTSF datasets.

Implements TiDE-paper protocol (Das et al. 2023, TMLR §5.1):
- `L=720`, `H ∈ {96,192,336,720}`, 7:1:2 split, train-only StandardScaler
- MSE loss + AdamW + CosineAnnealing + EarlyStopping
- Dual metrics: normalized (TiDE Table 2) + inverse-scaled (requirement)
- Throughput: train time/epoch, inference time/batch, GPU peak memory

---

## Install

```bash
git clone https://github.com/Nripendrobiswas/GATiDE.git
cd GATiDE
pip install -r requirements.txt
```

### Kaggle / Colab (no git clone needed)

```python
!pip install torch numpy pandas scikit-learn tabulate pyyaml tqdm optuna
!git clone https://github.com/Nripendrobiswas/GATiDE.git
```

### Windows

```cmd
git clone https://github.com/Nripendrobiswas/GATiDE.git
cd GATiDE
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Data

CSVs go in `GATiDE/data/`:
```
ETTh1.csv  ETTh2.csv  ETTm1.csv  ETTm2.csv
weather.csv  electricity.csv  traffic.csv
```

---

## Usage

### Smoke test (1 epoch, CPU)

```bash
python run_benchmark.py --csv-dir GATiDE/data --datasets ETTh1 --horizons 96 \
  --models naive dlinear --epochs 1 --batch-size 32 --device cpu \
  --save-dir ./benchmark_outputs_smoke
```

### Full benchmark (all models, all horizons, 100 epochs)

```bash
python run_benchmark.py --csv-dir GATiDE/data --all-datasets --all-horizons \
  --models all --epochs 100 --batch-size 32 --device cuda \
  --save-dir ./benchmark_outputs
```

### 5-seed evaluation (TiDE Table 2 mean±std)

```bash
python run_benchmark.py --csv-dir GATiDE/data --all-datasets --all-horizons \
  --models all --seeds 0 1 2 3 4 --epochs 100 --device cuda \
  --save-dir ./benchmark_5seed
```

---

## Tuned Parameters (NO YAML)

Pass tuned hyperparams directly via CLI:

```bash
python run_benchmark.py --csv-dir GATiDE/data --datasets ETTh1 --horizons 96 \
  --models gatide \
  --model-kwargs '{"hidden_size":128,"num_encoder_layers":1,"num_decoder_layers":1,"decoder_output_dim":8,"temporal_decoder_hidden":64,"dropout":0.1,"use_layer_norm":true}' \
  --tune-lr 0.004232010666246005 \
  --tune-batch-size 512 \
  --weight-decay 1e-4 --optimizer adamw --scheduler cosine \
  --patience 10 --grad-clip 1.0 \
  --epochs 100 --device cuda \
  --save-dir ./benchmark_final
```

### CLI flags for tuned params

| Flag | Purpose |
|------|---------|
| `--model-kwargs '{...}'` | JSON applied to ALL models |
| `--model-kwargs-gatide '{...}'` | JSON for GATiDE only |
| `--model-kwargs-tide '{...}'` | JSON for TiDE only |
| `--model-kwargs-dlinear '{...}'` | JSON for DLinear only |
| `--model-kwargs-patchtst '{...}'` | JSON for PatchTST only |
| `--model-kwargs-naive '{...}'` | JSON for Naive only |
| `--tune-lr 0.004` | Override learning rate |
| `--tune-batch-size 512` | Override batch size |

### Kaggle example (tuned GATiDE)

```python
!python GATiDE/run_benchmark.py \
  --csv-dir GATiDE/data --datasets ETTh1 --horizons 96 \
  --models gatide \
  --model-kwargs '{"hidden_size":128,"num_encoder_layers":1,"num_decoder_layers":1,"decoder_output_dim":8,"temporal_decoder_hidden":64,"dropout":0.1,"use_layer_norm":true}' \
  --tune-lr 0.004232010666246005 --tune-batch-size 512 \
  --weight-decay 1e-4 --optimizer adamw --scheduler cosine --patience 10 --grad-clip 1.0 \
  --epochs 100 --device cuda --split-convention tide \
  --save-dir /kaggle/working/benchmark_final
```

---

## Optuna Tuning

Equal-budget hyperparameter search (50 trials per model×dataset×horizon):

```bash
pip install optuna

# Single setting
python tune_optuna.py --csv-dir GATiDE/data --dataset ETTh1 --horizon 96 \
  --model gatide --n-trials 50 --device cuda

# All settings (7 datasets × 4 horizons × 4 models × 50 trials)
python tune_optuna.py --csv-dir GATiDE/data --dataset all --horizon all \
  --model all --n-trials 50 --device cuda
```

Outputs:
- `tuned_configs/{dataset}_H{horizon}_{model}_best.json` — best params
- `tuned_configs/tuned_best.yaml` — aggregated config

---

## Programmatic use (Python)

```python
from benchmark.benchmark import run_benchmark

df = run_benchmark(
    csv_dir="GATiDE/data",
    datasets=["ETTh1", "weather"],
    horizons=[96, 192],
    models=["gatide", "tide", "dlinear"],
    lookback=720, n_epochs=50, batch_size=32,
    device="cuda",
    save_dir="./my_results"
)
print(df.groupby(["model", "horizon"])[["mse", "mae"]].mean())
```

With tuned params (no YAML):
```python
df = run_benchmark(
    csv_dir="GATiDE/data", datasets=["ETTh1"], horizons=[96],
    models=["gatide"], n_epochs=100, batch_size=512, lr=0.0042,
    device="cuda", save_dir="./tuned_run",
    model_kwargs={"gatide": {"hidden_size":128, "num_encoder_layers":1,
                              "num_decoder_layers":1, "decoder_output_dim":8,
                              "temporal_decoder_hidden":64, "dropout":0.1,
                              "use_layer_norm":True}}
)
```

---

## Output

```
benchmark_final/
  benchmark_results.csv           # one row per (dataset, horizon, model, seed)
  predictions/
    ETTh1_H96_gatide_pred.npy     # (N, H, C) original scale
    ETTh1_H96_gatide_true.npy
```

CSV columns: `dataset, horizon, model, seed, mse, mae, mse_norm, mae_norm, val_mse, val_mae, train_time_per_epoch_s, inference_ms_per_batch, peak_memory_mb, epochs_run, n_params, batch_size, lr, split_convention, status`

---

## Models

| Model | File | Architecture |
|-------|------|-------------|
| GATiDE | `benchmark/models/gatide_adapter.py` | GatedResidualBlock + SegmentAttentionFusion |
| TiDE | `benchmark/models/tide.py` | ResidualBlock encoder/decoder + temporal decoder |
| DLinear | `benchmark/models/dlinear.py` | MovingAvg decomposition + linear |
| PatchTST | `benchmark/models/patchtst.py` | Patching + TransformerEncoder |
| Naive | `benchmark/models/naive.py` | Persistence (last/mean/drift) |

---

## Citation

```bibtex
@inproceedings{das2023tide,
  title={Long-term Forecasting with TiDE: Time-series Dense Encoder},
  author={Das, Abhimanyu and Kong, Weihao and Leber, Andrew and others},
  journal={Transactions on Machine Learning Research},
  year={2023}
}
```

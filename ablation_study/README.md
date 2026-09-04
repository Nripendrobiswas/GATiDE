# GATiDE Ablation Study

Standalone ablation package — **does not modify** `benchmark/` or `src/ga_tide/`. Drop this folder at repo root (`GATiDE/ablation_study/`) and push to GitHub.

```
GATiDE/
  benchmark/              # original benchmark (untouched)
  src/ga_tide/            # original model
  ablation_study/         # ← this folder (self-contained)
    models/gatide_ablation.py
    configs/ablation_config.yaml
    run_ablation.py
    analyze_ablation.py
    results/
```

## What is ablated

| Variant key | Class `models/gatide_ablation.py` | Gating `GatedResidualBlock` `src/ga_tide/model.py:51` | LayerNorm `gatide_adapter.py:111` | Skip `gatide_adapter.py:214` | Params vs GATiDE |
|---|---|---|---|---|---|
| `gatide` | `GATiDEAblation(use_gating=True, use_layer_norm=True, use_skip=True)` | GRB `skip + sigmoid(Wg*x)*h` | LN if `d>1` | `Linear(L->H)` | 5.26M (ETTh1 L720 H96 C7 hidden256) |
| `gatide_nogate` | `GATiDEAblation(use_gating=False, use_layer_norm=True, use_skip=True)` | **RB** `skip + h` `benchmark/models/tide.py:27` | LN | Linear | 3.57M (-1.68M) isolates **gate** |
| `gatide_no_ln` | `GATiDEAblation(use_gating=True, use_layer_norm=False, use_skip=True)` | GRB | **none** | Linear | 5.25M |
| `gatide_no_skip` | `GATiDEAblation(use_gating=True, use_layer_norm=True, use_skip=False)` | GRB | LN | **none** | same |
| `tide` | `benchmark/models/tide.py:49 TiDE` | RB | conditional | Linear | 3.57M external baseline |

- **True GRB ablation = `gatide` vs `gatide_nogate`** with `use_layer_norm=True` fixed — same arch, same `hidden_size`, same `num_layers`, only `Wg` removed.
- **LN ablation = `gatide` vs `gatide_no_ln`** (already validated: LN 0.599±0.024 vs noLN 0.824±0.036 `mse_norm` ETTh1 H96 5 seeds).
- **SAF** `models/gatide_ablation.py:SegmentAttentionFusion` is bypassed when `|D|=1` (covariate-free `benchmark/models/gatide_adapter.py:187`). To ablate SAF use `--use-covariates` (requires time covariates `benchmark/datasets.py:131`).

## Install
```bash
git clone https://github.com/Nripendrobiswas/GATiDE.git
cd GATiDE
pip install -r requirements.txt          # or ablation_study/requirements.txt
pip install -r ablation_study/requirements.txt  # optional, same
```

## Quick run (Kaggle / Colab)

```python
!git clone https://github.com/Nripendrobiswas/GATiDE.git
%cd GATiDE
!pip install -r requirements.txt

# 1. True GRB ablation, ETTh1 H96, 5 seeds (TiDE paper §5.1 mean±std)
!python ablation_study/run_ablation.py --csv-dir GATiDE/data --datasets ETTh1 --horizons 96 --ablations gatide gatide_nogate tide --epochs 100 --seeds 0 1 2 3 4 --save-dir ./ablation_study/results/grb_96

# 2. Full matrix (all ablations, all horizons)
!python ablation_study/run_ablation.py --csv-dir GATiDE/data --all-datasets --all-horizons --ablations all --epochs 100 --seeds 0 1 2 3 4 --save-dir ./ablation_study/results/full
```

## Local (Windows, same as `benchmark/run_benchmark.py:1`)
```bash
python ablation_study/run_ablation.py --csv-dir "E:/Machine Learning Research/GATiDE Final Verse/GATiDE/data" --datasets ETTh1 --horizons 96 --ablations gatide gatide_nogate --epochs 100 --seeds 0 1 2 3 4 --save-dir ./ablation_study/results/grb_local
```

## CLI flags

| Flag | Purpose | Default |
|---|---|---|
| `--csv-dir` | folder with `ETTh1.csv` etc. | `GATiDE/data` |
| `--datasets` | list or `all` | `ETTh1` |
| `--horizons` | `96 192 336 720` or `all` | `96` |
| `--ablations` | `gatide gatide_nogate gatide_no_ln gatide_no_skip tide dlinear naive` or `all` | `gatide gatide_nogate tide` |
| `--epochs` | training epochs | `100` |
| `--batch-size` | batch | `32` |
| `--lr` | learning rate | `1e-3` |
| `--seeds` | `0 1 2 3 4` for mean±std | `42` |
| `--use-covariates` | enable SAF (needs covariates) | `False` |
| `--device` | `auto`/`cpu`/`cuda` | `auto` |
| `--save-dir` | output | `./ablation_study/results` |

Pass tuned hyperparams without YAML `benchmark/run_benchmark.py:91`:
```bash
python ablation_study/run_ablation.py --csv-dir GATiDE/data --datasets ETTh1 --horizons 96 --ablations gatide --model-kwargs '{"hidden_size":128,"num_encoder_layers":1,"dropout":0.1,"use_layer_norm":true}' --tune-lr 0.004 --tune-batch-size 512 --epochs 100 --seeds 0 1 2 3 4 --save-dir ./ablation_study/results/tuned
```

## Outputs
```
results/grb_96/
  ablation_results.csv   # one row per (dataset,horizon,ablation,seed) : mse, mae, mse_norm, mae_norm, val_*, train_time_per_epoch_s, inference_ms_per_batch, peak_memory_mb, n_params, status
  predictions/
    ETTh1_H96_gatide_seed0_pred.npy  # (N,H,C) original scale
    ETTh1_H96_gatide_seed0_true.npy
  summary.txt            # mean±std table (TiDE Table 2 style)
```

Analyze after run:
```bash
python ablation_study/analyze_ablation.py --csv ./ablation_study/results/grb_96/ablation_results.csv --out ./ablation_study/results/grb_96/summary.txt
# also prints LaTeX table for paper
```

## Reproducing the 5-seed LN result in this folder

The LN ablation already run via `benchmark/run_benchmark.py` (not this folder) gave:
- `use_layer_norm True`: `mse_norm 0.5995±0.0244` n=5 ETTh1 H96
- `use_layer_norm False`: `0.8246±0.0367` → **-27.3%** with LN, no overlap (`max LN 0.6255 < min noLN 0.7736`).

To reproduce via ablation_study:
```bash
python ablation_study/run_ablation.py --csv-dir GATiDE/data --datasets ETTh1 --horizons 96 --ablations gatide gatide_no_ln --epochs 100 --seeds 0 1 2 3 4 --save-dir ./ablation_study/results/ln_repro
```

## Push to GitHub
```bash
git add ablation_study/
git commit -m "Add standalone ablation_study (GRB/LN/Skip) without touching benchmark/"
git push origin main
```

## Citation
Same as repo `README.md: Citation` — Das et al. TiDE 2023, Zeng et al. DLinear 2023, Nie et al. PatchTST 2023.

## Notes
- No file in `benchmark/` or `src/` is modified. `ablation_study/models/gatide_ablation.py` duplicates `GatedResidualBlock`/`ResidualBlock` logic for isolation — see header comments for line anchors.
- `SAF` requires `segment_dims >=2`. Current covariate-free benchmark `L* C` single segment bypasses SAF by design `gatide_adapter.py:187`. Enable `--use-covariates` to test SAF.
- `hidden_size % num_attn_heads ==0` required `gatide_adapter.py:180`.

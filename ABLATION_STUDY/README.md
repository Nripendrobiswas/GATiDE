# GA-TiDE Ablation Study

2x2 factorial ablation isolating the two architectural deltas of **GA-TiDE** vs the
original **TiDE** (Das et al., 2023), under identical data protocol, training loop,
and hyperparameters.

## Variant matrix

| Registry key | Gate | Segment attention | Covariate mixing | Class |
|---|---|---|---|---|
| `tide` | no | no | vanilla concat (cov-aware) | `benchmark.models.tide.TiDE` |
| `gatide-gate` | **yes** | no | vanilla concat | `GATiDEPure(fusion="concat")` |
| `gatide-attn` | no | **yes** | attention tokens | `GATiDEPure(use_gate=False)` |
| `gatide` | **yes** | **yes** | attention tokens | `GATiDEPure` |

- **gate** = `GatedResidualBlock` (`skip(x) + sigmoid(gate(x)) * h`, dropout between
  fc1/fc2, LayerNorm(1) guard) vs vanilla `ResidualBlock` (`skip(x) + h`)
- **segment attention** = `SegmentAttentionFusion` (project segments to hidden tokens,
  self-attention across them) vs vanilla TiDE's concat pre-mixing
- All four variants receive the **same time covariates** (TiDE §5.1 time-derived
  features, train-only standardized) via `--use-covariates`, so the comparison is
  controlled: identical information, only the architecture differs.

## Shared-tuned policy

`run_ablation.py` reads `tuned_configs/tuned_best.yaml` and copies the **tuned
`gatide` parameters** (including lr/batch_size) to every variant at each
(dataset, horizon) setting. If no tuned entry exists for a setting, it falls back to
`configs/default.yaml` gatide params. Hyperparameters are therefore held constant
across the 2x2; observed deltas are attributable to architecture alone.

## Usage

```powershell
# 1) tune gatide once per setting (shared hyperparameter source)
python tune_optuna.py --dataset ETTh1 --horizon 96 --model gatide --batch-sizes 32 64 128 256 512 --n-trials 50 --n-epochs 100 --device auto
python tune_optuna.py --dataset ETTh1 --horizon 336 --model gatide --batch-sizes 32 64 128 256 512 --n-trials 50 --n-epochs 100 --device auto
python tune_optuna.py --dataset weather --horizon 96 --model gatide --batch-sizes 32 64 128 256 512 --n-trials 50 --n-epochs 100 --device auto
python tune_optuna.py --dataset weather --horizon 336 --model gatide --batch-sizes 32 64 128 256 512 --n-trials 50 --n-epochs 100 --device auto

# 2) run the ablation grid (focused scope, ~4-6 h on CPU)
python ablation_study/run_ablation.py --datasets ETTh1 weather --horizons 96 336 --seeds 0 1 2 --n-epochs 50 --device auto
```

Quick smoke (minutes):

```powershell
python ablation_study/run_ablation.py --datasets ETTh1 --horizons 96 --seeds 0 --n-epochs 2 --device cpu --no-save-predictions
```

## Outputs (`ablation_study/outputs/`)

- `benchmark_results.csv` — raw rows, one per (dataset, horizon, variant, seed)
- `ablation_summary.csv` — per setting: variant mean±std MSE/MAE (original +
  normalized scale), `delta_mse_vs_tide`, `delta_mae_vs_tide`, `pct_mse_change`
- `predictions/*.npy` — per-run inverse-scaled pred/true arrays
- Console — psql-style ablation tables

## Notes / caveats for the paper

- The `tide` baseline here is **covariate-aware** (vanilla concat path, mirroring
  Darts `TiDEModel` with future covariates). This differs from the main benchmark
  protocol, where baselines remain covariate-free under `--use-covariates` — that
  asymmetry is intentional here so the 2x2 is information-controlled.
- Gate main effect = `gatide` − `gatide-attn` (also `gatide-gate` − `tide`);
  attention main effect = `gatide` − `gatide-gate` (also `gatide-attn` − `tide`);
  interaction = whether the two effects are additive.
- Dropout placement and the LayerNorm(1) fix are **not** ablated in this study
  (they travel with the gate); state this when reporting.

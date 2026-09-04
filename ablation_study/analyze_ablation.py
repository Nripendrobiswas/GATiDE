#!/usr/bin/env python
"""
Analyze ablation_results.csv -> summary, LaTeX table, t-test
Usage:
  python ablation_study/analyze_ablation.py --csv ./ablation_study/results/grb_96/ablation_results.csv --out ./ablation_study/results/grb_96/summary.txt
  python ablation_study/analyze_ablation.py --csv ./ablation_study/results/grb_96/ablation_results.csv --latex
"""
from __future__ import annotations
import argparse
import os
import pandas as pd
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="path to ablation_results.csv")
    p.add_argument("--out", type=str, default=None, help="output summary txt (default <csv_dir>/summary.txt)")
    p.add_argument("--latex", action="store_true", help="print LaTeX table to stdout")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    print(f"[info] Loaded {args.csv} shape {df.shape}")
    print(df.head().to_string())

    # Mean per ablation
    mean_orig = df.groupby("ablation")[["mse","mae"]].mean()
    mean_norm = df.groupby("ablation")[["mse_norm","mae_norm"]].mean()
    print("\n=== Mean per ablation (original) ===")
    print(mean_orig.round(4).to_string())
    print("\n=== Mean per ablation (normalized, TiDE Table 2) ===")
    print(mean_norm.round(4).to_string())

    # mean±std per dataset,horizon,ablation
    if "seed" in df.columns and df["seed"].nunique() > 1:
        g = df.groupby(["dataset","horizon","ablation"])[["mse_norm","mae_norm"]].agg(["mean","std"])
        print("\n=== Per (dataset,horizon,ablation) mean±std normalized ===")
        print(g.round(4).to_string())
        # flatten for csv
        flat = g.copy()
        flat.columns = ["_".join(col) for col in flat.columns]
        flat = flat.reset_index()
        out_csv = os.path.join(os.path.dirname(args.csv), "ablation_summary_meanstd.csv")
        flat.to_csv(out_csv, index=False)
        print(f"\n[saved] {out_csv}")

        # Pairwise t-test gatide vs gatide_nogate if both present
        ablations = df["ablation"].unique().tolist()
        if "gatide" in ablations and "gatide_nogate" in ablations:
            try:
                from scipy.stats import ttest_ind
                has_scipy = True
            except ImportError:
                has_scipy = False
                print("\n[warn] scipy not installed, skipping t-test (pip install scipy)")
            if has_scipy:
                # take first dataset/horizon for test
                for (ds, hor), sub in df.groupby(["dataset","horizon"]):
                    a = sub[sub["ablation"]=="gatide"]["mse_norm"].dropna().values
                    b = sub[sub["ablation"]=="gatide_nogate"]["mse_norm"].dropna().values
                    if len(a)>1 and len(b)>1:
                        t, p = ttest_ind(a, b, equal_var=False)
                        # also simple difference
                        print(f"\n[t-test] {ds} H={hor} gatide vs gatide_nogate mse_norm: mean {a.mean():.4f} vs {b.mean():.4f} diff {a.mean()-b.mean():.4f} t={t:.2f} p={p:.4g} n={len(a)}/{len(b)}")
                        # check overlap
                        print(f"  gatide range [{a.min():.4f}, {a.max():.4f}] std {a.std():.4f}")
                        print(f"  nogate range [{b.min():.4f}, {b.max():.4f}] std {b.std():.4f}")
                        if a.max() < b.min() or b.max() < a.min():
                            print("  -> no overlap (highly significant)")
                        break

    # Throughput
    if "train_time_per_epoch_s" in df.columns:
        th = df.groupby("ablation")[["train_time_per_epoch_s","inference_ms_per_batch","peak_memory_mb","n_params"]].mean()
        print("\n=== Throughput mean ===")
        print(th.round(4).to_string())

    # Save summary txt
    out_path = args.out or os.path.join(os.path.dirname(args.csv), "summary.txt")
    with open(out_path, "w") as f:
        f.write("=== Mean per ablation (original) ===\n")
        f.write(mean_orig.round(4).to_string())
        f.write("\n\n=== Mean per ablation (normalized) ===\n")
        f.write(mean_norm.round(4).to_string())
        if "seed" in df.columns and df["seed"].nunique()>1:
            f.write("\n\n=== Per (dataset,horizon,ablation) mean±std ===\n")
            f.write(g.round(4).to_string())
        if "train_time_per_epoch_s" in df.columns:
            f.write("\n\n=== Throughput ===\n")
            f.write(th.round(4).to_string())
    print(f"\n[saved] {out_path}")

    if args.latex:
        # LaTeX table for paper: rows ablation, cols mse_norm mean±std etc.
        if "seed" in df.columns:
            latex_df = df.groupby("ablation")[["mse_norm","mae_norm"]].agg(["mean","std"]).round(4)
            # flatten
            latex_df.columns = [" ".join(col) for col in latex_df.columns]
            print("\n=== LaTeX ===")
            print(latex_df.to_latex())
            # simple booktabs rows
            print("\n% Booktabs rows for manuscript:")
            for ab in latex_df.index:
                m = latex_df.loc[ab, "mse_norm mean"]
                s = latex_df.loc[ab, "mse_norm std"]
                print(f"{ab} & {m:.4f} $\\pm$ {s:.4f} \\\\")

if __name__ == "__main__":
    main()

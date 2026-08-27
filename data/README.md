# Datasets

Benchmark data is **not committed**. Obtain it in one of two ways.

## Option 1 — Darts loaders (default)

`scripts/benchmark.py --source darts` downloads through Darts' bundled
dataset loaders on first use. Convenient, but note:

> Darts' `ElectricityDataset` is the 370-client LD2011_2014 series. The LTSF
> benchmark "Electricity" (ECL) is a 321-client preprocessed variant. They are
> **not the same data**, and numbers produced this way are not comparable to
> published ECL results.

## Option 2 — LTSF benchmark CSVs (required for comparability)

For any number reported against prior work, use the CSVs distributed with the
Autoformer repository (`https://github.com/thuml/Autoformer`), which are the
files Informer, Autoformer, DLinear, PatchTST and TiDE all evaluate on.

Place them here:

```
data/
├── ETTh1.csv
├── ETTh2.csv
├── ETTm1.csv
├── ETTm2.csv
├── weather.csv
└── electricity.csv
```

Then run with:

```bash
python scripts/benchmark.py --source csv --csv-dir data --dataset ETTh1
```

State in the paper which option produced each reported number.

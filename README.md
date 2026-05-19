# scPerturbEval

`scPerturbEval` is a lightweight evaluation framework for single-cell perturbation prediction.
It supports paired `real/pred` `.h5ad` evaluation, fold-level aggregation, and baseline workflows.

## What this repo includes

- Metric evaluation in `raw`, `pca`, and `deg` spaces.
- Baseline runners for Norman-style workflows.
- Fold aggregation utilities (mean/std across folds).
- Pathway recovery metrics (GSEA-based), when gene symbols are available.

## Repository layout

- `src/scPerturbEval/metrics.py`: core metric implementation.
- `src/scPerturbEval/evaluations.py`: curated evaluation CLI module.
- `src/scPerturbEval/run_norman_baselines.py`: baseline generation.
- `src/scPerturbEval/preprocess_norman_scdfm.py`: Norman preprocessing/splitting.
- `src/scPerturbEval/aggregate_fold_metrics.py`: fold aggregation.
- `scripts/`: CLI wrappers.

## Installation

```bash
pip install -r requirements.txt
```

## How to run

For complete step-by-step instructions, examples, and troubleshooting, see:

- [`USER_MANUAL.md`](./USER_MANUAL.md)


# scPerturbEval

Utilities for evaluating single-cell perturbation predictions from paired `.h5ad` files.

## Install

```bash
pip install -r requirements.txt
# or
pip install -e .
```

## CLI usage

```bash
python scripts/run_metrics.py \
  --real /path/to/real.h5ad \
  --pred /path/to/pred.h5ad \
  --condition-column condition \
  --metrics euclidean root_mean_squared_error mean_absolute_error pearson_distance spearman_distance cosine_distance r2_distance \
  --out results/fold0_metrics.csv
```

## Colab usage

```bash
git clone <your-repo-url>
cd scPerturbEval
pip install -r requirements.txt
```

Then run the notebook in `notebooks/`.

## Extra metrics in `compute_metrics_space`

Supported space-aware extras:
- `pcc_delta`
- `top_deg_recall`
- `top_deg_precision`
- `deg_direction_agreement`
- `deg_spearman_lfc`
- `pds_cosine` (cosine-only perturbation discrimination score)

For `top_deg_recall`, `top_deg_precision`, `deg_direction_agreement`, and `deg_spearman_lfc`,
the implementation uses DE-style comparisons (condition vs control in real and pred) with:
- per-gene log2 fold-change
- Welch t-test p-values
- Benjamini-Hochberg FDR correction

Useful flags:
- `--deg-fdr-threshold` (default `0.05`)
- `--lfc-eps` (default `1e-8`)

## Aggregate fold metrics (mean/std)

```bash
python -m scPerturbEval.aggregate_fold_metrics \
  --pattern "ckpts/additive/fold*/fold_metrics_space.csv" \
  --out-per-fold results/fold_metrics_per_fold.csv \
  --out-summary results/fold_metrics_summary.csv
```

You can also pass files directly:

```bash
python -m scPerturbEval.aggregate_fold_metrics \
  --csv-paths ckpts/additive/fold0/fold_metrics_space.csv ckpts/additive/fold1/fold_metrics_space.csv
```

## Norman Baselines

`run_norman_baselines` supports:
- `control_baseline`
- `global_delta`
- `one_layer_mlp`

Example (MLP baseline, 5000 steps):

```bash
python -m scPerturbEval.run_norman_baselines \
  --processed-dir /path/to/processed_fold0 \
  --baseline one_layer_mlp \
  --steps 5000 \
  --hidden-dim 1024 \
  --lr 1e-3 \
  --seed 42 \
  --out-dir /path/to/out/fold0/one_layer_mlp
```

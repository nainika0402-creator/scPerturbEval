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

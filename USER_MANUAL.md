# scPerturbEval User Manual

This guide is a practical, end-to-end walkthrough for running preprocessing, baselines, evaluation metrics, and fold aggregation.

## 1) What This Repo Expects

You generally work with paired `.h5ad` files:
- `real.h5ad`: ground truth expression
- `pred.h5ad`: model predictions

Minimum structure:
- `X`: expression matrix (cells x genes/features)
- `obs`: has a condition label column (for example `condition` or `perturbation`)
- `var_names`: gene/features axis

Important:
- Control-referenced metrics need a control condition label present in both files.
- Pathway metrics need real mappable gene symbols in `var_names` (numeric names like `0,1,2...` will not work for GSEA mapping).

## 2) Install

From `scPerturbEval/`:

```bash
pip install -r requirements.txt
```

## 3) Preprocess Norman Data (Optional, Baseline Workflow)

Use this when you want the scDFM-style train/test split artifacts:

```bash
python scripts/preprocess_norman_scdfm.py \
  --norman-h5ad /path/to/norman.h5ad \
  --out-dir /path/to/fold0_preprocessed \
  --split-method additive \
  --fold 0
```

Outputs:
- `norman_processed_all.h5ad`
- `norman_processed_train.h5ad`
- `norman_processed_test.h5ad`
- `norman_preprocess_meta.json`
- split file (`split_results.pkl` or `split_results_unseen.pkl`)

## 4) Run Baselines

Entry point:
- `scripts/run_baselines.py`

Supported baselines:
- `control_baseline`
- `global_delta`
- `linear_baseline`
- `one_layer_mlp`

Example (MLP):

```bash
python scripts/run_baselines.py \
  --processed-dir /path/to/fold0_preprocessed \
  --baseline one_layer_mlp \
  --steps 5000 \
  --hidden-dim 1024 \
  --lr 1e-3 \
  --seed 42 \
  --early-stopping \
  --val-fraction 0.2 \
  --patience 20 \
  --min-delta 1e-4 \
  --out-dir /path/to/fold0_baseline
```

Baseline outputs:
- `<baseline>_pred.h5ad`
- `<baseline>_real.h5ad`
- `<baseline>_metrics.csv`

## 5) Run Evaluations

Use the curated metric entry point:
- `python -m scPerturbEval.evaluations`

Spaces:
- `raw`
- `pca`
- `deg`

Example (raw metrics):

```bash
python -m scPerturbEval.evaluations \
  --real /path/to/fold0_real.h5ad \
  --pred /path/to/fold0_pred.h5ad \
  --condition-column condition \
  --control-label control \
  --space raw \
  --metrics root_mean_squared_error pearson_distance wmse pcc_delta pearson_delta_pert weighted_r2_delta pds_cosine matrix_distance top_deg_recall deg_direction_agreement deg_spearman_lfc \
  --out /path/to/fold0_metrics_raw.csv
```

Example (PCA distribution pass):

```bash
python -m scPerturbEval.evaluations \
  --real /path/to/fold0_real.h5ad \
  --pred /path/to/fold0_pred.h5ad \
  --condition-column condition \
  --space pca \
  --n-components 20 \
  --metrics wasserstein mmd \
  --out /path/to/fold0_metrics_pca.csv
```

Pathway metrics (raw only, gene symbols required):

```bash
python -m scPerturbEval.evaluations \
  --real /path/to/fold0_real.h5ad \
  --pred /path/to/fold0_pred.h5ad \
  --condition-column condition \
  --space raw \
  --metrics pathway_nes_spearman pathway_topk_jaccard \
  --pathway-gene-sets MSigDB_Hallmark_2020 \
  --pathway-top-k 10 \
  --pathway-reference perturbed_centroid \
  --out /path/to/fold0_pathway.csv
```

## 6) Aggregate Across Folds

Entry point:
- `scripts/aggregate_fold_metrics.py`

Example:

```bash
python scripts/aggregate_fold_metrics.py \
  --pattern "/path/to/scdfm_fold*_metrics_raw.csv" \
  --out-per-fold /path/to/per_fold_metrics.csv \
  --out-summary /path/to/summary_metrics.csv
```

Summary output columns:
- `metric`
- `mean` (across folds)
- `std` (across folds)
- `n_folds`

## 7) Metrics in `evaluations.py`

Allowed metrics:
- `root_mean_squared_error`
- `pearson_distance`
- `wmse`
- `wasserstein`
- `mmd`
- `pcc_delta`
- `pearson_delta_pert`
- `weighted_r2_delta`
- `pds_cosine`
- `matrix_distance`
- `top_deg_recall`
- `deg_direction_agreement`
- `deg_spearman_lfc`
- `pathway_nes_spearman`
- `pathway_topk_jaccard`

## 8) Common Failure Modes

1. Pathway metrics return `NaN`
- `gseapy` missing, or
- `var_names` are not real gene symbols.

2. Control-referenced metrics fail
- control label not present in both input files.

3. `mmd` is slow
- run it in a separate pass, often in `pca` space with fewer components.

4. Fold mismatch
- make sure each fold uses its corresponding real/pred pair before aggregation.

## 9) Recommended Fold Workflow

For each fold `k`:
1. Run baseline to get `*_real.h5ad`, `*_pred.h5ad`.
2. Run `evaluations` raw pass.
3. Run optional PCA distribution pass.
4. Save fold CSVs with clear names (`foldk_*`).

Then run one aggregation command over all fold CSVs.

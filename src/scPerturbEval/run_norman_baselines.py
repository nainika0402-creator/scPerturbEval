from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .preprocess_norman_scdfm import load_preprocessed_norman


def _to_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _safe_mean(X: np.ndarray) -> np.ndarray:
    if X.shape[0] == 0:
        raise ValueError("Cannot compute mean of empty matrix.")
    return X.mean(axis=0)


def _condition_means(X: np.ndarray, obs: pd.DataFrame) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    cond_arr = obs["condition"].astype(str).to_numpy()
    for cond in np.unique(cond_arr):
        mask = cond_arr == cond
        out[cond] = _safe_mean(X[mask])
    return out


def _build_delta_by_train_condition(train_condition_means: dict[str, np.ndarray], control_mean: np.ndarray) -> dict[str, np.ndarray]:
    deltas: dict[str, np.ndarray] = {}
    for cond, mu in train_condition_means.items():
        if cond.lower() == "control" or cond.lower() == "control+control":
            continue
        deltas[cond] = mu - control_mean
    return deltas


def run_baseline(
    processed_dir: Path,
    baseline: str,
    out_dir: Path,
) -> dict[str, Path]:
    adata_all, adata_train, adata_test, _meta = load_preprocessed_norman(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_x = _to_dense(adata_train.X)
    test_x = _to_dense(adata_test.X)
    test_obs = adata_test.obs.copy()

    if "condition" not in test_obs:
        raise ValueError("Expected 'condition' in preprocessed test .obs")
    if "mode" not in test_obs:
        raise ValueError("Expected 'mode' in preprocessed test .obs")

    train_obs = adata_train.obs.copy()
    train_cond_arr = train_obs["condition"].astype(str).to_numpy()
    train_control_mask = train_obs["is_control"].to_numpy().astype(bool)
    if not np.any(train_control_mask):
        raise ValueError("No control cells found in preprocessed train data.")
    control_mean_train = _safe_mean(train_x[train_control_mask])

    train_condition_means = _condition_means(train_x, train_obs)
    delta_by_train_condition = _build_delta_by_train_condition(train_condition_means, control_mean_train)
    if len(delta_by_train_condition) == 0:
        raise ValueError("No non-control training perturbations found to build delta baselines.")

    delta_global = np.mean(np.stack(list(delta_by_train_condition.values()), axis=0), axis=0)

    eval_mask = test_obs["mode"].astype(str).to_numpy() == "test"
    eval_obs = test_obs.loc[eval_mask].copy()
    eval_x = test_x[eval_mask]

    pred_x = np.zeros_like(eval_x, dtype=np.float32)
    # control baseline: use test control-state mean (mu_control,test)
    test_control_mask = test_obs["is_control"].to_numpy().astype(bool)
    if not np.any(test_control_mask):
        raise ValueError("No control cells found in preprocessed test data.")
    control_mean_test = _safe_mean(test_x[test_control_mask])

    for cond in eval_obs["condition"].astype(str).unique():
        idx = np.where(eval_obs["condition"].astype(str).to_numpy() == cond)[0]
        if baseline == "control_baseline":
            pred_vec = control_mean_test
        elif baseline == "global_delta":
            pred_vec = control_mean_test + delta_global
        else:
            raise ValueError(f"Unsupported baseline: {baseline}")

        pred_x[idx] = np.repeat(pred_vec[None, :], len(idx), axis=0)

    rows = []
    for cond in eval_obs["condition"].astype(str).unique():
        idx = np.where(eval_obs["condition"].astype(str).to_numpy() == cond)[0]
        real_c = eval_x[idx]
        pred_c = pred_x[idx]
        real_m = _safe_mean(real_c)
        pred_m = _safe_mean(pred_c)
        corr = pearsonr(real_m, pred_m)[0]
        rows.append(
            {
                "condition": cond,
                "n_cells": int(len(idx)),
                "rmse": float(np.sqrt(mean_squared_error(real_m, pred_m))),
                "mae": float(mean_absolute_error(real_m, pred_m)),
                "pearson": float(corr) if not np.isnan(corr) else np.nan,
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)

    pred_adata = ad.AnnData(X=pred_x, obs=eval_obs.copy(), var=adata_test.var.copy())
    real_adata = ad.AnnData(X=eval_x, obs=eval_obs.copy(), var=adata_test.var.copy())

    metrics_path = out_dir / f"{baseline}_metrics.csv"
    pred_path = out_dir / f"{baseline}_pred.h5ad"
    real_path = out_dir / f"{baseline}_real.h5ad"
    metrics_df.to_csv(metrics_path, index=False)
    pred_adata.write_h5ad(pred_path)
    real_adata.write_h5ad(real_path)

    return {"metrics": metrics_path, "pred": pred_path, "real": real_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load preprocessed Norman data and run baseline models.")
    parser.add_argument("--processed-dir", type=Path, required=True, help="Directory with preprocessed Norman files.")
    parser.add_argument(
        "--baseline",
        choices=["control_baseline", "global_delta"],
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for predictions and metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_baseline(
        processed_dir=args.processed_dir,
        baseline=args.baseline,
        out_dir=args.out_dir,
    )
    print("Wrote:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

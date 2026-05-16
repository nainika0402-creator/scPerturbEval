from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cosine as cosine_distance_fn
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .loader import (
    DatasetSpec,
    LoaderPolicy,
    PairedAnnDataLoader,
    extract_condition_matrices,
)


VECTOR_METRICS = {
    "euclidean",
    "root_mean_squared_error",
    "mean_absolute_error",
    "pearson_distance",
    "spearman_distance",
    "cosine_distance",
    "r2_distance",
    "mmd",
}

DISTRIBUTION_METRICS = {"edistance", "wasserstein", "sym_kldiv"}


def _to_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _safe_mean(X: np.ndarray) -> np.ndarray:
    if X.shape[0] == 0:
        raise ValueError("Cannot compute mean on empty matrix")
    return X.mean(axis=0)


def _gaussian_mmd2(X: np.ndarray, Y: np.ndarray, sigma: float | None = None) -> float:
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return np.nan

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if sigma is None:
        joint = np.vstack([X, Y])
        if joint.shape[0] > 2000:
            idx = np.random.RandomState(42).choice(joint.shape[0], 2000, replace=False)
            joint = joint[idx]
        dists = np.sum((joint[:, None, :] - joint[None, :, :]) ** 2, axis=2)
        med = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
        sigma = np.sqrt(max(med, 1e-12))

    gamma = 1.0 / (2.0 * sigma * sigma)

    XX = np.exp(-gamma * np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    YY = np.exp(-gamma * np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2))
    XY = np.exp(-gamma * np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2))

    return float(XX.mean() + YY.mean() - 2.0 * XY.mean())


def _vector_metric(metric: str, real_x: np.ndarray, pred_x: np.ndarray) -> float:
    r = _safe_mean(real_x)
    p = _safe_mean(pred_x)

    if metric == "euclidean":
        return float(np.linalg.norm(p - r))
    if metric == "root_mean_squared_error":
        return float(np.sqrt(mean_squared_error(r, p)))
    if metric == "mean_absolute_error":
        return float(mean_absolute_error(r, p))
    if metric == "pearson_distance":
        corr = pearsonr(r, p)[0]
        if np.isnan(corr):
            return np.nan
        return float(1.0 - corr)
    if metric == "spearman_distance":
        corr = spearmanr(r, p)[0]
        if np.isnan(corr):
            return np.nan
        return float(1.0 - corr)
    if metric == "cosine_distance":
        return float(cosine_distance_fn(r, p))
    if metric == "r2_distance":
        return float(1.0 - r2_score(r, p))
    if metric == "mmd":
        return _gaussian_mmd2(pred_x, real_x)

    raise ValueError(f"Unsupported vector metric: {metric}")


def _distribution_metric(metric: str, real_x: np.ndarray, pred_x: np.ndarray) -> float:
    if metric == "wasserstein":
        # 1D approximation by averaging feature-wise Wasserstein distances.
        vals = [wasserstein_distance(real_x[:, i], pred_x[:, i]) for i in range(real_x.shape[1])]
        return float(np.mean(vals))

    # Use pertpy for remaining distribution metrics.
    try:
        import anndata as ad
        import pertpy as pt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pertpy is required for distribution metrics (edistance/sym_kldiv). "
            "Install requirements from hell/requirements.txt"
        ) from exc

    obs_real = pd.DataFrame({"Expcategory": ["stimulated"] * real_x.shape[0]})
    obs_pred = pd.DataFrame({"Expcategory": ["imputed"] * pred_x.shape[0]})

    merged = ad.concat(
        [ad.AnnData(X=real_x, obs=obs_real), ad.AnnData(X=pred_x, obs=obs_pred)],
        join="inner",
    )
    merged.layers["X"] = merged.X

    distance = pt.tools.Distance(metric=metric, layer_key="X")
    pairwise_df = distance.onesided_distances(
        merged,
        groupby="Expcategory",
        selected_group="imputed",
        groups=["stimulated"],
    )
    val = float(pairwise_df["stimulated"])
    if metric == "sym_kldiv":
        val = float(np.log2(val + 1.0))
    return val


def compute_metrics(
    real_path: Path,
    pred_path: Path,
    condition_column: str,
    metrics: List[str],
    min_cells_per_condition: int,
) -> pd.DataFrame:
    spec = DatasetSpec(
        condition_columns=[condition_column],
        split_column=None,
        min_cells_per_condition=min_cells_per_condition,
    )
    policy = LoaderPolicy(
        gene_alignment="intersection",
        split_policy="none",
        dense_mode="on_extract",
        copy_mode="deep",
    )

    pair = PairedAnnDataLoader(spec=spec, policy=policy).prepare(real_path, pred_path)

    split = "all"
    conditions = pair.condition_index.split_to_conditions[split]

    rows: List[Dict] = []
    for condition in conditions:
        real_x, pred_x = extract_condition_matrices(pair, split=split, condition=condition, dense_mode="on_extract")
        real_x = _to_dense(real_x)
        pred_x = _to_dense(pred_x)

        if real_x.shape[0] == 0 or pred_x.shape[0] == 0:
            continue

        for metric in metrics:
            if metric in VECTOR_METRICS:
                score = _vector_metric(metric, real_x, pred_x)
            elif metric in DISTRIBUTION_METRICS:
                score = _distribution_metric(metric, real_x, pred_x)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            rows.append(
                {
                    "condition": condition[0],
                    "metric": metric,
                    "score": score,
                    "n_real": int(real_x.shape[0]),
                    "n_pred": int(pred_x.shape[0]),
                    "n_genes": int(real_x.shape[1]),
                }
            )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute perturbation generalization metrics on paired h5ad files.")
    parser.add_argument("--real", required=True, help="Path to real/reference .h5ad")
    parser.add_argument("--pred", required=True, help="Path to predicted .h5ad")
    parser.add_argument(
        "--condition-column",
        default="perturbation",
        help="obs column used to define perturbation condition (default: perturbation)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "euclidean",
            "root_mean_squared_error",
            "mean_absolute_error",
            "pearson_distance",
            "spearman_distance",
            "cosine_distance",
            "r2_distance",
            "mmd",
        ],
        help="Metrics to compute",
    )
    parser.add_argument("--min-cells", type=int, default=1, help="Minimum cells per condition per dataset")
    parser.add_argument("--out", default="hell/metrics_output.csv", help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = compute_metrics(
        real_path=Path(args.real),
        pred_path=Path(args.pred),
        condition_column=args.condition_column,
        metrics=args.metrics,
        min_cells_per_condition=args.min_cells,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote {len(df)} rows to {out}")
    if not df.empty:
        summary = df.groupby("metric", as_index=False)["score"].mean().rename(columns={"score": "mean_score"})
        print("\nMean score per metric:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_distance_fn
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
        vals = [wasserstein_distance(real_x[:, i], pred_x[:, i]) for i in range(real_x.shape[1])]
        return float(np.mean(vals))

    try:
        import anndata as ad
        import pertpy as pt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pertpy is required for distribution metrics (edistance/sym_kldiv)."
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

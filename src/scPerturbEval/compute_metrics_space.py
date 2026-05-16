from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cosine as cosine_distance_fn
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from .compute_metrics import DISTRIBUTION_METRICS, VECTOR_METRICS, _distribution_metric, _vector_metric
from .loader import DatasetSpec, LoaderPolicy, PairedAnnDataLoader, extract_condition_matrices

SPACE_EXTRA_METRICS = {
    "pcc_delta",
    "top_deg_recall",
    "top_deg_precision",
    "deg_direction_agreement",
    "deg_spearman_lfc",
    "pds_cosine",
}


def _to_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _safe_mean(X: np.ndarray) -> np.ndarray:
    if X.shape[0] == 0:
        raise ValueError("Cannot compute mean on empty matrix")
    return X.mean(axis=0)


def _apply_space_transform(
    real_x: np.ndarray,
    pred_x: np.ndarray,
    space: str,
    n_components: int,
    deg_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    if space == "raw":
        return real_x, pred_x, int(real_x.shape[1])

    if space == "pca":
        # Fit PCA on pooled condition data so both real/pred are projected consistently.
        pooled = np.vstack([real_x, pred_x])
        n_comp = max(1, min(n_components, pooled.shape[1], pooled.shape[0]))
        pca = PCA(n_components=n_comp, random_state=0)
        pca.fit(pooled)
        return pca.transform(real_x), pca.transform(pred_x), int(n_comp)

    if space == "deg":
        if deg_mask is None:
            return real_x, pred_x, int(real_x.shape[1])
        if int(deg_mask.sum()) == 0:
            return real_x, pred_x, int(real_x.shape[1])
        return real_x[:, deg_mask], pred_x[:, deg_mask], int(deg_mask.sum())

    raise ValueError(f"Unsupported space: {space}")


def _build_deg_mask(
    real_x: np.ndarray,
    ctrl_real_x: Optional[np.ndarray],
    *,
    deg_lfc: float,
    deg_top_n: int,
) -> Optional[np.ndarray]:
    if ctrl_real_x is None or ctrl_real_x.shape[0] == 0:
        return None
    cond_mean = real_x.mean(axis=0)
    ctrl_mean = ctrl_real_x.mean(axis=0)
    logfc = np.log2((cond_mean + 1e-8) / (ctrl_mean + 1e-8))
    keep = np.abs(logfc) >= deg_lfc
    if deg_top_n > 0:
        top_idx = np.argsort(np.abs(logfc))[::-1][:deg_top_n]
        top_keep = np.zeros_like(keep, dtype=bool)
        top_keep[top_idx] = True
        keep = keep & top_keep
    return keep


def _top_k_indices(vec: np.ndarray, k: int) -> np.ndarray:
    k_eff = int(max(1, min(k, vec.shape[0])))
    return np.argsort(np.abs(vec))[::-1][:k_eff]


def _top_deg_recall_precision(delta_real: np.ndarray, delta_pred: np.ndarray, k: int) -> tuple[float, float]:
    real_top = set(_top_k_indices(delta_real, k).tolist())
    pred_top = set(_top_k_indices(delta_pred, k).tolist())
    inter = len(real_top & pred_top)
    # Fixed-k variant: both denominators are k.
    k_eff = float(min(max(1, k), delta_real.shape[0], delta_pred.shape[0]))
    recall = inter / k_eff
    precision = inter / k_eff
    return float(recall), float(precision)


def _direction_agreement(delta_real: np.ndarray, delta_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(delta_real) == np.sign(delta_pred)))


def _rank_norm_score(distances: np.ndarray, correct_idx: int) -> float:
    order = np.argsort(distances)
    rank = int(np.flatnonzero(order == correct_idx)[0])
    n = int(distances.shape[0])
    if n <= 1:
        return 1.0
    return float(1.0 - (rank / n))


def compute_metrics_with_space(
    real_path: Path,
    pred_path: Path,
    condition_column: str,
    metrics: List[str],
    min_cells_per_condition: int,
    *,
    space: str = "raw",
    n_components: int = 50,
    control_label: Optional[str] = None,
    deg_lfc: float = 0.25,
    deg_top_n: int = 0,
    top_k_deg: int = 50,
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

    ctrl_real_x = None
    ctrl_pred_x = None
    needs_control = (
        space == "deg" or any(m in SPACE_EXTRA_METRICS for m in metrics)
    )
    if needs_control and control_label is None:
        raise ValueError(
            "control_label is required for DEG space and delta/DEG metrics "
            "(pcc_delta, top_deg_recall, top_deg_precision, "
            "deg_direction_agreement, deg_spearman_lfc, pds_cosine)."
        )

    if control_label is not None and needs_control:
        ctrl_key = (control_label,)
        if ctrl_key in conditions:
            ctrl_real_x, ctrl_pred_x = extract_condition_matrices(
                pair,
                split=split,
                condition=ctrl_key,
                dense_mode="on_extract",
            )
            ctrl_real_x = _to_dense(ctrl_real_x)
            ctrl_pred_x = _to_dense(ctrl_pred_x)
        else:
            raise ValueError(
                f"Control label '{control_label}' not found in condition column '{condition_column}'."
            )

    rows: List[Dict] = []
    wants_pds = "pds_cosine" in metrics
    cond_to_delta_real: dict[str, np.ndarray] = {}
    cond_to_delta_pred: dict[str, np.ndarray] = {}

    for condition in conditions:
        real_x, pred_x = extract_condition_matrices(pair, split=split, condition=condition, dense_mode="on_extract")
        real_x = _to_dense(real_x)
        pred_x = _to_dense(pred_x)
        if real_x.shape[0] == 0 or pred_x.shape[0] == 0:
            continue

        deg_mask = None
        if space == "deg":
            deg_mask = _build_deg_mask(
                real_x=real_x,
                ctrl_real_x=ctrl_real_x,
                deg_lfc=deg_lfc,
                deg_top_n=deg_top_n,
            )

        tx_real, tx_pred, n_features = _apply_space_transform(
            real_x=real_x,
            pred_x=pred_x,
            space=space,
            n_components=n_components,
            deg_mask=deg_mask,
        )
        tx_ctrl_real = None
        tx_ctrl_pred = None
        if ctrl_real_x is not None and ctrl_pred_x is not None and space != "pca":
            tx_ctrl_real, tx_ctrl_pred, _ = _apply_space_transform(
                real_x=ctrl_real_x,
                pred_x=ctrl_pred_x,
                space=space,
                n_components=n_components,
                deg_mask=deg_mask,
            )

        row = {
            "condition": condition[0],
            "space": space,
            "n_real": int(real_x.shape[0]),
            "n_pred": int(pred_x.shape[0]),
            "n_genes_aligned": int(real_x.shape[1]),
            "n_features_eval": int(n_features),
        }

        for metric in metrics:
            if metric in SPACE_EXTRA_METRICS:
                if metric in {"pcc_delta", "top_deg_recall", "top_deg_precision", "deg_direction_agreement", "deg_spearman_lfc"} and space == "pca":
                    raise ValueError(
                        f"Metric '{metric}' is not supported in PCA space because gene identity is not preserved."
                    )
                if ctrl_real_x is None or ctrl_pred_x is None:
                    raise ValueError(
                        f"Metric '{metric}' requires a valid control condition from --control-label."
                    )
                if tx_ctrl_real is None or tx_ctrl_pred is None:
                    raise ValueError(
                        f"Metric '{metric}' requires control-transformed matrices in non-PCA space."
                    )

                delta_real = _safe_mean(tx_real) - _safe_mean(tx_ctrl_real)
                delta_pred = _safe_mean(tx_pred) - _safe_mean(tx_ctrl_pred)
                if wants_pds:
                    cond_to_delta_real[condition[0]] = delta_real
                    cond_to_delta_pred[condition[0]] = delta_pred

                if metric == "pcc_delta":
                    corr = pearsonr(delta_real, delta_pred)[0]
                    score = float(corr) if not np.isnan(corr) else np.nan
                elif metric in {"top_deg_recall", "top_deg_precision"}:
                    recall, precision = _top_deg_recall_precision(delta_real, delta_pred, top_k_deg)
                    score = recall if metric == "top_deg_recall" else precision
                elif metric == "deg_direction_agreement":
                    score = _direction_agreement(delta_real, delta_pred)
                elif metric == "deg_spearman_lfc":
                    corr = spearmanr(delta_real, delta_pred)[0]
                    score = float(corr) if not np.isnan(corr) else np.nan
                elif metric == "pds_cosine":
                    # Placeholder; computed after all conditions are processed.
                    score = np.nan
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
            elif metric in VECTOR_METRICS:
                score = _vector_metric(metric, tx_real, tx_pred)
            elif metric in DISTRIBUTION_METRICS:
                score = _distribution_metric(metric, tx_real, tx_pred)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            row[metric] = score

        rows.append(row)

    if wants_pds and len(rows) > 0:
        labels = [r["condition"] for r in rows]
        for i, cond in enumerate(labels):
            if cond not in cond_to_delta_pred or cond not in cond_to_delta_real:
                continue
            pred_eff = cond_to_delta_pred[cond]
            dists = []
            for other_cond in labels:
                real_eff = cond_to_delta_real.get(other_cond)
                if real_eff is None:
                    dists.append(np.nan)
                    continue
                d = cosine_distance_fn(pred_eff, real_eff)
                dists.append(float(d) if not np.isnan(d) else np.inf)
            dists_arr = np.asarray(dists, dtype=float)
            if not np.any(np.isfinite(dists_arr)):
                rows[i]["pds_cosine"] = np.nan
                continue
            rows[i]["pds_cosine"] = _rank_norm_score(dists_arr, correct_idx=i)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute perturbation metrics with raw/pca/deg evaluation space.")
    parser.add_argument("--real", required=True, help="Path to real/reference .h5ad")
    parser.add_argument("--pred", required=True, help="Path to predicted .h5ad")
    parser.add_argument("--condition-column", default="condition", help="obs column defining conditions")
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
        ],
        help="Metrics to compute",
    )
    parser.add_argument("--space", choices=["raw", "pca", "deg"], default="raw", help="Evaluation space")
    parser.add_argument("--n-components", type=int, default=50, help="PCA components when --space pca")
    parser.add_argument("--control-label", default=None, help="Control perturbation label for DEG space")
    parser.add_argument("--deg-lfc", type=float, default=0.25, help="Absolute log2 fold-change threshold for DEG space")
    parser.add_argument("--deg-top-n", type=int, default=0, help="Optional cap on number of DEGs (0 = no cap)")
    parser.add_argument("--top-k-deg", type=int, default=50, help="K for top DEG recall/precision metrics")
    parser.add_argument("--min-cells", type=int, default=1, help="Minimum cells per condition")
    parser.add_argument("--out", default="results/metrics_space.csv", help="Output CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = compute_metrics_with_space(
        real_path=Path(args.real),
        pred_path=Path(args.pred),
        condition_column=args.condition_column,
        metrics=args.metrics,
        min_cells_per_condition=args.min_cells,
        space=args.space,
        n_components=args.n_components,
        control_label=args.control_label,
        deg_lfc=args.deg_lfc,
        deg_top_n=args.deg_top_n,
        top_k_deg=args.top_k_deg,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    if not df.empty:
        present_metrics = [m for m in args.metrics if m in df.columns]
        summary = pd.DataFrame(
            {
                "metric": present_metrics,
                "mean_score": [float(df[m].mean()) for m in present_metrics],
            }
        )
        print("\nMean score per metric:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

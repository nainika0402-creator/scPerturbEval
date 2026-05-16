from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

from .compute_metrics import DISTRIBUTION_METRICS, VECTOR_METRICS, _distribution_metric, _vector_metric
from .loader import DatasetSpec, LoaderPolicy, PairedAnnDataLoader, extract_condition_matrices


def _to_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


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
    if control_label is not None and space == "deg":
        ctrl_key = (control_label,)
        if ctrl_key in conditions:
            ctrl_real_x, _ = extract_condition_matrices(pair, split=split, condition=ctrl_key, dense_mode="on_extract")
            ctrl_real_x = _to_dense(ctrl_real_x)

    rows: List[Dict] = []
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

        row = {
            "condition": condition[0],
            "space": space,
            "n_real": int(real_x.shape[0]),
            "n_pred": int(pred_x.shape[0]),
            "n_genes_aligned": int(real_x.shape[1]),
            "n_features_eval": int(n_features),
        }

        for metric in metrics:
            if metric in VECTOR_METRICS:
                score = _vector_metric(metric, tx_real, tx_pred)
            elif metric in DISTRIBUTION_METRICS:
                score = _distribution_metric(metric, tx_real, tx_pred)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            row[metric] = score

        rows.append(row)

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

from __future__ import annotations

import argparse
from pathlib import Path

from .metrics import compute_metrics_with_space


ALLOWED_METRICS = [
    "root_mean_squared_error",
    "pearson_distance",
    "wmse",
    "wasserstein",
    "mmd",
    "pcc_delta",
    "pearson_delta_pert",
    "weighted_r2_delta",
    "pds_cosine",
    "matrix_distance",
    "top_deg_recall",
    "deg_direction_agreement",
    "deg_spearman_lfc",
    "pathway_nes_spearman",
    "pathway_topk_jaccard",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected perturbation evaluation metrics with raw/pca/deg spaces."
    )
    parser.add_argument("--real", required=True, help="Path to real/reference .h5ad")
    parser.add_argument("--pred", required=True, help="Path to predicted .h5ad")
    parser.add_argument("--condition-column", default="condition", help="obs column defining conditions")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=ALLOWED_METRICS,
        help="Subset of supported metrics to compute",
    )
    parser.add_argument("--space", choices=["raw", "pca", "deg"], default="raw", help="Evaluation space")
    parser.add_argument("--n-components", type=int, default=50, help="PCA components when --space pca")
    parser.add_argument("--control-label", default=None, help="Control perturbation label for control-referenced metrics")
    parser.add_argument("--deg-lfc", type=float, default=0.25, help="Absolute log2 fold-change threshold for DEG space")
    parser.add_argument("--deg-top-n", type=int, default=0, help="Optional cap on number of DEGs (0 = no cap)")
    parser.add_argument("--top-k-deg", type=int, default=50, help="K for top DEG recall metric")
    parser.add_argument(
        "--pathway-gene-sets",
        default="MSigDB_Hallmark_2020",
        help="GSEApy gene sets name or GMT path for pathway recovery metrics.",
    )
    parser.add_argument("--pathway-top-k", type=int, default=10, help="Top-K pathways for pathway Jaccard.")
    parser.add_argument(
        "--pathway-reference",
        choices=["control", "perturbed_mean", "perturbed_centroid"],
        default="perturbed_centroid",
        help="Reference used to compute pathway deltas.",
    )
    parser.add_argument("--min-cells", type=int, default=1, help="Minimum cells per condition")
    parser.add_argument("--out", default="results/evaluations.csv", help="Output CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invalid = [m for m in args.metrics if m not in ALLOWED_METRICS]
    if invalid:
        raise ValueError(
            f"Unsupported metrics for evaluations.py: {invalid}. "
            f"Allowed metrics: {ALLOWED_METRICS}"
        )

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
        pathway_gene_sets=args.pathway_gene_sets,
        pathway_top_k=args.pathway_top_k,
        pathway_reference=args.pathway_reference,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    if not df.empty:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from pathlib import Path
import argparse

from scPerturbEval.compute_metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute perturbation metrics on paired h5ad files.")
    parser.add_argument("--real", required=True, help="Path to real/reference .h5ad")
    parser.add_argument("--pred", required=True, help="Path to predicted .h5ad")
    parser.add_argument("--condition-column", default="condition", help="obs column for condition")
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
    parser.add_argument("--min-cells", type=int, default=1, help="Minimum cells per condition")
    parser.add_argument("--out", default="results/metrics.csv", help="Output CSV path")
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


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


NON_METRIC_COLUMNS = {
    "condition",
    "space",
    "fold",
}


def _resolve_csv_paths(csv_paths: Iterable[str], pattern: str | None) -> list[Path]:
    out: list[Path] = [Path(p) for p in csv_paths]
    if pattern:
        out.extend(sorted(Path().glob(pattern)))
    out = [p for p in out if p.suffix.lower() == ".csv"]
    if not out:
        raise ValueError("No CSV files found. Pass --csv-paths and/or --pattern.")
    missing = [p for p in out if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CSV files: {[str(p) for p in missing]}")
    return out


def _infer_metric_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    metric_cols = [
        c
        for c in numeric_cols
        if c not in NON_METRIC_COLUMNS and not c.startswith("n_")
    ]
    return metric_cols


def aggregate_fold_metrics(
    csv_files: list[Path],
    metric_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_fold_rows: list[dict[str, float | str]] = []

    for fold_idx, path in enumerate(csv_files):
        df = pd.read_csv(path)
        if df.empty:
            continue

        cols = metric_columns or _infer_metric_columns(df)
        if not cols:
            raise ValueError(f"No metric columns found in {path}")

        row: dict[str, float | str] = {"fold": str(path)}
        for c in cols:
            if c not in df.columns:
                raise ValueError(f"Metric column '{c}' not found in {path}")
            row[c] = float(df[c].mean())
        per_fold_rows.append(row)

    if not per_fold_rows:
        raise ValueError("No fold rows produced (all inputs empty?).")

    per_fold_df = pd.DataFrame(per_fold_rows)
    metrics = [c for c in per_fold_df.columns if c != "fold"]

    summary = pd.DataFrame(
        {
            "metric": metrics,
            "mean": [float(per_fold_df[m].mean()) for m in metrics],
            "std": [float(per_fold_df[m].std(ddof=1)) if len(per_fold_df) > 1 else 0.0 for m in metrics],
            "n_folds": [int(len(per_fold_df)) for _ in metrics],
        }
    )

    return per_fold_df, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate fold metric CSVs into cross-fold mean/std."
    )
    parser.add_argument(
        "--csv-paths",
        nargs="*",
        default=[],
        help="Explicit metric CSV files (e.g. fold0_metrics.csv fold1_metrics.csv).",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Optional glob pattern (e.g. 'ckpts/additive/fold*/metrics_space.csv').",
    )
    parser.add_argument(
        "--metric-columns",
        nargs="*",
        default=None,
        help="Optional explicit metric columns; default: infer numeric non-count columns.",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path("results/fold_metrics_summary.csv"),
        help="Output CSV for cross-fold mean/std summary.",
    )
    parser.add_argument(
        "--out-per-fold",
        type=Path,
        default=Path("results/fold_metrics_per_fold.csv"),
        help="Output CSV for per-fold metric means.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_files = _resolve_csv_paths(args.csv_paths, args.pattern)
    per_fold_df, summary_df = aggregate_fold_metrics(
        csv_files=csv_files,
        metric_columns=args.metric_columns,
    )

    args.out_per_fold.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    per_fold_df.to_csv(args.out_per_fold, index=False)
    summary_df.to_csv(args.out_summary, index=False)

    print(f"Wrote per-fold means: {args.out_per_fold}")
    print(f"Wrote summary mean/std: {args.out_summary}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()


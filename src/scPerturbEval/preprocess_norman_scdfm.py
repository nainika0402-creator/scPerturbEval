from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import scanpy as sc


def _build_or_load_splits(
    adata,
    *,
    split_method: str,
    split_dir: Path,
    split_file_override: Path | None = None,
) -> list[dict]:
    split_dir.mkdir(parents=True, exist_ok=True)

    if split_method in {"additive", "combinations"}:
        split_file = split_file_override or (split_dir / "split_results.pkl")
        if split_file.exists():
            with split_file.open("rb") as f:
                return pickle.load(f)

        perturbations = np.unique(adata.obs["condition"])
        double_perturbation = np.array([p for p in perturbations if "ctrl" not in p])
        split_results: list[dict] = []
        for i in range(5):
            np.random.seed(42 + i)
            shuffled = double_perturbation.copy()
            np.random.shuffle(shuffled)
            split_idx = int(len(shuffled) * 0.3)
            test_double = shuffled[:split_idx]
            train_double = shuffled[split_idx:]
            split_results.append({"train": train_double.tolist(), "test": test_double.tolist()})

        with split_file.open("wb") as f:
            pickle.dump(split_results, f)
        return split_results

    if split_method == "unseen":
        split_file = split_file_override or (split_dir / "split_results_unseen.pkl")
        if split_file.exists():
            with split_file.open("rb") as f:
                return pickle.load(f)

        # Keep behavior aligned with scDFM implementation (non-deterministic by design).
        from random import shuffle

        split_results: list[dict] = []
        for _ in range(5):
            perturbations = np.unique(adata.obs["condition"])
            double_perturbation = [p for p in perturbations if "ctrl" not in p]
            single: list[str] = []
            [single.extend(p.split("+")) for p in double_perturbation]
            single = list(set(single))
            shuffle(single)
            remove_genes = single[:12]
            p_count = {}
            for p in double_perturbation:
                ps = p.split("+")
                count = int(ps[0] in remove_genes) + int(ps[1] in remove_genes)
                p_count[p] = count
            test_conditions = [p for p, count in p_count.items() if count > 0]
            test_conditions.extend([p + "+control" for p in remove_genes])
            split_results.append({"p_count": p_count, "test": test_conditions})

        with split_file.open("wb") as f:
            pickle.dump(split_results, f)
        return split_results

    raise ValueError(f"Unsupported split_method: {split_method}")


def preprocess_and_save_norman(
    norman_h5ad: Path,
    output_dir: Path,
    *,
    n_top_genes: int = 5000,
    infer_top_gene: int = 1000,
    split_method: str = "additive",
    fold: int = 0,
    split_file: Path | None = None,
) -> dict[str, Path]:
    adata = sc.read_h5ad(norman_h5ad)
    output_dir.mkdir(parents=True, exist_ok=True)

    sc.pp.highly_variable_genes(adata, inplace=True, n_top_genes=n_top_genes)
    unique_perturbation: list[str] = []
    [unique_perturbation.extend(p.split("+")) for p in adata.obs["condition"].unique()]
    unique_perturbation = list(np.unique(unique_perturbation))
    for perturbation in unique_perturbation:
        if perturbation in adata.var_names:
            adata.var.loc[perturbation, "highly_variable"] = True
    adata = adata[:, adata.var["highly_variable"]].copy()

    split_results = _build_or_load_splits(
        adata,
        split_method=split_method,
        split_dir=output_dir,
        split_file_override=split_file,
    )
    if fold < 0 or fold >= len(split_results):
        raise ValueError(f"Invalid fold {fold}. Expected 0..{len(split_results)-1}")

    adata.obs["condition"] = adata.obs["condition"].str.replace("ctrl", "control")
    adata.obs["Drug1"] = adata.obs["condition"].str.split("+").apply(lambda x: x[0])
    adata.obs["Drug2"] = adata.obs["condition"].str.split("+").apply(lambda x: x[-1])
    adata.obs["is_control"] = False
    adata.obs.loc[adata.obs["control"] == 1, "is_control"] = True
    adata.obs["mode"] = "train"

    if split_method == "combinations":
        split_results[fold]["test"] = split_results[fold]["test"][:15]
        remove_genes: list[str] = []
        [remove_genes.extend(p.split("+")) for p in split_results[fold]["test"]]
        remove_genes = list(set(remove_genes))
        split_results[fold]["test"].extend([p + "+control" for p in remove_genes])

    adata.obs.loc[adata.obs["condition"].isin(split_results[fold]["test"]), "mode"] = "test"

    adata_train = adata[adata.obs["mode"] == "train"].copy()
    adata_test = adata[(adata.obs["mode"] == "test") | (adata.obs["control"] == 1)].copy()

    sc.pp.highly_variable_genes(adata_test, inplace=True, n_top_genes=infer_top_gene)
    adata_test = adata_test[:, adata_test.var["highly_variable"]].copy()

    all_path = output_dir / "norman_processed_all.h5ad"
    train_path = output_dir / "norman_processed_train.h5ad"
    test_path = output_dir / "norman_processed_test.h5ad"
    adata.write_h5ad(all_path)
    adata_train.write_h5ad(train_path)
    adata_test.write_h5ad(test_path)

    meta = {
        "source_h5ad": str(norman_h5ad),
        "split_method": split_method,
        "fold": int(fold),
        "n_top_genes": int(n_top_genes),
        "infer_top_gene": int(infer_top_gene),
        "split_file": str(split_file) if split_file is not None else None,
        "n_obs_all": int(adata.n_obs),
        "n_vars_all": int(adata.n_vars),
        "n_obs_train": int(adata_train.n_obs),
        "n_vars_train": int(adata_train.n_vars),
        "n_obs_test": int(adata_test.n_obs),
        "n_vars_test": int(adata_test.n_vars),
    }
    (output_dir / "norman_preprocess_meta.json").write_text(json.dumps(meta, indent=2))

    return {
        "all": all_path,
        "train": train_path,
        "test": test_path,
        "meta": output_dir / "norman_preprocess_meta.json",
    }


def load_preprocessed_norman(processed_dir: Path):
    all_path = processed_dir / "norman_processed_all.h5ad"
    train_path = processed_dir / "norman_processed_train.h5ad"
    test_path = processed_dir / "norman_processed_test.h5ad"
    meta_path = processed_dir / "norman_preprocess_meta.json"

    missing = [p for p in [all_path, train_path, test_path, meta_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing preprocessed files: {[str(p) for p in missing]}")

    adata_all = sc.read_h5ad(all_path)
    adata_train = sc.read_h5ad(train_path)
    adata_test = sc.read_h5ad(test_path)
    meta = json.loads(meta_path.read_text())
    return adata_all, adata_train, adata_test, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scDFM-style Norman preprocessing and save outputs.")
    parser.add_argument("--norman-h5ad", type=Path, required=True, help="Path to raw Norman .h5ad")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write preprocessed files")
    parser.add_argument("--split-method", choices=["additive", "combinations", "unseen"], default="additive")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-top-genes", type=int, default=5000)
    parser.add_argument("--infer-top-gene", type=int, default=1000)
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Optional path to split pickle. If present, load it; if missing, generate and save there.",
    )
    parser.add_argument("--load-only", action="store_true", help="Load existing preprocessed files and print shapes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.load_only:
        adata_all, adata_train, adata_test, meta = load_preprocessed_norman(args.out_dir)
        print(f"Loaded all: {adata_all.shape}, train: {adata_train.shape}, test: {adata_test.shape}")
        print(json.dumps(meta, indent=2))
        return

    paths = preprocess_and_save_norman(
        norman_h5ad=args.norman_h5ad,
        output_dir=args.out_dir,
        n_top_genes=args.n_top_genes,
        infer_top_gene=args.infer_top_gene,
        split_method=args.split_method,
        fold=args.fold,
        split_file=args.split_file,
    )
    print("Wrote:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()

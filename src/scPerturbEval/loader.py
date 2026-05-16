"""Single-file paired AnnData loader.

This module provides a compact, configurable loader for paired real/predicted
AnnData objects with:
- path or in-memory AnnData ingestion
- configurable copy behavior
- schema/column validation
- gene alignment strategies
- optional split-aware condition indexing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

try:
    import anndata as ad
    import scanpy as sc
except ImportError as exc:  # pragma: no cover
    raise ImportError("anndata and scanpy are required. Install with: pip install anndata scanpy") from exc


# -----------------------------
# Errors
# -----------------------------


class DataLoadError(Exception):
    """Base loader exception."""


class SchemaError(DataLoadError):
    """Raised when required schema elements are missing or invalid."""


class MissingColumnError(SchemaError):
    """Raised when an expected obs column is missing."""


class GeneAlignmentError(DataLoadError):
    """Raised when gene alignment fails."""


class SplitError(DataLoadError):
    """Raised when split configuration is invalid."""


# -----------------------------
# Contracts
# -----------------------------


DataInput = Union[str, Path, ad.AnnData]
ConditionKey = tuple[str, ...]
SplitMap = dict[str, np.ndarray]
MaskMap = dict[ConditionKey, np.ndarray]


@dataclass(frozen=True)
class DatasetSpec:
    """Schema contract for paired datasets."""

    condition_columns: list[str]
    covariate_columns: list[str] = field(default_factory=list)
    split_column: Optional[str] = None
    control_value: Optional[str] = None
    min_cells_per_condition: int = 1
    require_categorical: bool = False


@dataclass(frozen=True)
class LoaderPolicy:
    """Behavior policy for alignment, loading and extraction."""

    gene_alignment: Literal["strict_equal", "intersection", "reference_order"] = "intersection"
    split_policy: Literal["from_obs", "from_external", "none"] = "from_obs"
    dense_mode: Literal["never", "on_extract", "always"] = "on_extract"
    fail_on_warnings: bool = False
    copy_mode: Literal["deep", "none"] = "deep"


@dataclass(frozen=True)
class LoadedAnnData:
    adata: ad.AnnData
    source: str
    backed: bool


@dataclass(frozen=True)
class GeneAlignmentResult:
    ref: ad.AnnData
    pred: ad.AnnData
    common_genes: pd.Index
    dropped_ref: int
    dropped_pred: int


@dataclass(frozen=True)
class ConditionIndex:
    split_to_conditions: dict[str, list[ConditionKey]]
    ref_masks: dict[str, MaskMap]
    pred_masks: dict[str, MaskMap]


@dataclass(frozen=True)
class LoadReport:
    ref_shape: tuple[int, int]
    pred_shape: tuple[int, int]
    n_common_genes: int
    dropped_ref_genes: int
    dropped_pred_genes: int
    split_counts: dict[str, int]
    warnings: list[str]


@dataclass(frozen=True)
class PreparedPair:
    ref: ad.AnnData
    pred: ad.AnnData
    condition_index: ConditionIndex
    report: LoadReport


# -----------------------------
# Components
# -----------------------------


class InputResolver:
    """Resolves path or AnnData input to an AnnData object."""

    def __init__(self, copy_mode: Literal["deep", "none"] = "deep"):
        self.copy_mode = copy_mode

    def resolve(self, data: DataInput, *, backed: Optional[str] = None) -> LoadedAnnData:
        if isinstance(data, (str, Path)):
            path = Path(data)
            if not path.exists():
                raise DataLoadError(f"Input file not found: {path}")
            # read_h5ad ignores backed mode in some versions; sc.read handles backed.
            adata = sc.read(path, backed=backed)
            return LoadedAnnData(adata=adata, source=str(path), backed=backed is not None)

        if not isinstance(data, ad.AnnData):
            raise DataLoadError(f"Unsupported input type: {type(data)}")

        if self.copy_mode == "deep":
            return LoadedAnnData(adata=data.copy(), source="<AnnData>", backed=False)
        return LoadedAnnData(adata=data, source="<AnnData>", backed=False)


class SchemaValidator:
    """Validates obs/var contracts for a pair of AnnData objects."""

    def validate_obs(self, adata: ad.AnnData, spec: DatasetSpec, role: str) -> None:
        required_obs = set(spec.condition_columns) | set(spec.covariate_columns)
        if spec.split_column is not None:
            required_obs.add(spec.split_column)

        missing = [c for c in required_obs if c not in adata.obs.columns]
        if missing:
            raise MissingColumnError(f"Missing required obs columns in {role}: {missing}")

        if spec.require_categorical:
            for col in spec.condition_columns:
                if not pd.api.types.is_categorical_dtype(adata.obs[col]):
                    raise SchemaError(
                        f"Condition column '{col}' in {role} is not categorical. "
                        "Set require_categorical=False or cast it."
                    )

    def validate_var(self, adata: ad.AnnData, role: str) -> None:
        if adata.var_names is None or len(adata.var_names) == 0:
            raise SchemaError(f"{role} has empty var_names")
        if adata.X is None:
            raise SchemaError(f"{role} has no X matrix")

    def validate_pair(self, ref: ad.AnnData, pred: ad.AnnData, spec: DatasetSpec) -> None:
        self.validate_obs(ref, spec, "ref")
        self.validate_obs(pred, spec, "pred")
        self.validate_var(ref, "ref")
        self.validate_var(pred, "pred")


class FeatureAligner:
    """Aligns genes between ref and pred according to policy."""

    def align(self, ref: ad.AnnData, pred: ad.AnnData, policy: LoaderPolicy) -> GeneAlignmentResult:
        ref_genes = pd.Index(ref.var_names.astype(str))
        pred_genes = pd.Index(pred.var_names.astype(str))

        if policy.gene_alignment == "strict_equal":
            if not ref_genes.equals(pred_genes):
                raise GeneAlignmentError("strict_equal policy failed: var_names differ")
            common = ref_genes
            aligned_ref = ref.copy()
            aligned_pred = pred.copy()
        else:
            common = ref_genes.intersection(pred_genes)
            if len(common) == 0:
                raise GeneAlignmentError("No overlapping var_names between ref and pred")

            if policy.gene_alignment == "reference_order":
                # Keep intersection in reference order.
                common = ref_genes[ref_genes.isin(common)]

            ref_idx = ref_genes.get_indexer(common)
            pred_idx = pred_genes.get_indexer(common)
            aligned_ref = ref[:, ref_idx].copy()
            aligned_pred = pred[:, pred_idx].copy()
            aligned_ref.var_names = common
            aligned_pred.var_names = common

        if policy.dense_mode == "always":
            aligned_ref.X = _to_dense(aligned_ref.X)
            aligned_pred.X = _to_dense(aligned_pred.X)

        dropped_ref = len(ref_genes) - len(common)
        dropped_pred = len(pred_genes) - len(common)

        return GeneAlignmentResult(
            ref=aligned_ref,
            pred=aligned_pred,
            common_genes=common,
            dropped_ref=dropped_ref,
            dropped_pred=dropped_pred,
        )


class ConditionIndexer:
    """Builds split-wise condition masks for aligned AnnData pairs."""

    def build(
        self,
        ref: ad.AnnData,
        pred: ad.AnnData,
        spec: DatasetSpec,
        split_map: Optional[SplitMap],
        policy: LoaderPolicy,
    ) -> ConditionIndex:
        split_names = _resolve_splits(ref, spec, split_map, policy)

        split_to_conditions: dict[str, list[ConditionKey]] = {}
        ref_masks: dict[str, MaskMap] = {}
        pred_masks: dict[str, MaskMap] = {}

        for split_name in split_names:
            ref_m = self._build_masks_for_adata(
                adata=ref,
                condition_columns=spec.condition_columns,
                min_cells=spec.min_cells_per_condition,
                split_name=split_name,
                split_column=spec.split_column,
                split_map=split_map,
                is_ref=True,
            )
            pred_m = self._build_masks_for_adata(
                adata=pred,
                condition_columns=spec.condition_columns,
                min_cells=spec.min_cells_per_condition,
                split_name=split_name,
                split_column=spec.split_column,
                split_map=split_map,
                is_ref=False,
            )

            common_keys = sorted(set(ref_m.keys()) & set(pred_m.keys()))
            ref_masks[split_name] = {k: ref_m[k] for k in common_keys}
            pred_masks[split_name] = {k: pred_m[k] for k in common_keys}
            split_to_conditions[split_name] = common_keys

        return ConditionIndex(
            split_to_conditions=split_to_conditions,
            ref_masks=ref_masks,
            pred_masks=pred_masks,
        )

    @staticmethod
    def _build_masks_for_adata(
        adata: ad.AnnData,
        condition_columns: list[str],
        min_cells: int,
        split_name: str,
        split_column: Optional[str],
        split_map: Optional[SplitMap],
        is_ref: bool,
    ) -> MaskMap:
        base_mask = np.ones(adata.n_obs, dtype=bool)

        if split_name != "all":
            if split_map is not None and is_ref:
                idx = split_map.get(split_name)
                if idx is None:
                    raise SplitError(f"split_map missing split '{split_name}'")
                base_mask = np.zeros(adata.n_obs, dtype=bool)
                base_mask[idx] = True
            elif split_column is not None and split_column in adata.obs.columns:
                base_mask &= (adata.obs[split_column].astype(str).values == split_name)

        obs_sub = adata.obs.loc[base_mask, condition_columns].astype(str)
        unique_conditions = obs_sub.drop_duplicates()

        masks: MaskMap = {}
        for _, row in unique_conditions.iterrows():
            key = tuple(str(row[c]) for c in condition_columns)
            mask = base_mask.copy()
            for col in condition_columns:
                mask &= (adata.obs[col].astype(str).values == str(row[col]))
            if int(mask.sum()) >= min_cells:
                masks[key] = mask
        return masks


# -----------------------------
# Facade
# -----------------------------


class PairedAnnDataLoader:
    """Facade for preparing aligned, indexed paired AnnData datasets."""

    def __init__(
        self,
        spec: DatasetSpec,
        policy: LoaderPolicy = LoaderPolicy(),
        validator: Optional[SchemaValidator] = None,
        resolver: Optional[InputResolver] = None,
        aligner: Optional[FeatureAligner] = None,
        indexer: Optional[ConditionIndexer] = None,
    ):
        self.spec = spec
        self.policy = policy
        self.validator = validator or SchemaValidator()
        self.resolver = resolver or InputResolver(copy_mode=policy.copy_mode)
        self.aligner = aligner or FeatureAligner()
        self.indexer = indexer or ConditionIndexer()

    def prepare(
        self,
        ref_data: DataInput,
        pred_data: DataInput,
        *,
        split_map: Optional[SplitMap] = None,
        backed: Optional[str] = None,
    ) -> PreparedPair:
        warnings: list[str] = []

        loaded_ref = self.resolver.resolve(ref_data, backed=backed)
        loaded_pred = self.resolver.resolve(pred_data, backed=backed)

        ref = loaded_ref.adata
        pred = loaded_pred.adata

        self.validator.validate_pair(ref, pred, self.spec)

        alignment = self.aligner.align(ref, pred, self.policy)
        ref_aligned, pred_aligned = alignment.ref, alignment.pred

        if alignment.dropped_ref > 0 or alignment.dropped_pred > 0:
            warnings.append(
                f"Aligned to {len(alignment.common_genes)} common genes "
                f"(dropped ref={alignment.dropped_ref}, pred={alignment.dropped_pred})"
            )

        condition_index = self.indexer.build(
            ref=ref_aligned,
            pred=pred_aligned,
            spec=self.spec,
            split_map=split_map,
            policy=self.policy,
        )

        split_counts = {
            split: len(keys) for split, keys in condition_index.split_to_conditions.items()
        }

        report = LoadReport(
            ref_shape=(int(ref_aligned.n_obs), int(ref_aligned.n_vars)),
            pred_shape=(int(pred_aligned.n_obs), int(pred_aligned.n_vars)),
            n_common_genes=int(len(alignment.common_genes)),
            dropped_ref_genes=int(alignment.dropped_ref),
            dropped_pred_genes=int(alignment.dropped_pred),
            split_counts=split_counts,
            warnings=warnings,
        )

        if self.policy.fail_on_warnings and warnings:
            raise DataLoadError("Warnings encountered with fail_on_warnings=True: " + "; ".join(warnings))

        return PreparedPair(
            ref=ref_aligned,
            pred=pred_aligned,
            condition_index=condition_index,
            report=report,
        )


# -----------------------------
# Utility
# -----------------------------


def _resolve_splits(
    ref: ad.AnnData,
    spec: DatasetSpec,
    split_map: Optional[SplitMap],
    policy: LoaderPolicy,
) -> list[str]:
    if policy.split_policy == "none":
        return ["all"]

    if policy.split_policy == "from_external":
        if split_map is None:
            raise SplitError("split_policy='from_external' requires split_map")
        return list(split_map.keys())

    # from_obs
    if spec.split_column is None:
        return ["all"]
    if spec.split_column not in ref.obs.columns:
        raise SplitError(f"split_column '{spec.split_column}' not found in ref.obs")

    values = ref.obs[spec.split_column].astype(str).unique().tolist()
    return values if values else ["all"]


def _to_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def extract_condition_matrices(
    pair: PreparedPair,
    split: str,
    condition: ConditionKey,
    *,
    dense_mode: Literal["never", "on_extract", "always"] = "on_extract",
) -> Tuple[Union[np.ndarray, sparse.spmatrix], Union[np.ndarray, sparse.spmatrix]]:
    """Extract matched matrices for one split/condition from PreparedPair."""
    if split not in pair.condition_index.ref_masks:
        raise KeyError(f"Unknown split: {split}")

    ref_mask = pair.condition_index.ref_masks[split][condition]
    pred_mask = pair.condition_index.pred_masks[split][condition]

    ref_x = pair.ref.X[ref_mask]
    pred_x = pair.pred.X[pred_mask]

    if dense_mode == "on_extract":
        return _to_dense(ref_x), _to_dense(pred_x)
    return ref_x, pred_x

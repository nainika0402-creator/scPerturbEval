from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn

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


def _tokenize_condition(cond: str) -> list[str]:
    return [t.strip() for t in cond.split("+") if t.strip() and t.strip().lower() != "control"]


def _build_condition_vocab(conditions: list[str]) -> list[str]:
    toks: set[str] = set()
    for c in conditions:
        toks.update(_tokenize_condition(c))
    return sorted(toks)


def _condition_to_multihot(cond: str, token_to_idx: dict[str, int]) -> np.ndarray:
    x = np.zeros(len(token_to_idx), dtype=np.float32)
    for t in _tokenize_condition(cond):
        if t in token_to_idx:
            x[token_to_idx[t]] = 1.0
    return x


class OneHiddenLayerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.fc2(self.fc1(x)))


def _train_one_layer_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden_dim: int,
    steps: int,
    lr: float,
    seed: int,
    early_stopping: bool,
    val_fraction: float,
    patience: int,
    min_delta: float,
) -> OneHiddenLayerMLP:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = OneHiddenLayerMLP(
        input_dim=x_train.shape[1],
        hidden_dim=hidden_dim,
        output_dim=y_train.shape[1],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x_t_all = torch.tensor(x_train, dtype=torch.float32)
    y_t_all = torch.tensor(y_train, dtype=torch.float32)

    use_es = early_stopping and x_train.shape[0] >= 3 and val_fraction > 0.0
    if use_es:
        n = x_train.shape[0]
        n_val = max(1, int(round(n * val_fraction)))
        n_val = min(n_val, n - 1)
        perm = np.random.permutation(n)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        x_tr = x_t_all[tr_idx]
        y_tr = y_t_all[tr_idx]
        x_val = x_t_all[val_idx]
        y_val = y_t_all[val_idx]
        best_state = None
        best_val = float("inf")
        bad_steps = 0
    else:
        x_tr = x_t_all
        y_tr = y_t_all
        x_val = None
        y_val = None

    model.train()
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_tr)
        loss = criterion(pred, y_tr)
        loss.backward()
        optimizer.step()
        if use_es:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)  # type: ignore[arg-type]
                val_loss = float(criterion(val_pred, y_val).item())  # type: ignore[arg-type]
            model.train()
            if val_loss < (best_val - min_delta):
                best_val = val_loss
                bad_steps = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_steps += 1
                if bad_steps >= patience:
                    break

    if use_es and best_state is not None:
        model.load_state_dict(best_state)

    return model


def _build_linear_baseline_predictions(
    adata_train: ad.AnnData,
    adata_test: ad.AnnData,
    *,
    covariate_col: str | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    train_x = _to_dense(adata_train.X)
    test_x = _to_dense(adata_test.X)
    train_obs = adata_train.obs.copy()
    test_obs = adata_test.obs.copy()

    for col in ["condition", "is_control"]:
        if col not in train_obs.columns or col not in test_obs.columns:
            raise ValueError(f"Linear baseline requires '{col}' in both train/test .obs")
    if covariate_col is not None and (covariate_col not in train_obs.columns or covariate_col not in test_obs.columns):
        raise ValueError(f"covariate_col '{covariate_col}' not found in both train/test .obs")

    train_ctrl_mask = train_obs["is_control"].to_numpy().astype(bool)
    train_pert_mask = ~train_ctrl_mask
    train_ctrl_idx = np.where(train_ctrl_mask)[0]
    train_pert_idx = np.where(train_pert_mask)[0]
    if len(train_ctrl_idx) == 0 or len(train_pert_idx) == 0:
        raise ValueError("Linear baseline needs both control and perturbed train cells.")

    if covariate_col is None:
        matched_ctrl_idx = rng.choice(train_ctrl_idx, size=len(train_pert_idx), replace=True)
    else:
        cov_train = train_obs[covariate_col].astype(str).to_numpy()
        matched_ctrl_idx = np.zeros(len(train_pert_idx), dtype=int)
        for k, pi in enumerate(train_pert_idx):
            pool = train_ctrl_idx[cov_train[train_ctrl_idx] == cov_train[pi]]
            if len(pool) == 0:
                pool = train_ctrl_idx
            matched_ctrl_idx[k] = int(rng.choice(pool))

    y_delta = train_x[train_pert_idx] - train_x[matched_ctrl_idx]
    x_train_df = pd.DataFrame({"perturbation": train_obs.iloc[train_pert_idx]["condition"].astype(str).to_numpy()})
    if covariate_col is not None:
        x_train_df["covariate"] = train_obs.iloc[train_pert_idx][covariate_col].astype(str).to_numpy()
    x_train = pd.get_dummies(x_train_df, dtype=float)

    lin = LinearRegression(fit_intercept=False)
    lin.fit(x_train.to_numpy(), y_delta)

    eval_mask = test_obs["mode"].astype(str).to_numpy() == "test"
    eval_obs = test_obs.loc[eval_mask].copy()
    eval_x = test_x[eval_mask]
    if eval_x.shape[0] == 0:
        raise ValueError("No test perturbation cells (mode=='test') found in test data.")

    x_eval_df = pd.DataFrame({"perturbation": eval_obs["condition"].astype(str).to_numpy()})
    if covariate_col is not None:
        x_eval_df["covariate"] = eval_obs[covariate_col].astype(str).to_numpy()
    x_eval = pd.get_dummies(x_eval_df, dtype=float).reindex(columns=x_train.columns, fill_value=0.0)
    pred_delta = lin.predict(x_eval.to_numpy())

    test_ctrl_mask = test_obs["is_control"].to_numpy().astype(bool)
    test_ctrl_idx = np.where(test_ctrl_mask)[0]
    if len(test_ctrl_idx) == 0:
        raise ValueError("No control cells found in test data for linear baseline prediction.")

    eval_idx_in_test = np.where(eval_mask)[0]
    control_base = np.zeros_like(eval_x, dtype=np.float32)
    if covariate_col is None:
        chosen_ctrl_idx = rng.choice(test_ctrl_idx, size=len(eval_idx_in_test), replace=True)
        control_base = test_x[chosen_ctrl_idx].astype(np.float32)
    else:
        cov_test = test_obs[covariate_col].astype(str).to_numpy()
        for row_i, ti in enumerate(eval_idx_in_test):
            pool = test_ctrl_idx[cov_test[test_ctrl_idx] == cov_test[ti]]
            if len(pool) == 0:
                pool = test_ctrl_idx
            control_base[row_i] = test_x[int(rng.choice(pool))]

    pred_x = control_base + pred_delta.astype(np.float32)
    return pred_x, eval_x, eval_obs


def run_baseline(
    processed_dir: Path,
    baseline: str,
    out_dir: Path,
    *,
    steps: int = 5000,
    hidden_dim: int = 1024,
    lr: float = 1e-3,
    seed: int = 42,
    early_stopping: bool = True,
    val_fraction: float = 0.2,
    patience: int = 20,
    min_delta: float = 1e-4,
    covariate_col: str | None = None,
) -> dict[str, Path]:
    adata_all, adata_train, adata_test, _meta = load_preprocessed_norman(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shared_genes = adata_train.var_names.intersection(adata_test.var_names)
    if len(shared_genes) == 0:
        raise ValueError("No shared genes between preprocessed train and test data.")

    adata_train = adata_train[:, shared_genes].copy()
    adata_test = adata_test[:, shared_genes].copy()

    train_x = _to_dense(adata_train.X)
    test_x = _to_dense(adata_test.X)
    test_obs = adata_test.obs.copy()

    if "condition" not in test_obs:
        raise ValueError("Expected 'condition' in preprocessed test .obs")
    if "mode" not in test_obs:
        raise ValueError("Expected 'mode' in preprocessed test .obs")

    train_obs = adata_train.obs.copy()
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
    mlp_model = None
    mlp_token_to_idx: dict[str, int] | None = None
    if baseline == "one_layer_mlp":
        train_conditions = sorted([c for c in train_condition_means.keys() if c.lower() not in {"control", "control+control"}])
        if len(train_conditions) == 0:
            raise ValueError("No non-control train conditions for one_layer_mlp.")
        vocab = _build_condition_vocab(train_conditions)
        if len(vocab) == 0:
            raise ValueError("No perturbation tokens found for one_layer_mlp.")
        mlp_token_to_idx = {t: i for i, t in enumerate(vocab)}
        x_train = np.vstack([_condition_to_multihot(c, mlp_token_to_idx) for c in train_conditions]).astype(np.float32)
        y_train = np.vstack([train_condition_means[c] for c in train_conditions]).astype(np.float32)
        mlp_model = _train_one_layer_mlp(
            x_train=x_train,
            y_train=y_train,
            hidden_dim=hidden_dim,
            steps=steps,
            lr=lr,
            seed=seed,
            early_stopping=early_stopping,
            val_fraction=val_fraction,
            patience=patience,
            min_delta=min_delta,
        )
        mlp_model.eval()

    if baseline == "linear_baseline":
        pred_x, eval_x, eval_obs = _build_linear_baseline_predictions(
            adata_train=adata_train,
            adata_test=adata_test,
            covariate_col=covariate_col,
            seed=seed,
        )
    else:
        for cond in eval_obs["condition"].astype(str).unique():
            idx = np.where(eval_obs["condition"].astype(str).to_numpy() == cond)[0]
            if baseline == "control_baseline":
                pred_vec = control_mean_test
            elif baseline == "global_delta":
                pred_vec = control_mean_test + delta_global
            elif baseline == "one_layer_mlp":
                assert mlp_model is not None and mlp_token_to_idx is not None
                x_eval = torch.tensor(_condition_to_multihot(cond, mlp_token_to_idx)[None, :], dtype=torch.float32)
                with torch.no_grad():
                    pred_vec = mlp_model(x_eval).cpu().numpy()[0]
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
        choices=["control_baseline", "global_delta", "one_layer_mlp", "linear_baseline"],
        required=True,
    )
    parser.add_argument("--steps", type=int, default=5000, help="Training steps for one_layer_mlp baseline")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden size for one_layer_mlp baseline")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for one_layer_mlp baseline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for one_layer_mlp baseline")
    parser.add_argument("--early-stopping", dest="early_stopping", action="store_true", help="Enable early stopping for one_layer_mlp")
    parser.add_argument("--no-early-stopping", dest="early_stopping", action="store_false", help="Disable early stopping for one_layer_mlp")
    parser.set_defaults(early_stopping=True)
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction for one_layer_mlp early stopping")
    parser.add_argument("--patience", type=int, default=20, help="Patience steps for one_layer_mlp early stopping")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum validation improvement for one_layer_mlp early stopping")
    parser.add_argument(
        "--covariate-col",
        default=None,
        help="Optional covariate obs column for linear_baseline matched-control sampling and features.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for predictions and metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_baseline(
        processed_dir=args.processed_dir,
        baseline=args.baseline,
        out_dir=args.out_dir,
        steps=args.steps,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        seed=args.seed,
        early_stopping=args.early_stopping,
        val_fraction=args.val_fraction,
        patience=args.patience,
        min_delta=args.min_delta,
        covariate_col=args.covariate_col,
    )
    print("Wrote:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

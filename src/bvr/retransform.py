"""Cross-fitted retransformation for Curve 2 (BUILD PACK Section 5 follow-up,
CONTEXT.md D14).

`monetary_model.fit`'s Duan (1983) smearing factor is computed from residuals
on the same rows the booster was fit on -- the booster has already partly
memorised those rows, so the factor is biased low and under-forecasts spend
on unseen customers. This module computes a cross-fitted, per-prediction-level
("per bin") smearing factor instead: every row's residual comes from a fold
where that row's customer was held out of the fit, so it generalises to
unseen customers the way `predict_spend` must at serving time.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

N_BINS = 10
N_RESIDUAL_QUANTILES = 201
MIN_RESIDUALS_PER_BIN = 30


def fit_retransform(
    rows_active: pd.DataFrame,
    feature_cols: list[str],
    categorical: list[str],
    params: dict,
    num_boost_round: int,
    seed: int = 26,
) -> dict:
    """rows_active: ALL training-split customer-month rows with active == 1."""
    rows_active = rows_active.reset_index(drop=True)
    y = np.log1p(rows_active["net_revenue"].clip(lower=0)).to_numpy()
    customer_ids = rows_active["Customer ID"].to_numpy()
    customers = np.sort(rows_active["Customer ID"].unique())

    oof = np.full(len(rows_active), np.nan)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(customers):
        train_mask = np.isin(customer_ids, customers[train_idx])
        test_mask = np.isin(customer_ids, customers[test_idx])

        X_train = rows_active.loc[train_mask, feature_cols]
        y_train = y[train_mask]
        train_set = lgb.Dataset(
            X_train, label=y_train, categorical_feature=categorical, free_raw_data=False
        )
        booster = lgb.train(params, train_set, num_boost_round=num_boost_round)

        X_test = rows_active.loc[test_mask, feature_cols]
        oof[test_mask] = booster.predict(X_test, num_iteration=num_boost_round)

    if np.isnan(oof).any():
        raise ValueError("some rows lack an out-of-fold prediction")

    resid = y - oof
    edges = np.quantile(oof, np.linspace(0, 1, N_BINS + 1))[1:-1]
    bin_idx = np.digitize(oof, edges)

    smearing = []
    residual_quantiles = []
    for b in range(N_BINS):
        mask = bin_idx == b
        n = int(mask.sum())
        if n < MIN_RESIDUALS_PER_BIN:
            raise ValueError(f"bin {b} has only {n} residuals (< {MIN_RESIDUALS_PER_BIN})")
        smearing.append(float(np.mean(np.exp(resid[mask]))))
        residual_quantiles.append(
            np.quantile(resid[mask], np.linspace(0, 1, N_RESIDUAL_QUANTILES)).tolist()
        )

    return {
        "method": "cross_fitted_per_bin_smearing",
        "edges": edges.tolist(),
        "smearing": smearing,
        "residual_quantiles": residual_quantiles,
        "global_smearing": float(np.mean(np.exp(resid))),
        "n_rows": len(rows_active),
        "n_customers": len(customers),
        "seed": seed,
    }


def bin_index(pred_log: np.ndarray, rt: dict) -> np.ndarray:
    return np.digitize(np.asarray(pred_log, dtype=float), np.asarray(rt["edges"], dtype=float))


def expected_spend(pred_log: np.ndarray, rt: dict) -> np.ndarray:
    pred_log = np.asarray(pred_log, dtype=float)
    b = bin_index(pred_log, rt)
    smearing = np.asarray(rt["smearing"], dtype=float)[b]
    return np.clip(smearing * np.exp(pred_log) - 1.0, a_min=0.0, a_max=None)

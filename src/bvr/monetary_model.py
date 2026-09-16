"""Curve 2 -- spend model (BUILD PACK Section 5).

E[net revenue in month t | active, features, t]. LightGBM regressor on
log1p(net revenue), back-transformed with a Duan (1983) smearing factor
computed on the fit rows only (never the holdout).

Design decision: log1p + smearing, not raw-scale regression. Rejected
alternative: raw-scale regression -- a small number of very large buyers
dominate the squared-error loss and starve the model of signal on
everyone else (BUILD PACK Section 5).
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from bvr.features import ALL_FEATURES, CATEGORICAL_FEATURES

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "outputs" / "checkpoints" / "curve2_spend.txt"

FEATURE_COLS = ALL_FEATURES + ["t"]

LGB_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "seed": 26,
    "verbose": -1,
}


def fit(rows: pd.DataFrame, train_ids: set, seed: int = 26):
    train_rows = rows[rows["Customer ID"].isin(train_ids) & (rows["active"] == 1)]
    train_customers = sorted(train_rows["Customer ID"].unique().tolist())
    fit_cust, es_cust = train_test_split(train_customers, test_size=0.10, random_state=seed)

    fit_rows = train_rows[train_rows["Customer ID"].isin(fit_cust)]
    es_rows = train_rows[train_rows["Customer ID"].isin(es_cust)]

    X_fit = fit_rows[FEATURE_COLS]
    y_fit = np.log1p(fit_rows["net_revenue"].clip(lower=0))
    X_es = es_rows[FEATURE_COLS]
    y_es = np.log1p(es_rows["net_revenue"].clip(lower=0))

    train_set = lgb.Dataset(
        X_fit, label=y_fit, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False
    )
    es_set = lgb.Dataset(
        X_es, label=y_es, categorical_feature=CATEGORICAL_FEATURES,
        reference=train_set, free_raw_data=False,
    )

    booster = lgb.train(
        LGB_PARAMS,
        train_set,
        num_boost_round=2000,
        valid_sets=[es_set],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    pred_log_fit = booster.predict(X_fit, num_iteration=booster.best_iteration)
    resid = y_fit.to_numpy() - pred_log_fit
    smearing = float(np.mean(np.exp(resid)))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_PATH))
    info = {
        "best_iteration": int(booster.best_iteration),
        "fit_customers": len(fit_cust),
        "early_stop_customers": len(es_cust),
        "smearing_factor": smearing,
    }
    return booster, smearing, info


def predict_spend(booster: lgb.Booster, smearing: float, rows: pd.DataFrame) -> np.ndarray:
    pred_log = booster.predict(rows[FEATURE_COLS], num_iteration=booster.best_iteration)
    return np.clip(smearing * np.exp(pred_log) - 1.0, a_min=0.0, a_max=None)


def load(path: Path = MODEL_PATH) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))

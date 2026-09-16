"""Curve 1 -- activity model (BUILD PACK Section 5).

P(active in month t | customer features, t). LightGBM binary classifier,
`t` is a feature. Early stopping on a 10% slice of TRAIN customers only --
the 30% holdout is never touched while fitting.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from bvr.features import ALL_FEATURES, CATEGORICAL_FEATURES

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "outputs" / "checkpoints" / "curve1_activity.txt"

FEATURE_COLS = ALL_FEATURES + ["t"]

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "seed": 26,
    "verbose": -1,
}


def fit(rows: pd.DataFrame, train_ids: set, seed: int = 26):
    train_rows = rows[rows["Customer ID"].isin(train_ids)]
    train_customers = sorted(train_rows["Customer ID"].unique().tolist())
    fit_cust, es_cust = train_test_split(train_customers, test_size=0.10, random_state=seed)

    fit_rows = train_rows[train_rows["Customer ID"].isin(fit_cust)]
    es_rows = train_rows[train_rows["Customer ID"].isin(es_cust)]

    X_fit, y_fit = fit_rows[FEATURE_COLS], fit_rows["active"]
    X_es, y_es = es_rows[FEATURE_COLS], es_rows["active"]

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
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_PATH))
    info = {
        "best_iteration": int(booster.best_iteration),
        "fit_customers": len(fit_cust),
        "early_stop_customers": len(es_cust),
    }
    return booster, info


def predict_proba(booster: lgb.Booster, rows: pd.DataFrame) -> np.ndarray:
    return booster.predict(rows[FEATURE_COLS], num_iteration=booster.best_iteration)


def load(path: Path = MODEL_PATH) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))

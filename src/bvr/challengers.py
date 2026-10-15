"""Train-only customer-value challengers. Never reads or tunes against the holdout.

Run after the existing 'features' step:
    python -m bvr.challengers
Writes outputs/challenger_validation.json. No model is automatically promoted.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from bvr.features import ALL_FEATURES, CATEGORICAL_FEATURES

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"
SEED = 26


def target_totals(monthly: pd.DataFrame) -> pd.DataFrame:
    """One row/customer. Keep signed net-revenue targets, including cancellations."""
    needed = {"Customer ID", "t", "net_revenue"}
    if not needed.issubset(monthly.columns):
        raise ValueError("monthly targets missing required columns")
    counts = monthly.groupby("Customer ID")["t"].nunique()
    if not counts.eq(12).all():
        raise ValueError("every customer must have all 12 forecast months")
    y6 = monthly.loc[monthly["t"] <= 6].groupby("Customer ID")["net_revenue"].sum()
    y12 = monthly.groupby("Customer ID")["net_revenue"].sum()
    return pd.DataFrame({"actual_6m": y6, "actual_12m": y12})


def _wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    den = float(np.abs(actual).sum())
    return float(np.abs(actual - predicted).sum() / den) if den else float("nan")


def score(actual6: np.ndarray, actual12: np.ndarray, pred6: np.ndarray, pred12: np.ndarray) -> dict:
    """Customer-level and portfolio-level accuracy are separate by design."""
    actual6 = np.asarray(actual6, dtype=float)
    actual12 = np.asarray(actual12, dtype=float)
    pred6 = np.asarray(pred6, dtype=float)
    pred12 = np.asarray(pred12, dtype=float)
    if not (len(actual6) == len(actual12) == len(pred6) == len(pred12)):
        raise ValueError("score array lengths differ")
    den12 = float(np.abs(actual12).sum())
    order = np.argsort(pred12, kind="stable")
    n_top = max(1, int(np.ceil(len(order) / 10)))
    top = order[-n_top:]
    rank_corr = pd.Series(pred12).corr(pd.Series(actual12), method="spearman")
    return {
        "wape_6m": _wape(actual6, pred6),
        "wape_12m": _wape(actual12, pred12),
        "portfolio_revenue_gap_12m": (
            float(abs(pred12.sum() - actual12.sum()) / den12) if den12 else float("nan")
        ),
        "spearman_12m": float(rank_corr) if pd.notna(rank_corr) else None,
        "top_decile_capture_12m": (
            float(actual12[top].sum() / actual12.sum()) if actual12.sum() else None
        ),
        "n_customers": len(actual12),
    }


def _matrix(df: pd.DataFrame, known_categories: dict[str, list]) -> pd.DataFrame:
    data = df[ALL_FEATURES].copy()
    for col, categories in known_categories.items():
        data[col] = pd.Categorical(data[col], categories=categories)
    return data


def run_validation(
    features: pd.DataFrame,
    monthly: pd.DataFrame,
    train_ids: set,
    holdout_ids: set,
    seed: int = SEED,
) -> dict:
    """Split train customers again. Neither training nor selection touches holdout."""
    train_ids, holdout_ids = set(train_ids), set(holdout_ids)
    if not train_ids or train_ids.intersection(holdout_ids):
        raise ValueError("empty/overlapping train and holdout customer IDs")
    customers = set(features["Customer ID"])
    if not train_ids.issubset(customers):
        raise ValueError("some training IDs have no feature row")
    ids = np.asarray(sorted(train_ids))
    fit_ids, val_ids = train_test_split(ids, test_size=0.20, random_state=seed)
    fit_ids, val_ids = set(fit_ids), set(val_ids)
    if (fit_ids & val_ids) or (fit_ids & holdout_ids) or (val_ids & holdout_ids):
        raise AssertionError("customer split leakage")

    targets = target_totals(monthly)
    frame = features.set_index("Customer ID").join(targets, how="inner")
    fit = frame.loc[sorted(fit_ids)]
    val = frame.loc[sorted(val_ids)]
    known = {
        col: sorted(fit[col].dropna().unique().tolist())
        for col in CATEGORICAL_FEATURES
    }
    x_fit, x_val = _matrix(fit, known), _matrix(val, known)
    y6, y12 = val["actual_6m"].to_numpy(), val["actual_12m"].to_numpy()

    # Holdout-access-free baseline: carry trailing spend forward unchanged.
    history = val["monetary_total"].clip(lower=0).to_numpy()
    candidates = {"repeat_spend": score(y6, y12, history / 2, history)}
    for key, objective in [("direct_mae", "regression_l1"), ("direct_tweedie", "tweedie")]:
        forecasts = {}
        for horizon in (6, 12):
            col = "actual_" + str(horizon) + "m"
            # Tweedie requires non-negative outcomes; cancellations are still
            # retained in the signed ground-truth error computation.
            train_target = fit[col].clip(lower=0).to_numpy()
            model = lgb.LGBMRegressor(
                objective=objective,
                n_estimators=300,
                learning_rate=0.04,
                num_leaves=15,
                min_child_samples=25,
                colsample_bytree=0.9,
                random_state=seed,
                verbosity=-1,
            )
            model.fit(x_fit, train_target, categorical_feature=CATEGORICAL_FEATURES)
            forecasts[horizon] = np.maximum(model.predict(x_val), 0)
        candidates[key] = score(y6, y12, forecasts[6], forecasts[12])

    winner = min(candidates, key=lambda key: candidates[key]["wape_12m"])
    return {
        "selection_set": "20% of existing training customers only",
        "fit_customers": len(fit_ids),
        "validation_customers": len(val_ids),
        "holdout_customers_accessed": 0,
        "holdout_ids_excluded": len(holdout_ids),
        "target_definition": "signed net revenue over months 1-6 and 1-12",
        "negative_12m_targets_in_validation": int((y12 < 0).sum()),
        "comparison": candidates,
        "best_validation_wape_12m": winner,
        "release_status": "experiment_only; no automatic promotion or claims of improved holdout performance",
    }


def main() -> None:
    feat_path = OUT_DIR / "features_t0.parquet"
    target_path = OUT_DIR / "targets_monthly.parquet"
    split_path = OUT_DIR / "split_ids.json"
    if not all(path.exists() for path in (feat_path, target_path, split_path)):
        raise SystemExit("Run 'python -m bvr.cli features' with the original dataset first")
    split = json.loads(split_path.read_text())
    result = run_validation(
        pd.read_parquet(feat_path),
        pd.read_parquet(target_path),
        set(split["train_ids"]),
        set(split["holdout_ids"]),
    )
    out = OUT_DIR / "challenger_validation.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(out), "selected_on_validation": result["best_validation_wape_12m"]}))


if __name__ == "__main__":
    main()

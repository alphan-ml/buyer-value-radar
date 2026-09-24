"""Evaluation for Buyer Value Radar (BUILD PACK Section 6). Everything here
runs on the 30% holdout. Nothing on the site is typed by hand -- every
number comes from outputs/metrics.json and outputs/calibration.json,
written by this module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold

from bvr import combine as combine_mod
from bvr import monetary_model, predictive, retransform, survival_model
from bvr.features import CATEGORICAL_FEATURES

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"


def wape(pred: np.ndarray, actual: np.ndarray) -> float:
    denom = float(np.sum(np.abs(actual)))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(pred - actual)) / denom)


def mape_pos_actual(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float, int]:
    """Mean and median absolute percentage error, actual > 0 only.

    Design decision: only the median is written to metrics.json (CONTEXT.md
    D16). The mean is dominated by customers whose actual is a tiny positive
    residual of cancellation netting (e.g. GBP 0.01) -- one such row can send
    the mean into the trillions, so it is computed here (for callers that
    want it) but not reported. Rejected alternative: winsorizing or dropping
    near-zero actuals -- that would silently change a real, verified number
    rather than disclose it.
    """
    mask = actual > 0
    if mask.sum() == 0:
        return float("nan"), float("nan"), 0
    vals = np.abs((pred[mask] - actual[mask]) / actual[mask])
    return float(np.mean(vals)), float(np.median(vals)), int(mask.sum())


def spearman_corr(pred: np.ndarray, actual: np.ndarray) -> float:
    if len(pred) < 2:
        return float("nan")
    corr = pd.Series(pred).corr(pd.Series(actual), method="spearman")
    return float(corr) if pd.notna(corr) else float("nan")


def top_decile_capture(pred: np.ndarray, actual: np.ndarray) -> float:
    """Share of actual spend held by the top predicted decile."""
    total = float(np.sum(actual))
    if total == 0:
        return float("nan")
    df = pd.DataFrame({"pred": pred, "actual": actual})
    df["decile"] = pd.qcut(df["pred"].rank(method="first"), 10, labels=False) + 1
    top_actual = float(df.loc[df["decile"] == 10, "actual"].sum())
    return top_actual / total


def _decile_table(pred: np.ndarray, actual: np.ndarray) -> list[dict]:
    df = pd.DataFrame({"pred_12": pred, "actual_12": actual})
    df["decile"] = pd.qcut(df["pred_12"].rank(method="first"), 10, labels=False) + 1
    return _seg_table(df, "decile", pred_col="pred_12", actual_col="actual_12")


def repeat_baseline(feat: pd.DataFrame) -> pd.DataFrame:
    """Naive baseline: each customer's own trailing spend (`monetary_total`,
    the feature table's spend over the feature window ending at T0 -- see
    CONTEXT.md D10) carried forward as the 12-month prediction, halved for
    the 6-month prediction."""
    out = feat[["Customer ID", "monetary_total"]].copy()
    out["pred_12"] = out["monetary_total"].clip(lower=0.0)
    out["pred_6"] = out["pred_12"] / 2.0
    return out[["Customer ID", "pred_6", "pred_12"]]


def segment_mean_baseline(
    feat: pd.DataFrame, train_ids: set, actual_6: pd.Series, actual_12: pd.Series
) -> pd.DataFrame:
    """Naive baseline: bucket every customer into a buyer-size tercile using
    edges computed from TRAINING customers' `monetary_total` only, then
    predict the training-tercile's mean actual spend (training rows only,
    at each horizon) -- see CONTEXT.md D11. Applying training-only edges and
    training-only means to holdout customers keeps this leak-free."""
    train_feat = feat[feat["Customer ID"].isin(train_ids)][["Customer ID", "monetary_total"]].copy()
    _, edges = pd.qcut(train_feat["monetary_total"], 3, retbins=True, duplicates="drop")
    edges = np.asarray(edges, dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf
    labels = ["low", "mid", "high"][: len(edges) - 1]

    train_feat["tercile"] = pd.cut(train_feat["monetary_total"], bins=edges, labels=labels)
    train_feat = train_feat.set_index("Customer ID")
    train_actual_6 = actual_6.reindex(train_feat.index).fillna(0.0)
    train_actual_12 = actual_12.reindex(train_feat.index).fillna(0.0)
    mean_6 = train_actual_6.groupby(train_feat["tercile"], observed=True).mean()
    mean_12 = train_actual_12.groupby(train_feat["tercile"], observed=True).mean()

    out = feat[["Customer ID", "monetary_total"]].copy()
    out["tercile"] = pd.cut(out["monetary_total"], bins=edges, labels=labels)
    out["pred_6"] = out["tercile"].map(mean_6).astype(float)
    out["pred_12"] = out["tercile"].map(mean_12).astype(float)
    return out[["Customer ID", "pred_6", "pred_12"]]


def _baseline_metrics(pred_df: pd.DataFrame, actual_6: pd.Series, actual_12: pd.Series) -> dict:
    m = (
        pred_df.set_index("Customer ID")
        .join(actual_6.rename("actual_6"))
        .join(actual_12.rename("actual_12"))
        .fillna(0.0)
    )
    p6, p12 = m["pred_6"].to_numpy(), m["pred_12"].to_numpy()
    a6, a12 = m["actual_6"].to_numpy(), m["actual_12"].to_numpy()
    return {
        "wape_6m": wape(p6, a6),
        "wape_12m": wape(p12, a12),
        "decile_table_12m": _decile_table(p12, a12),
        "spearman_12m": spearman_corr(p12, a12),
        "top_decile_capture_12m": top_decile_capture(p12, a12),
    }


def _seg_table(
    df: pd.DataFrame, col: str, pred_col: str = "ltv_12", actual_col: str = "actual_12"
) -> list[dict]:
    """Per-group error table, one row per value of `col`.

    Two error metrics are reported, and they can diverge sharply within a
    group where some customers are over-predicted and others under-predicted:

    - `wape`: customer-level WAPE -- sum(|predicted_i - actual_i|) over the
      customers in the group, divided by sum(|actual_i|). This is what the
      model actually gets right or wrong per customer; over- and
      under-estimates do not cancel.
    - `aggregate_revenue_error`: the net error if the group's totals were
      used on their own -- |sum(predicted_i) - sum(actual_i)| / sum(actual_i).
      Kept as a separate, clearly-named metric because it can look much
      better than `wape` purely from cancellation, not accuracy.
    """

    def _group_metrics(g: pd.DataFrame) -> pd.Series:
        pred = g[pred_col].to_numpy()
        actual = g[actual_col].to_numpy()
        pred_sum = float(pred.sum())
        actual_sum = float(actual.sum())
        return pd.Series(
            {
                "customers": len(g),
                "predicted": pred_sum,
                "actual": actual_sum,
                "wape": wape(pred, actual),
                "aggregate_revenue_error": wape(
                    np.array([pred_sum]), np.array([actual_sum])
                ),
            }
        )

    g = df.groupby(col, observed=True)[[pred_col, actual_col]].apply(_group_metrics)
    g["customers"] = g["customers"].astype(int)
    return g.reset_index().to_dict(orient="records")


def _revenue_error(pred: np.ndarray, actual: np.ndarray) -> float:
    """Signed (pred - actual) / actual on the totals -- positive means over-forecast."""
    denom = float(np.sum(actual))
    if denom == 0:
        return float("nan")
    return float((np.sum(pred) - denom) / denom)


def bootstrap_wape_ci(
    pred: np.ndarray, actual: np.ndarray, n_boot: int = 1000, seed: int = 26
) -> list[float]:
    """95% bootstrap CI (2.5/97.5 percentiles) for WAPE, resampling customers
    with replacement."""
    rng = np.random.default_rng(seed)
    n = len(pred)
    idx = rng.integers(0, n, size=(n_boot, n))
    vals = np.array([wape(pred[i], actual[i]) for i in idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def evaluate(
    holdout_rows: pd.DataFrame,   # (Customer ID, t, active, net_revenue, ...features) holdout rows
    p_active_pred: pd.DataFrame,  # (Customer ID, t, value) predicted P(active)
    e_spend_pred: pd.DataFrame,   # (Customer ID, t, value) predicted E[spend|active]
    ltv_pred: pd.DataFrame,       # (Customer ID, ltv_6, ltv_12) -- the exact mean
    predictive_pred: pd.DataFrame,  # (Customer ID, p10_6, p50_6, p90_6, p10_12, p50_12, p90_12)
    feat: pd.DataFrame,           # customer features (for the segment table and baselines)
    train_rows: pd.DataFrame,     # same shape as holdout_rows, but for TRAIN customers only
    train_ids: set,               # train customer ids (for the segment-mean baseline)
    cv_results: dict,             # section D cross-validation table (bvr.eval.cross_validate)
) -> tuple[dict, dict]:
    actual_6 = holdout_rows[holdout_rows["t"] <= 6].groupby("Customer ID")["net_revenue"].sum()
    actual_12 = holdout_rows.groupby("Customer ID")["net_revenue"].sum()

    merged = (
        ltv_pred.set_index("Customer ID")
        .join(actual_6.rename("actual_6"))
        .join(actual_12.rename("actual_12"))
        .join(predictive_pred.set_index("Customer ID"))
        .fillna(0.0)
    )

    wape_6 = wape(merged["ltv_6"].to_numpy(), merged["actual_6"].to_numpy())
    wape_12 = wape(merged["ltv_12"].to_numpy(), merged["actual_12"].to_numpy())
    wape_6m_p50 = wape(merged["p50_6"].to_numpy(), merged["actual_6"].to_numpy())
    wape_12m_p50 = wape(merged["p50_12"].to_numpy(), merged["actual_12"].to_numpy())
    revenue_error_6m_mean = _revenue_error(
        merged["ltv_6"].to_numpy(), merged["actual_6"].to_numpy()
    )
    revenue_error_12m_mean = _revenue_error(
        merged["ltv_12"].to_numpy(), merged["actual_12"].to_numpy()
    )
    interval_coverage_12m_p10_p90 = float(
        np.mean(
            (merged["actual_12"].to_numpy() >= merged["p10_12"].to_numpy())
            & (merged["actual_12"].to_numpy() <= merged["p90_12"].to_numpy())
        )
    )
    wape_12m_p50_ci95 = bootstrap_wape_ci(
        merged["p50_12"].to_numpy(), merged["actual_12"].to_numpy()
    )
    _mape_6, median_ape_6, n_pos_6 = mape_pos_actual(
        merged["ltv_6"].to_numpy(), merged["actual_6"].to_numpy()
    )
    _mape_12, median_ape_12, n_pos_12 = mape_pos_actual(
        merged["ltv_12"].to_numpy(), merged["actual_12"].to_numpy()
    )

    # Calibration by month: sum predicted vs sum actual across holdout customers, per t.
    p_e = p_active_pred.merge(e_spend_pred, on=["Customer ID", "t"], suffixes=("_p", "_spend"))
    p_e["pred_contribution"] = p_e["value_p"] * p_e["value_spend"]
    by_month_pred = p_e.groupby("t")["pred_contribution"].sum()
    by_month_actual = holdout_rows.groupby("t")["net_revenue"].sum()
    calibration_by_month = [
        {
            "t": int(t),
            "predicted": float(by_month_pred.get(t, 0.0)),
            "actual": float(by_month_actual.get(t, 0.0)),
        }
        for t in range(1, 13)
    ]

    # Calibration by decile (12-month value). Same two-metric shape as
    # _seg_table: `wape` is customer-level (errors don't cancel within a
    # decile), `aggregate_revenue_error` is the net error on the decile's
    # totals.
    dec = merged.copy()
    dec["decile"] = pd.qcut(dec["ltv_12"].rank(method="first"), 10, labels=False) + 1
    decile_table = pd.DataFrame(_seg_table(dec, "decile"))

    # Curve 1 alone: ROC-AUC and Brier by t.
    c1 = holdout_rows.merge(p_active_pred, on=["Customer ID", "t"])
    curve1_by_t = []
    for t, g in c1.groupby("t"):
        y = g["active"].to_numpy()
        p = g["value"].to_numpy()
        auc = float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))
        brier = float(brier_score_loss(y, p))
        curve1_by_t.append({"t": int(t), "roc_auc": auc, "brier": brier, "n": len(g)})

    # Curve 2 alone: WAPE on active rows.
    c2 = holdout_rows[holdout_rows["active"] == 1].merge(e_spend_pred, on=["Customer ID", "t"])
    curve2_wape = wape(c2["value"].to_numpy(), c2["net_revenue"].to_numpy())

    # Segment table (12-month, holdout).
    seg_base = merged.join(
        feat.set_index("Customer ID")[["country_group", "first_purchase_quarter", "monetary_total"]]
    )
    seg_base["buyer_size_tercile"] = pd.qcut(
        seg_base["monetary_total"].rank(method="first"), 3, labels=["low", "mid", "high"]
    )

    # Baselines (BUILD PACK follow-up: see CONTEXT.md D10/D11) -- same holdout
    # customers, same actual_6/actual_12.
    holdout_feat = feat[feat["Customer ID"].isin(merged.index)]
    train_actual_6 = train_rows[train_rows["t"] <= 6].groupby("Customer ID")["net_revenue"].sum()
    train_actual_12 = train_rows.groupby("Customer ID")["net_revenue"].sum()

    repeat_pred = repeat_baseline(holdout_feat)
    segment_mean_pred = segment_mean_baseline(feat, train_ids, train_actual_6, train_actual_12)
    segment_mean_pred = segment_mean_pred[segment_mean_pred["Customer ID"].isin(merged.index)]

    baselines = {
        "repeat": _baseline_metrics(repeat_pred, actual_6, actual_12),
        "segment_mean": _baseline_metrics(segment_mean_pred, actual_6, actual_12),
    }

    ranking = {
        "spearman_12m": spearman_corr(merged["ltv_12"].to_numpy(), merged["actual_12"].to_numpy()),
        "top_decile_capture_12m": top_decile_capture(
            merged["ltv_12"].to_numpy(), merged["actual_12"].to_numpy()
        ),
    }

    metrics = {
        "holdout_customers": len(merged),
        "point_forecast": "p50",
        # Aliases of the _mean fields, kept for one release so nothing
        # reading the old names breaks (CONTEXT.md D15).
        "wape_6m": wape_6,
        "wape_12m": wape_12,
        "wape_6m_mean": wape_6,
        "wape_12m_mean": wape_12,
        "wape_6m_p50": wape_6m_p50,
        "wape_12m_p50": wape_12m_p50,
        "revenue_error_6m_mean": revenue_error_6m_mean,
        "revenue_error_12m_mean": revenue_error_12m_mean,
        "interval_coverage_12m_p10_p90": interval_coverage_12m_p10_p90,
        "wape_12m_p50_ci95": wape_12m_p50_ci95,
        "median_ape_6m_actual_gt0": median_ape_6,
        "mape_6m_n": n_pos_6,
        "median_ape_12m_actual_gt0": median_ape_12,
        "mape_12m_n": n_pos_12,
        "curve1_by_t": curve1_by_t,
        "curve2_wape_active_rows": curve2_wape,
        "decile_table_12m": decile_table.to_dict(orient="records"),
        "segment_country_group": _seg_table(seg_base, "country_group"),
        "segment_buyer_size_tercile": _seg_table(seg_base, "buyer_size_tercile"),
        "segment_first_purchase_quarter": _seg_table(seg_base, "first_purchase_quarter"),
        "baselines": baselines,
        "ranking": ranking,
        "cv": cv_results,
    }
    calibration = {"calibration_by_month": calibration_by_month}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (OUT_DIR / "calibration.json").write_text(json.dumps(calibration, indent=2, default=str))
    return metrics, calibration


def _fold_predictive_table(
    p_active_df: pd.DataFrame, pred_log_df: pd.DataFrame, rt: dict
) -> pd.DataFrame:
    """p_active_df / pred_log_df: (Customer ID, t, value) for one fold's test
    customers. Returns one row per customer: p50_6, p50_12, p10_12, p90_12.
    predictive.summarize's seed is always 26 (see its docstring), never the
    fold/rep seed."""
    p_active_wide = p_active_df.pivot(index="Customer ID", columns="t", values="value").sort_index(
        axis=1
    )
    pred_log_wide = pred_log_df.pivot(index="Customer ID", columns="t", values="value").sort_index(
        axis=1
    )
    records = []
    for cust in p_active_wide.index:
        summary = predictive.summarize(
            p_active_wide.loc[cust].to_numpy(),
            pred_log_wide.loc[cust].to_numpy(),
            rt,
            horizons=(6, 12),
            seed=26,
        )
        records.append(
            {
                "Customer ID": cust,
                "p50_6": summary[6]["p50"],
                "p50_12": summary[12]["p50"],
                "p10_12": summary[12]["p10"],
                "p90_12": summary[12]["p90"],
            }
        )
    return pd.DataFrame.from_records(records).set_index("Customer ID")


def cross_validate(
    rows: pd.DataFrame,
    customer_ids: set,
    seed_base: int = 26,
    n_repeats: int = 3,
    n_splits: int = 5,
) -> dict:
    """Section D: 5-fold x 3-repeat customer cross-validation over ALL
    customers (CONTEXT.md D16) -- the headline evaluation, because the
    single 30% holdout is noisy enough (bootstrap 95% interval 0.60-0.80 on
    12-month WAPE) that its point estimate is not trustworthy on its own.

    Each fold refits Curve 1, Curve 2 and the cross-fitted retransform
    exactly as production (same functions, same early-stop logic), only on
    that fold's training customers, then scores the fold's held-out
    customers. `revenue_error_12m_mean_old_in_sample` reports what the
    pre-fix (in-sample smearing) method would have given, for comparison.
    """
    customers = np.sort(np.array(sorted(customer_ids)))
    per_fold: list[dict] = []

    for rep in range(n_repeats):
        seed = seed_base + rep
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in kf.split(customers):
            fold_train_ids = set(customers[train_idx].tolist())
            fold_test_ids = set(customers[test_idx].tolist())

            booster1, _info1 = survival_model.fit(rows, fold_train_ids, seed=seed, save=False)
            booster2, smearing_in_sample, info2 = monetary_model.fit(
                rows, fold_train_ids, seed=seed, save=False
            )

            train_rows_active = rows[
                rows["Customer ID"].isin(fold_train_ids) & (rows["active"] == 1)
            ]
            rt = retransform.fit_retransform(
                train_rows_active,
                monetary_model.FEATURE_COLS,
                CATEGORICAL_FEATURES,
                monetary_model.LGB_PARAMS,
                num_boost_round=info2["best_iteration"],
                seed=seed,
            )
            # Tweedie variant, refit the same way, so the D13 promotion gate
            # (below, in cli.step_eval) can compare cross-validated numbers
            # instead of the noisy single holdout.
            booster2t, _vp, _info2t = monetary_model.fit_tweedie(
                rows, fold_train_ids, seed=seed, save=False
            )

            test_rows = rows[rows["Customer ID"].isin(fold_test_ids)]
            pred_log_arr = booster2.predict(
                test_rows[monetary_model.FEATURE_COLS], num_iteration=booster2.best_iteration
            )
            p_active_arr = survival_model.predict_proba(booster1, test_rows)
            e_spend_new_arr = retransform.expected_spend(pred_log_arr, rt)
            e_spend_old_arr = np.clip(
                smearing_in_sample * np.exp(pred_log_arr) - 1.0, a_min=0.0, a_max=None
            )
            e_spend_tweedie_arr = monetary_model.predict_spend_tweedie(booster2t, test_rows)

            p_active_df = test_rows[["Customer ID", "t"]].copy()
            p_active_df["value"] = p_active_arr
            e_spend_new_df = test_rows[["Customer ID", "t"]].copy()
            e_spend_new_df["value"] = e_spend_new_arr
            e_spend_old_df = test_rows[["Customer ID", "t"]].copy()
            e_spend_old_df["value"] = e_spend_old_arr
            e_spend_tweedie_df = test_rows[["Customer ID", "t"]].copy()
            e_spend_tweedie_df["value"] = e_spend_tweedie_arr
            pred_log_df = test_rows[["Customer ID", "t"]].copy()
            pred_log_df["value"] = pred_log_arr

            ltv_new, _ = combine_mod.combine(p_active_df, e_spend_new_df, horizons=(6, 12))
            ltv_old, _ = combine_mod.combine(p_active_df, e_spend_old_df, horizons=(6, 12))
            ltv_tweedie, _ = combine_mod.combine(p_active_df, e_spend_tweedie_df, horizons=(6, 12))

            actual_6 = test_rows[test_rows["t"] <= 6].groupby("Customer ID")["net_revenue"].sum()
            actual_12 = test_rows.groupby("Customer ID")["net_revenue"].sum()

            pred_table = _fold_predictive_table(p_active_df, pred_log_df, rt)

            merged_new = (
                ltv_new.set_index("Customer ID")
                .join(actual_6.rename("actual_6"))
                .join(actual_12.rename("actual_12"))
                .join(pred_table)
                .fillna(0.0)
            )
            merged_old = (
                ltv_old.set_index("Customer ID")
                .join(actual_6.rename("actual_6"))
                .join(actual_12.rename("actual_12"))
                .fillna(0.0)
            )
            merged_tweedie = (
                ltv_tweedie.set_index("Customer ID")
                .join(actual_6.rename("actual_6"))
                .join(actual_12.rename("actual_12"))
                .fillna(0.0)
            )

            per_fold.append(
                {
                    "wape_12m_p50": wape(
                        merged_new["p50_12"].to_numpy(), merged_new["actual_12"].to_numpy()
                    ),
                    "wape_12m_mean": wape(
                        merged_new["ltv_12"].to_numpy(), merged_new["actual_12"].to_numpy()
                    ),
                    "wape_6m_p50": wape(
                        merged_new["p50_6"].to_numpy(), merged_new["actual_6"].to_numpy()
                    ),
                    "wape_6m_mean": wape(
                        merged_new["ltv_6"].to_numpy(), merged_new["actual_6"].to_numpy()
                    ),
                    "revenue_error_12m_mean": _revenue_error(
                        merged_new["ltv_12"].to_numpy(), merged_new["actual_12"].to_numpy()
                    ),
                    "revenue_error_12m_mean_old_in_sample": _revenue_error(
                        merged_old["ltv_12"].to_numpy(), merged_old["actual_12"].to_numpy()
                    ),
                    "spearman_12m": spearman_corr(
                        merged_new["ltv_12"].to_numpy(), merged_new["actual_12"].to_numpy()
                    ),
                    "top_decile_capture_12m": top_decile_capture(
                        merged_new["ltv_12"].to_numpy(), merged_new["actual_12"].to_numpy()
                    ),
                    "interval_coverage_12m_p10_p90": float(
                        np.mean(
                            (merged_new["actual_12"].to_numpy() >= merged_new["p10_12"].to_numpy())
                            & (merged_new["actual_12"].to_numpy() <= merged_new["p90_12"].to_numpy())
                        )
                    ),
                    "wape_12m_tweedie": wape(
                        merged_tweedie["ltv_12"].to_numpy(), merged_tweedie["actual_12"].to_numpy()
                    ),
                    "top_decile_capture_12m_tweedie": top_decile_capture(
                        merged_tweedie["ltv_12"].to_numpy(), merged_tweedie["actual_12"].to_numpy()
                    ),
                }
            )

    fold_df = pd.DataFrame(per_fold)
    summary = {
        col: {"mean": float(fold_df[col].mean()), "sd": float(fold_df[col].std(ddof=1))}
        for col in fold_df.columns
    }
    return {
        "n_repeats": n_repeats,
        "n_splits": n_splits,
        "n_folds": len(per_fold),
        "seed_base": seed_base,
        "metrics": summary,
    }

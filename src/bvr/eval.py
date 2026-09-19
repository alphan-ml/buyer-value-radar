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

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"


def wape(pred: np.ndarray, actual: np.ndarray) -> float:
    denom = float(np.sum(np.abs(actual)))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(pred - actual)) / denom)


def mape_pos_actual(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float, int]:
    """Mean and median absolute percentage error, actual > 0 only.

    Design decision: mean MAPE is reported (BUILD PACK Section 6 asks for
    it explicitly) but is dominated by customers whose actual is a tiny
    positive residual of cancellation netting (e.g. GBP 0.01) -- one such
    row can send the mean into the trillions. Median APE is reported
    alongside it so the number on the page is not misleading. Rejected
    alternative: winsorizing or dropping near-zero actuals -- that would
    silently change a real, verified number rather than disclose it.
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


def evaluate(
    holdout_rows: pd.DataFrame,   # (Customer ID, t, active, net_revenue, ...features) holdout rows
    p_active_pred: pd.DataFrame,  # (Customer ID, t, value) predicted P(active)
    e_spend_pred: pd.DataFrame,   # (Customer ID, t, value) predicted E[spend|active]
    ltv_pred: pd.DataFrame,       # (Customer ID, ltv_6, ltv_12)
    feat: pd.DataFrame,           # customer features (for the segment table and baselines)
    train_rows: pd.DataFrame,     # same shape as holdout_rows, but for TRAIN customers only
    train_ids: set,               # train customer ids (for the segment-mean baseline)
) -> tuple[dict, dict]:
    actual_6 = holdout_rows[holdout_rows["t"] <= 6].groupby("Customer ID")["net_revenue"].sum()
    actual_12 = holdout_rows.groupby("Customer ID")["net_revenue"].sum()

    merged = (
        ltv_pred.set_index("Customer ID")
        .join(actual_6.rename("actual_6"))
        .join(actual_12.rename("actual_12"))
        .fillna(0.0)
    )

    wape_6 = wape(merged["ltv_6"].to_numpy(), merged["actual_6"].to_numpy())
    wape_12 = wape(merged["ltv_12"].to_numpy(), merged["actual_12"].to_numpy())
    mape_6, median_ape_6, n_pos_6 = mape_pos_actual(
        merged["ltv_6"].to_numpy(), merged["actual_6"].to_numpy()
    )
    mape_12, median_ape_12, n_pos_12 = mape_pos_actual(
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
        "wape_6m": wape_6,
        "wape_12m": wape_12,
        "mape_6m_actual_gt0": mape_6,
        "median_ape_6m_actual_gt0": median_ape_6,
        "mape_6m_n": n_pos_6,
        "mape_12m_actual_gt0": mape_12,
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
    }
    calibration = {"calibration_by_month": calibration_by_month}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (OUT_DIR / "calibration.json").write_text(json.dumps(calibration, indent=2, default=str))
    return metrics, calibration

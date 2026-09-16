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


def _seg_table(df: pd.DataFrame, col: str) -> list[dict]:
    g = df.groupby(col, observed=True).agg(
        customers=("ltv_12", "size"), predicted=("ltv_12", "sum"), actual=("actual_12", "sum")
    )
    g["wape"] = g.apply(
        lambda r: wape(np.array([r["predicted"]]), np.array([r["actual"]])), axis=1
    )
    return g.reset_index().to_dict(orient="records")


def evaluate(
    holdout_rows: pd.DataFrame,   # (Customer ID, t, active, net_revenue, ...features) holdout rows
    p_active_pred: pd.DataFrame,  # (Customer ID, t, value) predicted P(active)
    e_spend_pred: pd.DataFrame,   # (Customer ID, t, value) predicted E[spend|active]
    ltv_pred: pd.DataFrame,       # (Customer ID, ltv_6, ltv_12)
    feat: pd.DataFrame,           # customer features (for the segment table)
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

    # Calibration by decile (12-month value).
    dec = merged.copy()
    dec["decile"] = pd.qcut(dec["ltv_12"].rank(method="first"), 10, labels=False) + 1
    decile_table = (
        dec.groupby("decile")
        .agg(customers=("ltv_12", "size"), predicted=("ltv_12", "sum"), actual=("actual_12", "sum"))
        .reset_index()
    )
    decile_table["wape"] = decile_table.apply(
        lambda r: wape(np.array([r["predicted"]]), np.array([r["actual"]])), axis=1
    )

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
    }
    calibration = {"calibration_by_month": calibration_by_month}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (OUT_DIR / "calibration.json").write_text(json.dumps(calibration, indent=2, default=str))
    return metrics, calibration

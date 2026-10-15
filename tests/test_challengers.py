import numpy as np
import pandas as pd
import pytest

from bvr.activity_audit import audit_activity_labels
from bvr.challengers import paired_wape_delta_ci, run_validation, score, target_totals
from bvr.features import ALL_FEATURES, T0


def test_target_totals_keeps_signed_cancellations():
    rows = pd.DataFrame([
        {"Customer ID": 1, "t": month, "net_revenue": -2.0 if month == 2 else 4.0}
        for month in range(1, 13)
    ])
    table = target_totals(rows)
    assert table.loc[1, "actual_6m"] == 18.0
    assert table.loc[1, "actual_12m"] == 42.0
    with pytest.raises(ValueError):
        target_totals(rows[rows["t"] < 12])


def test_customer_error_and_portfolio_error_are_not_interchangeable():
    result = score([10, 10], [10, 10], [0, 20], [0, 20])
    assert result["wape_12m"] == 1.0
    assert result["portfolio_revenue_gap_12m"] == 0.0


def test_activity_audit_detects_purchase_offset_by_cancellation():
    dates = [T0 + pd.Timedelta(days=2), T0 + pd.Timedelta(days=2)]
    cleaned = pd.DataFrame([
        {"Customer ID": 1, "InvoiceDate": dates[0], "Invoice": "100", "net_value": 12.0},
        {"Customer ID": 1, "InvoiceDate": dates[1], "Invoice": "C100", "net_value": -12.0},
    ])
    targets = pd.DataFrame([{"Customer ID": 1, "t": 1, "net_revenue": 0.0, "active": 0}])
    result = audit_activity_labels(cleaned, targets)
    assert result["purchase_but_net_inactive_months"] == 1
    assert result["pct_activity_label_disagreement"] == 1.0


def test_challenger_validation_never_scores_heldout_customers():
    features = []
    targets = []
    for customer in range(1, 51):
        row = {"Customer ID": customer}
        for col in ALL_FEATURES:
            row[col] = "United Kingdom" if col == "country_group" else (
                "2010Q1" if col == "first_purchase_quarter" else float(customer + 1)
            )
        features.append(row)
        for t in range(1, 13):
            targets.append({"Customer ID": customer, "t": t, "net_revenue": float(customer * 2 + t)})
    holdout = set(range(41, 51))
    result = run_validation(pd.DataFrame(features), pd.DataFrame(targets), set(range(1, 41)), holdout)
    assert result["holdout_customers_accessed"] == 0
    assert result["holdout_ids_excluded"] == 10
    assert result["fit_customers"] + result["validation_customers"] == 40
    assert set(result["comparison"]) == {"repeat_spend", "direct_mae", "direct_tweedie"}
    assert np.isfinite(result["comparison"]["direct_mae"]["wape_12m"])
    assert result["release_status"].startswith("experiment_only")

def test_paired_bootstrap_keeps_same_customers_and_sign():
    actual = np.array([10., 20., 30., 40.])
    model = np.array([10., 20., 30., 40.])
    baseline = np.array([0., 0., 0., 0.])
    out = paired_wape_delta_ci(actual, model, baseline, n_resamples=200)
    assert out["delta_wape_12m"] == -1.0
    assert out["delta_wape_95pct_ci"] == pytest.approx([-1., -1.])
    assert out["p_bootstrap_challenger_better"] == 1.0


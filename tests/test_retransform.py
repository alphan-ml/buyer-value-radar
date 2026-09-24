"""Cross-fitted retransform tests (BUILD PACK Section 5 follow-up,
CONTEXT.md D14)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bvr.retransform import bin_index, expected_spend, fit_retransform

LGB_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.2,
    "num_leaves": 7,
    "min_data_in_leaf": 5,
    "verbose": -1,
    "seed": 26,
}


def _synthetic_rows(n_customers: int = 400, seed: int = 0) -> pd.DataFrame:
    """One active row per customer. `x` drives the signal; noise is
    customer-specific so a booster fit on some customers overfits their
    particular noise and under-estimates it on held-out customers -- exactly
    the situation cross-fitting is meant to correct."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_customers)
    noise = rng.normal(scale=1.5, size=n_customers)
    log_net_revenue = 3.0 + 0.8 * x + noise
    net_revenue = np.expm1(log_net_revenue).clip(min=0)
    return pd.DataFrame(
        {
            "Customer ID": np.arange(n_customers),
            "t": 1,
            "active": 1,
            "net_revenue": net_revenue,
            "x": x,
        }
    )


def test_fit_retransform_output_shapes():
    rows = _synthetic_rows()
    rt = fit_retransform(rows, ["x"], [], LGB_PARAMS, num_boost_round=20, seed=26)

    assert len(rt["edges"]) == 9
    assert len(rt["smearing"]) == 10
    assert len(rt["residual_quantiles"]) == 10
    for q in rt["residual_quantiles"]:
        assert len(q) == 201
    assert rt["n_rows"] == len(rows)
    assert rt["n_customers"] == rows["Customer ID"].nunique()
    assert rt["seed"] == 26


def test_residual_quantiles_are_non_decreasing_per_bin():
    rows = _synthetic_rows()
    rt = fit_retransform(rows, ["x"], [], LGB_PARAMS, num_boost_round=20, seed=26)
    for q in rt["residual_quantiles"]:
        assert np.all(np.diff(q) >= -1e-9)


def test_cross_fitted_smearing_exceeds_in_sample_on_noisy_data():
    rows = _synthetic_rows(n_customers=600, seed=1)
    rt = fit_retransform(rows, ["x"], [], LGB_PARAMS, num_boost_round=30, seed=26)

    # In-sample: fit once on everything, no held-out customers.
    import lightgbm as lgb

    y = np.log1p(rows["net_revenue"].clip(lower=0)).to_numpy()
    ds = lgb.Dataset(rows[["x"]], label=y)
    booster = lgb.train(LGB_PARAMS, ds, num_boost_round=30)
    pred_in_sample = booster.predict(rows[["x"]])
    smearing_in_sample = float(np.mean(np.exp(y - pred_in_sample)))

    assert rt["global_smearing"] > smearing_in_sample


def test_raises_on_bin_with_too_few_residuals():
    rows = _synthetic_rows(n_customers=20, seed=2)  # too few rows for 10 bins x 30
    with pytest.raises(ValueError):
        fit_retransform(rows, ["x"], [], LGB_PARAMS, num_boost_round=5, seed=26)


def test_bin_index_and_expected_spend():
    rt = {
        "edges": [0.0, 1.0],
        "smearing": [1.0, 2.0, 3.0],
    }
    assert list(bin_index(np.array([-1.0, 0.5, 1.5]), rt)) == [0, 1, 2]

    pred_log = np.array([-1.0, 0.5, 1.5])
    out = expected_spend(pred_log, rt)
    expected = np.array(
        [
            max(1.0 * np.exp(-1.0) - 1.0, 0.0),
            max(2.0 * np.exp(0.5) - 1.0, 0.0),
            max(3.0 * np.exp(1.5) - 1.0, 0.0),
        ]
    )
    np.testing.assert_allclose(out, expected)


def test_expected_spend_is_clipped_at_zero():
    rt = {"edges": [0.0], "smearing": [0.01, 0.01]}
    out = expected_spend(np.array([-10.0]), rt)
    assert out[0] == 0.0

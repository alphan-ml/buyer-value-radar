"""Predictive P10/P50/P90 summary tests (BUILD PACK Section 5 follow-up,
CONTEXT.md D15)."""
from __future__ import annotations

import numpy as np
import pytest

from bvr.predictive import summarize
from bvr.retransform import expected_spend

RT = {
    "edges": [0.0],
    "smearing": [1.2, 1.5],
    "residual_quantiles": [
        np.linspace(-0.5, 0.5, 201).tolist(),
        np.linspace(-0.3, 0.3, 201).tolist(),
    ],
}


def _flat(value: float, n: int = 12) -> np.ndarray:
    return np.full(n, value)


def test_deterministic():
    p_active = np.linspace(0.1, 0.9, 12)
    pred_log = np.linspace(2.0, 4.0, 12)
    out1 = summarize(p_active, pred_log, RT, seed=26)
    out2 = summarize(p_active, pred_log, RT, seed=26)
    assert out1 == out2


def test_p10_le_p50_le_p90():
    p_active = np.linspace(0.1, 0.9, 12)
    pred_log = np.linspace(2.0, 4.0, 12)
    out = summarize(p_active, pred_log, RT, seed=26)
    for h in (6, 12):
        assert out[h]["p10"] <= out[h]["p50"] <= out[h]["p90"]


def test_mean_equals_exact_sum():
    p_active = np.linspace(0.1, 0.9, 12)
    pred_log = np.linspace(2.0, 4.0, 12)
    out = summarize(p_active, pred_log, RT, seed=26)
    exp_spend = expected_spend(pred_log, RT)
    for h in (6, 12):
        expected_mean = float(np.sum(p_active[:h] * exp_spend[:h]))
        assert out[h]["mean"] == expected_mean


def test_p_active_zero_gives_all_zeros():
    p_active = _flat(0.0)
    pred_log = _flat(3.0)
    out = summarize(p_active, pred_log, RT, seed=26)
    for h in (6, 12):
        assert out[h]["mean"] == 0.0
        assert out[h]["p10"] == 0.0
        assert out[h]["p50"] == 0.0
        assert out[h]["p90"] == 0.0


def test_p_active_one_zero_residuals_gives_exact_smearing_curve():
    rt = {"edges": [], "smearing": [1.0], "residual_quantiles": [np.zeros(201).tolist()]}
    p_active = _flat(1.0)
    lp = 2.3
    pred_log = _flat(lp)
    out = summarize(p_active, pred_log, rt, seed=26)
    expected_12 = 12 * (np.exp(lp) - 1.0)
    assert out[12]["mean"] == pytest.approx(expected_12)
    assert out[12]["p10"] == pytest.approx(expected_12)
    assert out[12]["p50"] == pytest.approx(expected_12)
    assert out[12]["p90"] == pytest.approx(expected_12)

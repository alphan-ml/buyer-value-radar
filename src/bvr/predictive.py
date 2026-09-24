"""Predictive P10/P50/P90 spend range for one customer (BUILD PACK Section 5
follow-up, CONTEXT.md D15). WAPE is minimised by the median, not the mean, so
the point forecast returned for a single customer is P50; `mean` (the exact
expectation, not a simulated one) is kept for totals.

Assumption: months are drawn independently given the customer's features
(t is itself a feature) -- this does not model any month-to-month
correlation in a given customer's own shocks.
"""
from __future__ import annotations

import numpy as np

from bvr.retransform import bin_index, expected_spend

N_RESIDUAL_QUANTILES = 201
_QUANTILE_GRID = np.linspace(0, 1, N_RESIDUAL_QUANTILES)


def summarize(
    p_active: np.ndarray,
    pred_log: np.ndarray,
    rt: dict,
    horizons: tuple[int, ...] = (6, 12),
    n_draws: int = 4000,
    seed: int = 26,
) -> dict:
    p_active = np.asarray(p_active, dtype=float)
    pred_log = np.asarray(pred_log, dtype=float)

    rng = np.random.default_rng(seed)
    u_active = rng.random((12, n_draws))
    u_resid = rng.random((12, n_draws))

    bin_idx = bin_index(pred_log, rt)
    eps = np.empty((12, n_draws))
    for m in range(12):
        eps[m] = np.interp(u_resid[m], _QUANTILE_GRID, rt["residual_quantiles"][bin_idx[m]])

    draws = np.clip(np.exp(pred_log[:, None] + eps) - 1.0, a_min=0.0, a_max=None)
    draws = draws * (u_active < p_active[:, None])

    exp_spend = expected_spend(pred_log, rt)

    out = {}
    for h in horizons:
        total_h = draws[:h].sum(axis=0)
        p10, p50, p90 = np.percentile(total_h, [10, 50, 90])
        mean_h = float(np.sum(p_active[:h] * exp_spend[:h]))
        out[h] = {"mean": mean_h, "p10": float(p10), "p50": float(p50), "p90": float(p90)}
    return out

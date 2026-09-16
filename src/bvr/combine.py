"""Combine Curve 1 and Curve 2 into LTV_h (BUILD PACK Section 5).

LTV_h = sum_{t=1..h} P(active_t) * E[spend_t | active]  for h in {6, 12}.
"""
from __future__ import annotations

import pandas as pd


def combine(p_active: pd.DataFrame, e_spend: pd.DataFrame, horizons=(6, 12)):
    """p_active, e_spend: DataFrames with columns Customer ID, t, value.
    Returns (ltv_df, per_t_contribution_df)."""
    merged = p_active.merge(e_spend, on=["Customer ID", "t"], suffixes=("_p", "_spend"))
    merged["contribution"] = merged["value_p"] * merged["value_spend"]
    per_t = merged.pivot(index="Customer ID", columns="t", values="contribution").sort_index(axis=1)

    result = pd.DataFrame(index=per_t.index)
    for h in horizons:
        cols = [c for c in per_t.columns if c <= h]
        result[f"ltv_{h}"] = per_t[cols].sum(axis=1)
    return result.reset_index(), per_t.reset_index()


if __name__ == "__main__":
    # Hand-computed 3-customer, 3-month toy -- same fixture used by
    # tests/test_combine.py.
    p_active = pd.DataFrame({
        "Customer ID": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "t":           [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "value":       [1.0, 0.5, 0.0, 0.2, 0.2, 0.2, 0.9, 0.9, 0.9],
    })
    e_spend = pd.DataFrame({
        "Customer ID": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "t":           [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "value":       [100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 10.0, 10.0, 10.0],
    })
    result, _ = combine(p_active, e_spend, horizons=(2, 3))
    print(result)
    # By hand:
    # customer 1: ltv_2 = 1.0*100 + 0.5*100 = 150; ltv_3 = 150 + 0.0*100 = 150
    # customer 2: ltv_2 = 0.2*50 + 0.2*50 = 20;  ltv_3 = 20 + 0.2*50 = 30
    # customer 3: ltv_2 = 0.9*10 + 0.9*10 = 18;  ltv_3 = 18 + 0.9*10 = 27

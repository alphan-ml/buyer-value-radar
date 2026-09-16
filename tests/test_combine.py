"""combine() unit test: hand-computed 3-customer, 3-month toy
(BUILD PACK Section 5 / Section 7)."""
from __future__ import annotations

import pandas as pd
import pytest

from bvr.combine import combine


@pytest.fixture
def toy():
    p_active = pd.DataFrame(
        {
            "Customer ID": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "t": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "value": [1.0, 0.5, 0.0, 0.2, 0.2, 0.2, 0.9, 0.9, 0.9],
        }
    )
    e_spend = pd.DataFrame(
        {
            "Customer ID": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "t": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "value": [100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 10.0, 10.0, 10.0],
        }
    )
    return p_active, e_spend


def test_combine_hand_computed(toy):
    p_active, e_spend = toy
    result, _ = combine(p_active, e_spend, horizons=(2, 3))
    result = result.set_index("Customer ID")

    assert result.loc[1, "ltv_2"] == pytest.approx(150.0)   # 1.0*100 + 0.5*100
    assert result.loc[1, "ltv_3"] == pytest.approx(150.0)   # + 0.0*100
    assert result.loc[2, "ltv_2"] == pytest.approx(20.0)    # 0.2*50 + 0.2*50
    assert result.loc[2, "ltv_3"] == pytest.approx(30.0)    # + 0.2*50
    assert result.loc[3, "ltv_2"] == pytest.approx(18.0)    # 0.9*10 + 0.9*10
    assert result.loc[3, "ltv_3"] == pytest.approx(27.0)    # + 0.9*10

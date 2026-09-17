"""_seg_table unit test (BUILD PACK Section 7): a 3-customer toy group where
per-customer over- and under-estimates cancel in the group total. This must
NOT make the group look error-free -- `wape` (customer-level) should catch
it even though `aggregate_revenue_error` (net, on the group's totals) does
not."""
from __future__ import annotations

import pandas as pd
import pytest

from bvr.eval import _seg_table


@pytest.fixture
def toy():
    # Customer 1: predicted 100, actual 80 -> +20 over.
    # Customer 2: predicted 50, actual 70 -> -20 under.
    # Customer 3: predicted 90, actual 90 -> exact.
    # Group totals: predicted 240 == actual 240 -> net error is zero,
    # even though two of the three customers were mispredicted.
    return pd.DataFrame(
        {
            "ltv_12": [100.0, 50.0, 90.0],
            "actual_12": [80.0, 70.0, 90.0],
            "seg": ["A", "A", "A"],
        }
    )


def test_seg_table_reports_customer_level_wape_when_errors_cancel(toy):
    rows = _seg_table(toy, "seg")
    assert len(rows) == 1
    row = rows[0]

    assert row["customers"] == 3
    assert row["aggregate_revenue_error"] == pytest.approx(0.0)
    # (|100-80| + |50-70| + |90-90|) / (80+70+90) = 40 / 240
    assert row["wape"] == pytest.approx(40.0 / 240.0)
    assert row["wape"] > 0.0

"""Baseline unit tests on small synthetic frames (BUILD PACK follow-up,
CONTEXT.md D10/D11)."""
from __future__ import annotations

import pandas as pd
import pytest

from bvr.eval import repeat_baseline, segment_mean_baseline


def test_repeat_baseline_zero_prior_spend_is_zero():
    feat = pd.DataFrame({"Customer ID": [1, 2], "monetary_total": [0.0, 200.0]})
    out = repeat_baseline(feat).set_index("Customer ID")

    assert out.loc[1, "pred_12"] == 0.0
    assert out.loc[1, "pred_6"] == 0.0
    assert out.loc[2, "pred_12"] == pytest.approx(200.0)
    assert out.loc[2, "pred_6"] == pytest.approx(100.0)


def test_segment_mean_uses_training_rows_only():
    # 6 training customers, 2 per tercile of monetary_total; 2 holdout
    # customers whose actual spend must NOT move any customer's prediction.
    feat = pd.DataFrame({
        "Customer ID": [1, 2, 3, 4, 5, 6, 7, 8],
        "monetary_total": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 25.0, 55.0],
    })
    train_ids = {1, 2, 3, 4, 5, 6}
    actual_6 = pd.Series({i: 1.0 for i in range(1, 9)})

    actual_12_a = pd.Series(
        {1: 100.0, 2: 100.0, 3: 200.0, 4: 200.0, 5: 300.0, 6: 300.0, 7: 10.0, 8: 10.0}
    )
    actual_12_b = actual_12_a.copy()
    actual_12_b[7] = 999_999.0  # holdout customer's actual changes...
    actual_12_b[8] = -999_999.0  # ...and again, wildly.

    out_a = segment_mean_baseline(feat, train_ids, actual_6, actual_12_a).set_index("Customer ID")
    out_b = segment_mean_baseline(feat, train_ids, actual_6, actual_12_b).set_index("Customer ID")

    # Predictions for every customer (train AND holdout) are unaffected by
    # the holdout customers' own actual values -- only training rows feed
    # the segment mean.
    pd.testing.assert_frame_equal(out_a, out_b)

    # And the segment mean is a real training-tercile average, not a
    # placeholder: low tercile (customers 1, 2) trains to 100, high tercile
    # (customers 5, 6) trains to 300.
    assert out_a.loc[1, "pred_12"] == pytest.approx(100.0)
    assert out_a.loc[2, "pred_12"] == pytest.approx(100.0)
    assert out_a.loc[5, "pred_12"] == pytest.approx(300.0)
    assert out_a.loc[6, "pred_12"] == pytest.approx(300.0)

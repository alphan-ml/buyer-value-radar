"""No-future-leak test (BUILD PACK Section 5: 'no information after T0 may
be used -- write one test that proves it')."""
from __future__ import annotations

import pandas as pd

from bvr.clean import clean
from bvr.features import T0, build_features, build_targets


def _rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["Invoice"] = df["Invoice"].astype("string")
    df["StockCode"] = df["StockCode"].astype("string")
    df["Country"] = "United Kingdom"
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Customer ID"] = df["Customer ID"].astype("Int64")
    return df


def test_features_ignore_rows_after_t0():
    """A customer's post-T0 purchases must not move their pre-T0 features."""
    before = _rows(
        [
            {"Invoice": "500001", "StockCode": "A1", "Quantity": 1, "Price": 10.0,
             "InvoiceDate": "2010-06-01", "Customer ID": 1},
        ]
    )
    after_only = pd.concat(
        [
            before,
            _rows(
                [
                    {"Invoice": "500002", "StockCode": "A1", "Quantity": 100,
                     "Price": 500.0, "InvoiceDate": "2011-06-01", "Customer ID": 1},
                ]
            ),
        ],
        ignore_index=True,
    )

    cleaned_before, _ = clean(before)
    cleaned_after, _ = clean(after_only)

    feat_before = build_features(cleaned_before).set_index("Customer ID")
    feat_after = build_features(cleaned_after).set_index("Customer ID")

    # Adding a huge purchase after T0 must not change any pre-T0 feature.
    pd.testing.assert_frame_equal(feat_before.loc[[1]], feat_after.loc[[1]])


def test_targets_ignore_rows_at_or_before_t0():
    """Target rows must only reflect strictly-after-T0 activity."""
    df = _rows(
        [
            {"Invoice": "500001", "StockCode": "A1", "Quantity": 1, "Price": 999.0,
             "InvoiceDate": T0, "Customer ID": 1},  # exactly at T0: must not count
            {"Invoice": "500002", "StockCode": "A1", "Quantity": 1, "Price": 5.0,
             "InvoiceDate": "2010-12-10", "Customer ID": 1},  # day after T0: t=1
        ]
    )
    cleaned, _ = clean(df)
    targets = build_targets(cleaned, [1])
    t1 = targets[(targets["Customer ID"] == 1) & (targets["t"] == 1)].iloc[0]
    assert t1["net_revenue"] == 5.0

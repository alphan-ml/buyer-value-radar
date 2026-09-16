"""Cleaning-rule tests (BUILD PACK Section 4 / Section 7)."""
from __future__ import annotations

import pandas as pd

from bvr.clean import clean, customer_month_net_revenue


def _rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["Invoice"] = df["Invoice"].astype("string")
    df["StockCode"] = df["StockCode"].astype("string")
    df["Country"] = "United Kingdom"
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Customer ID"] = df["Customer ID"].astype("Int64")
    return df


def test_cancellation_lines_are_kept():
    df = _rows(
        [
            {"Invoice": "500001", "StockCode": "A1", "Quantity": 2, "Price": 10.0,
             "InvoiceDate": "2010-01-01", "Customer ID": 1},
            {"Invoice": "C500002", "StockCode": "A1", "Quantity": -2, "Price": 10.0,
             "InvoiceDate": "2010-01-02", "Customer ID": 1},
        ]
    )
    cleaned, stats = clean(df)
    assert stats["rows_kept"] == 2
    assert stats["kept_cancellation_lines"] == 1
    assert cleaned["is_cancellation"].tolist() == [False, True]


def test_bad_qty_or_price_noncancel_dropped():
    df = _rows(
        [
            {"Invoice": "500001", "StockCode": "A1", "Quantity": 0, "Price": 10.0,
             "InvoiceDate": "2010-01-01", "Customer ID": 1},
            {"Invoice": "500002", "StockCode": "A1", "Quantity": 2, "Price": 0.0,
             "InvoiceDate": "2010-01-01", "Customer ID": 1},
            {"Invoice": "500003", "StockCode": "A1", "Quantity": 2, "Price": 10.0,
             "InvoiceDate": "2010-01-01", "Customer ID": 1},
        ]
    )
    _cleaned, stats = clean(df)
    assert stats["dropped_bad_qty_or_price_noncancel"] == 2
    assert stats["rows_kept"] == 1


def test_non_product_stockcode_dropped():
    df = _rows(
        [
            {"Invoice": "500001", "StockCode": "POST", "Quantity": 1, "Price": 18.0,
             "InvoiceDate": "2010-01-01", "Customer ID": 1},
            {"Invoice": "500002", "StockCode": "A1", "Quantity": 1, "Price": 18.0,
             "InvoiceDate": "2010-01-01", "Customer ID": 1},
        ]
    )
    _cleaned, stats = clean(df)
    assert stats["dropped_non_product_stockcode"] == 1
    assert stats["rows_kept"] == 1


def test_no_customer_id_excluded():
    df = _rows(
        [
            {"Invoice": "500001", "StockCode": "A1", "Quantity": 1, "Price": 18.0,
             "InvoiceDate": "2010-01-01", "Customer ID": pd.NA},
            {"Invoice": "500002", "StockCode": "A1", "Quantity": 1, "Price": 18.0,
             "InvoiceDate": "2010-01-01", "Customer ID": 1},
        ]
    )
    _cleaned, stats = clean(df)
    assert stats["dropped_no_customer_id"] == 1
    assert stats["rows_kept"] == 1


def test_customer_month_net_revenue_active_flag():
    df = _rows(
        [
            {"Invoice": "500001", "StockCode": "A1", "Quantity": 2, "Price": 10.0,
             "InvoiceDate": "2010-01-05", "Customer ID": 1},
            {"Invoice": "C500002", "StockCode": "A1", "Quantity": -2, "Price": 10.0,
             "InvoiceDate": "2010-01-10", "Customer ID": 1},
        ]
    )
    cleaned, _ = clean(df)
    cm = customer_month_net_revenue(cleaned)
    assert len(cm) == 1
    row = cm.iloc[0]
    assert row["net_revenue"] == 0.0
    assert bool(row["active"]) is False  # net <= 0 is not active

"""Schema tests (BUILD PACK Section 7)."""
from __future__ import annotations

import pandas as pd
import pytest

from bvr import schema


def _good_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Invoice": pd.array(["536365", "536366"], dtype="string"),
            "StockCode": pd.array(["85123A", "POST"], dtype="string"),
            "Description": pd.array(["WHITE HANGING HEART", "POSTAGE"], dtype="string"),
            "Quantity": pd.array([6, 1], dtype="int64"),
            "InvoiceDate": pd.to_datetime(["2010-01-01", "2010-01-02"]),
            "Price": pd.array([2.55, 18.0], dtype="float64"),
            "Customer ID": pd.array([17850, pd.NA], dtype="Int64"),
            "Country": pd.array(["United Kingdom", "United Kingdom"], dtype="string"),
        }
    )


def test_valid_frame_passes():
    schema.validate_raw(_good_df())


def test_missing_column_fails():
    df = _good_df().drop(columns=["Price"])
    with pytest.raises(schema.SchemaError):
        schema.validate_raw(df)


def test_required_null_fails():
    df = _good_df()
    df.loc[0, "Invoice"] = pd.NA
    with pytest.raises(schema.SchemaError):
        schema.validate_raw(df)


def test_quantity_out_of_bounds_fails():
    df = _good_df()
    df.loc[0, "Quantity"] = schema.QUANTITY_ABS_MAX + 1
    with pytest.raises(schema.SchemaError):
        schema.validate_raw(df)


def test_price_out_of_bounds_fails():
    df = _good_df()
    df.loc[0, "Price"] = -(schema.PRICE_ABS_MAX + 1)
    with pytest.raises(schema.SchemaError):
        schema.validate_raw(df)


def test_date_out_of_bounds_fails():
    df = _good_df()
    df.loc[0, "InvoiceDate"] = pd.Timestamp("2020-01-01")
    with pytest.raises(schema.SchemaError):
        schema.validate_raw(df)

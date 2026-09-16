"""Column schema and allowed ranges for the raw Online Retail II data
(BUILD PACK Section 7: 'bvr check' fails loudly on violation)."""
from __future__ import annotations

import pandas as pd

RAW_DTYPES = {
    "Invoice": "string",
    "StockCode": "string",
    "Description": "string",
    "Quantity": "int64",
    "InvoiceDate": "datetime64[ns]",
    "Price": "float64",
    "Customer ID": "Int64",
    "Country": "string",
}

REQUIRED_NON_NULL = ["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Country"]
# Customer ID and Description may be null (known data-quality gaps, Section 3).

# Sanity ranges from the real, verified fetch (Sep 13, 2026) -- wide bounds,
# meant to catch a corrupted or truncated download, not to filter real rows.
# Verified real extremes in the raw file: Quantity -80,995 .. 80,995;
# Price -53,594.36 (Invoice A506401, "Adjust bad debt" bookkeeping line,
# dropped as a non-product StockCode in clean.py) .. 38,970.00.
QUANTITY_ABS_MAX = 100_000
PRICE_ABS_MAX = 60_000
DATE_MIN = pd.Timestamp("2009-12-01")
DATE_MAX = pd.Timestamp("2011-12-31")


class SchemaError(AssertionError):
    pass


def validate_raw(df: pd.DataFrame) -> None:
    missing_cols = [c for c in RAW_DTYPES if c not in df.columns]
    if missing_cols:
        raise SchemaError(f"missing columns: {missing_cols}")

    for col in REQUIRED_NON_NULL:
        n_null = int(df[col].isna().sum())
        if n_null:
            raise SchemaError(f"required column '{col}' has {n_null} null values")

    if (df["Quantity"].abs() > QUANTITY_ABS_MAX).any():
        raise SchemaError(f"Quantity exceeds sanity bound of {QUANTITY_ABS_MAX}")
    if (df["Price"].abs() > PRICE_ABS_MAX).any():
        raise SchemaError(f"Price exceeds sanity bound of {PRICE_ABS_MAX}")
    if df["InvoiceDate"].min() < DATE_MIN or df["InvoiceDate"].max() > DATE_MAX:
        raise SchemaError(
            f"InvoiceDate out of expected range: "
            f"{df['InvoiceDate'].min()} .. {df['InvoiceDate'].max()}"
        )

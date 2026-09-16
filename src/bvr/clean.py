"""Cleaning rules for Online Retail II (BUILD PACK Section 4).

Rules (exact, from the pack):
  - Cancellation lines (Invoice starts with "C"): KEPT. They net against the
    original order at the customer-month level.
  - Non-cancellation lines with Quantity <= 0 or Price <= 0: DROPPED.
  - StockCode lines that are not products (postage, bank charges, manual
    adjustments, test rows, and similar all-letter codes): DROPPED.
  - Lines with no Customer ID: EXCLUDED from the model (kept in the overall
    business view in check.py, but customer-level modeling needs a customer).
  - customer-month net revenue = sum(Quantity * Price) over kept lines,
    cancellations included; a month with net <= 0 counts as "not active".

Everything here operates on the real parquet from fetch.py. No sampling.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bvr.check import NON_PRODUCT_CODES

ROOT = Path(__file__).resolve().parents[2]
PARQUET_PATH = ROOT / "data" / "raw" / "online_retail_ii.parquet"


def load_raw() -> pd.DataFrame:
    return pd.read_parquet(PARQUET_PATH)


def clean(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (cleaned_df, stats). cleaned_df is customer-attributed, kept lines only."""
    if df is None:
        df = load_raw()

    n_total = len(df)
    is_cancel = df["Invoice"].str.startswith("C", na=False)
    codes_upper = df["StockCode"].astype("string").str.upper()
    is_non_product = codes_upper.isin({c.upper() for c in NON_PRODUCT_CODES})
    has_customer = df["Customer ID"].notna()

    drop_bad_qty_price = (~is_cancel) & ((df["Quantity"] <= 0) | (df["Price"] <= 0))

    keep_mask = (~drop_bad_qty_price) & (~is_non_product) & has_customer
    cleaned = df.loc[keep_mask].copy()
    cleaned["is_cancellation"] = is_cancel.loc[keep_mask]
    cleaned["net_value"] = cleaned["Quantity"] * cleaned["Price"]

    stats = {
        "rows_in": n_total,
        "dropped_no_customer_id": int((~has_customer).sum()),
        "dropped_bad_qty_or_price_noncancel": int(drop_bad_qty_price.sum()),
        "dropped_non_product_stockcode": int((is_non_product & has_customer & ~drop_bad_qty_price).sum()),
        "rows_kept": len(cleaned),
        "kept_cancellation_lines": int(cleaned["is_cancellation"].sum()),
    }
    return cleaned, stats


def customer_month_net_revenue(cleaned: pd.DataFrame) -> pd.DataFrame:
    """One row per (Customer ID, calendar month): net revenue and active flag."""
    df = cleaned.copy()
    df["month"] = df["InvoiceDate"].dt.to_period("M")
    grouped = (
        df.groupby(["Customer ID", "month"])["net_value"]
        .sum()
        .reset_index()
        .rename(columns={"net_value": "net_revenue"})
    )
    grouped["active"] = grouped["net_revenue"] > 0
    return grouped


if __name__ == "__main__":
    cleaned, stats = clean()
    print(json.dumps(stats, indent=2))
    cm = customer_month_net_revenue(cleaned)
    print(f"customer-month rows: {len(cm):,}")
    print(f"customers represented: {cm['Customer ID'].nunique():,}")
    print(f"active customer-months: {int(cm['active'].sum()):,} of {len(cm):,} "
          f"({100 * cm['active'].mean():.2f}%)")

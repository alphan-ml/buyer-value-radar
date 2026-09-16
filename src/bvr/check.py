"""Day-1 data checks for Online Retail II (BUILD PACK Section 3).

Prints, and writes to outputs/data_quality.json, exactly the facts the
build-deploy pack asks for before any model training happens. Every number
here comes from the real parquet written by fetch.py -- nothing is typed
by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from bvr import schema

ROOT = Path(__file__).resolve().parents[2]
PARQUET_PATH = ROOT / "data" / "raw" / "online_retail_ii.parquet"
OUT_PATH = ROOT / "outputs" / "data_quality.json"

# StockCode values that are not products (postage, bank charges, manual
# adjustments, discounts, etc.) -- Section 4 of the pack: drop these before
# feature building, but they are counted here first.
NON_PRODUCT_CODES = {
    "POST", "D", "M", "BANK CHARGES", "PADS", "DOT", "CRUK", "C2", "AMAZONFEE",
    "ADJUST", "ADJUST2", "TEST001", "TEST002", "S", "m", "B",
    # "B" = "Adjust bad debt" -- a bookkeeping write-off line (Invoice
    # A506401, Price -53,594.36, real value verified Sep 13, 2026), not a
    # cancellation (its Invoice does not start with "C") and not a product.
}


def check() -> dict:
    df = pd.read_parquet(PARQUET_PATH)
    schema.validate_raw(df)
    n_rows = len(df)

    date_min = df["InvoiceDate"].min()
    date_max = df["InvoiceDate"].max()

    has_customer = df["Customer ID"].notna()
    n_with_customer = int(has_customer.sum())
    n_no_customer = int((~has_customer).sum())
    pct_no_customer = round(100 * n_no_customer / n_rows, 3)
    n_unique_customers = int(df.loc[has_customer, "Customer ID"].nunique())

    is_cancellation = df["Invoice"].str.startswith("C", na=False)
    n_cancel = int(is_cancellation.sum())
    pct_cancel = round(100 * n_cancel / n_rows, 3)

    non_cancel = ~is_cancellation
    n_qty_le0 = int((non_cancel & (df["Quantity"] <= 0)).sum())
    n_price_le0 = int((non_cancel & (df["Price"] <= 0)).sum())

    codes_upper = df["StockCode"].astype("string").str.upper()
    is_non_product = codes_upper.isin({c.upper() for c in NON_PRODUCT_CODES})
    non_product_counts = (
        codes_upper[is_non_product].value_counts().to_dict()
    )
    n_non_product_rows = int(is_non_product.sum())

    # Repeat-buyer group: customers with more than one distinct invoice date
    # (using the calendar date, not the timestamp, per the pack's definition
    # of "more than one invoice date").
    cust_df = df.loc[has_customer, ["Customer ID", "Invoice", "InvoiceDate"]].copy()
    cust_df["invoice_day"] = cust_df["InvoiceDate"].dt.date
    days_per_customer = cust_df.groupby("Customer ID")["invoice_day"].nunique()
    n_repeat_customers = int((days_per_customer > 1).sum())
    pct_repeat_customers = round(100 * n_repeat_customers / n_unique_customers, 3)

    # Net revenue by month = sum(Quantity * Price) over KEPT lines, cancellation
    # lines included (they net against the original order), non-cancellation
    # lines with Quantity<=0 or Price<=0 excluded (Section 4 cleaning rule).
    # This check reports the raw monthly net revenue over all rows (pre any
    # exclusion) so Leon can see the shape before cleaning is applied; the
    # cleaning-rule-applied version is recomputed identically in clean.py.
    kept_mask = is_cancellation | (df["Quantity"] > 0) & (df["Price"] > 0)
    df["_line_value"] = df["Quantity"] * df["Price"]
    monthly = (
        df.loc[kept_mask]
        .assign(month=df.loc[kept_mask, "InvoiceDate"].dt.to_period("M").astype(str))
        .groupby("month")["_line_value"]
        .sum()
        .round(2)
    )
    monthly_table = [{"month": m, "net_revenue_gbp": float(v)} for m, v in monthly.items()]

    result = {
        "row_count": n_rows,
        "date_min": str(date_min),
        "date_max": str(date_max),
        "unique_customers": n_unique_customers,
        "rows_with_customer_id": n_with_customer,
        "rows_without_customer_id": n_no_customer,
        "pct_rows_without_customer_id": pct_no_customer,
        "cancellation_lines": n_cancel,
        "pct_cancellation_lines": pct_cancel,
        "non_cancellation_qty_le_0": n_qty_le0,
        "non_cancellation_price_le_0": n_price_le0,
        "non_product_stockcode_rows": n_non_product_rows,
        "non_product_stockcode_counts": non_product_counts,
        "repeat_buyer_customers": n_repeat_customers,
        "pct_repeat_buyer_customers": pct_repeat_customers,
        "monthly_net_revenue_gbp_kept_lines": monthly_table,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    r = check()
    print(f"row_count: {r['row_count']:,}")
    print(f"date range: {r['date_min']} .. {r['date_max']}")
    print(f"unique_customers: {r['unique_customers']:,}")
    print(f"rows_without_customer_id: {r['rows_without_customer_id']:,} "
          f"({r['pct_rows_without_customer_id']}%)")
    print(f"cancellation_lines: {r['cancellation_lines']:,} "
          f"({r['pct_cancellation_lines']}%)")
    print(f"non_cancellation Quantity<=0: {r['non_cancellation_qty_le_0']:,}")
    print(f"non_cancellation Price<=0: {r['non_cancellation_price_le_0']:,}")
    print(f"non_product StockCode rows: {r['non_product_stockcode_rows']:,} "
          f"{r['non_product_stockcode_counts']}")
    print(f"repeat_buyer_customers: {r['repeat_buyer_customers']:,} "
          f"({r['pct_repeat_buyer_customers']}% of unique customers)")
    print(f"monthly net revenue rows: {len(r['monthly_net_revenue_gbp_kept_lines'])}")
    for row in r["monthly_net_revenue_gbp_kept_lines"]:
        print(f"  {row['month']}: {row['net_revenue_gbp']:,.2f}")
    print(f"\nwrote {OUT_PATH}", file=sys.stderr)

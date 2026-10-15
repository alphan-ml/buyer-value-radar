"""Audit customer activity labels without modifying the released two-curve model.

Net positive revenue is not the same event as making a purchase: returns and
cancellations can offset positive invoices in a customer-month.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bvr.features import T0, TARGET_MONTHS, _target_bucket

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"


def audit_activity_labels(cleaned: pd.DataFrame, monthly_targets: pd.DataFrame) -> dict:
    needed = {"Customer ID", "InvoiceDate", "Invoice", "net_value"}
    if not needed.issubset(cleaned.columns):
        raise ValueError("cleaned transactions missing audit columns")
    target_cols = {"Customer ID", "t", "net_revenue", "active"}
    if not target_cols.issubset(monthly_targets.columns):
        raise ValueError("targets missing activity/return columns")

    window_end = T0 + pd.DateOffset(months=TARGET_MONTHS)
    customer_ids = set(monthly_targets["Customer ID"])
    future = cleaned.loc[
        cleaned["Customer ID"].isin(customer_ids)
        & cleaned["InvoiceDate"].gt(T0)
        & cleaned["InvoiceDate"].le(window_end)
    ].copy()
    future["t"] = _target_bucket(future["InvoiceDate"]).astype("Int64")
    future = future.loc[future["t"].notna()]
    future["purchase_line"] = (
        future["net_value"].gt(0)
        & ~future["Invoice"].astype("string").str.startswith("C", na=False)
    )
    purchases = future.groupby(["Customer ID", "t"], observed=True)["purchase_line"].any()
    indexed = monthly_targets.set_index(["Customer ID", "t"])
    purchase_active = purchases.reindex(indexed.index, fill_value=False).astype(bool)
    net_active = indexed["active"].astype(bool)

    cancelled = purchase_active & ~net_active
    return {
        "customer_months": int(len(indexed)),
        "purchase_active_months": int(purchase_active.sum()),
        "net_revenue_active_months": int(net_active.sum()),
        "purchase_but_net_inactive_months": int(cancelled.sum()),
        "net_active_without_positive_purchase_months": int((net_active & ~purchase_active).sum()),
        "pct_activity_label_disagreement": float((purchase_active != net_active).mean()),
        "interpretation": (
            "A purchase-based activity definition differs from positive-net-revenue "
            "activity when returns/cancellations offset purchases. This is an audit, "
            "not a change to the previously reported model or holdout metrics."
        ),
    }


def main() -> None:
    from bvr import clean

    targets_path = OUT_DIR / "targets_monthly.parquet"
    if not targets_path.exists():
        raise SystemExit("Run 'python -m bvr.cli features' first")
    cleaned, _ = clean.clean(clean.load_raw())
    result = audit_activity_labels(cleaned, pd.read_parquet(targets_path))
    out = OUT_DIR / "activity_label_audit.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(out), "label_disagreement": result["pct_activity_label_disagreement"]}))


if __name__ == "__main__":
    main()

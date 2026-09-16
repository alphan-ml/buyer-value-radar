"""Feature engineering for Buyer Value Radar (BUILD PACK Section 5).

All features are computed strictly from cleaned transaction lines with
InvoiceDate <= T0 (Dec 9, 2010, end of day). No information from after T0
enters any feature -- see tests/test_no_leak.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"

T0 = pd.Timestamp("2010-12-09 23:59:59")
FEATURE_WINDOW_START = pd.Timestamp("2009-12-01 00:00:00")
TARGET_MONTHS = 12  # t = 1..12, each a calendar-month-shifted bucket after T0
WHOLESALE_QTY_THRESHOLD = 100

EUROPE_COUNTRIES = {
    "EIRE", "Germany", "France", "Netherlands", "Belgium", "Spain",
    "Switzerland", "Portugal", "Italy", "Austria", "Sweden", "Denmark",
    "Poland", "Norway", "Finland", "Channel Islands", "Cyprus", "Greece",
    "Iceland", "Malta", "European Community", "Lithuania", "Czech Republic",
}

CATEGORICAL_FEATURES = ["country_group", "first_purchase_quarter"]
NUMERIC_FEATURES = [
    "recency_days", "frequency", "tenure_days", "monetary_total",
    "monetary_mean_per_invoice", "monetary_last_3m", "active_months_share",
    "mean_line_quantity", "share_invoices_ge_100_units",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def country_group(country: str) -> str:
    if country == "United Kingdom":
        return "United Kingdom"
    if country in EUROPE_COUNTRIES:
        return "Europe"
    return "Rest of world"


def build_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """One row per customer with >=1 kept purchase in the feature window."""
    fw = cleaned[
        (cleaned["InvoiceDate"] >= FEATURE_WINDOW_START) & (cleaned["InvoiceDate"] <= T0)
    ].copy()
    fw = fw[fw["Customer ID"].notna()]

    customers = fw["Customer ID"].unique()

    fw["invoice_day"] = fw["InvoiceDate"].dt.date
    fw["month_period"] = fw["InvoiceDate"].dt.to_period("M")
    fw["is_wholesale_line"] = fw["Quantity"] >= WHOLESALE_QTY_THRESHOLD

    last_purchase = fw.groupby("Customer ID")["InvoiceDate"].max()
    first_purchase = fw.groupby("Customer ID")["InvoiceDate"].min()
    frequency = fw.groupby("Customer ID")["invoice_day"].nunique()
    net_total = fw.groupby("Customer ID")["net_value"].sum()
    months_active = fw.groupby("Customer ID")["month_period"].nunique()
    mean_qty = fw.groupby("Customer ID")["Quantity"].mean()

    invoice_net = fw.groupby(["Customer ID", "Invoice"])["net_value"].sum()
    mean_per_invoice = invoice_net.groupby("Customer ID").mean()

    invoice_has_wholesale = fw.groupby(["Customer ID", "Invoice"])["is_wholesale_line"].any()
    share_wholesale_invoices = invoice_has_wholesale.groupby("Customer ID").mean()

    # last-3-month net revenue: the 91 days immediately before and including T0.
    last3_start = T0 - pd.Timedelta(days=90)
    last3 = fw[fw["InvoiceDate"] >= last3_start].groupby("Customer ID")["net_value"].sum()

    country_mode = fw.groupby("Customer ID")["Country"].agg(lambda s: s.value_counts().idxmax())

    feat = pd.DataFrame(index=pd.Index(customers, name="Customer ID"))
    feat["recency_days"] = (T0 - last_purchase).dt.days
    feat["frequency"] = frequency
    feat["tenure_days"] = (T0 - first_purchase).dt.days
    feat["monetary_total"] = net_total
    feat["monetary_mean_per_invoice"] = mean_per_invoice
    feat["monetary_last_3m"] = last3.reindex(feat.index).fillna(0.0)

    months_since_first = (
        (T0.year - first_purchase.dt.year) * 12 + (T0.month - first_purchase.dt.month) + 1
    )
    feat["active_months_share"] = (months_active / months_since_first).clip(upper=1.0)
    feat["mean_line_quantity"] = mean_qty
    feat["share_invoices_ge_100_units"] = share_wholesale_invoices.reindex(feat.index).fillna(0.0)
    feat["country_group"] = country_mode.map(country_group)
    feat["first_purchase_quarter"] = first_purchase.dt.to_period("Q").astype(str)

    feat = feat.reset_index()
    return feat


def _target_bucket(dates: pd.Series) -> pd.Series:
    edges = [T0 + pd.DateOffset(months=k) for k in range(TARGET_MONTHS + 1)]
    t = pd.cut(dates, bins=edges, labels=list(range(1, TARGET_MONTHS + 1)), right=True)
    return t


def build_targets(cleaned: pd.DataFrame, customer_ids) -> pd.DataFrame:
    """One row per (customer, t) for every customer in scope and t = 1..12.
    net_revenue = 0 / active = 0 where a customer has no kept lines in that
    bucket (they still get a row -- the model needs to see non-buying months)."""
    customer_ids = list(customer_ids)
    window_end = T0 + pd.DateOffset(months=TARGET_MONTHS)

    tw = cleaned[
        cleaned["Customer ID"].isin(customer_ids)
        & (cleaned["InvoiceDate"] > T0)
        & (cleaned["InvoiceDate"] <= window_end)
    ].copy()
    tw["t"] = _target_bucket(tw["InvoiceDate"])
    tw = tw[tw["t"].notna()]

    monthly = (
        tw.groupby(["Customer ID", "t"], observed=True)["net_value"]
        .sum()
        .rename("net_revenue")
    )

    idx = pd.MultiIndex.from_product(
        [customer_ids, range(1, TARGET_MONTHS + 1)], names=["Customer ID", "t"]
    )
    full = monthly.reindex(idx, fill_value=0.0).reset_index()
    full["t"] = full["t"].astype("int64")
    full["net_revenue"] = full["net_revenue"].astype("float64")
    full["active"] = (full["net_revenue"] > 0).astype("int64")
    return full


def train_holdout_split(feat: pd.DataFrame, seed: int = 26, holdout_frac: float = 0.30):
    """70/30 by customer, stratified on repeat-buyer-in-feature-window."""
    repeat_buyer = (feat["frequency"] > 1).astype(int)
    train_ids, holdout_ids = train_test_split(
        feat["Customer ID"], test_size=holdout_frac, random_state=seed, stratify=repeat_buyer
    )
    return set(train_ids.tolist()), set(holdout_ids.tolist())


def assemble_rows(feat: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Join customer features onto (customer, t) target rows. t is a feature."""
    df = targets.merge(feat, on="Customer ID", how="left")
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype("category")
    return df

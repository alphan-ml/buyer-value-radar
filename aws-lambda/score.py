"""Buyer Value Radar -- Lambda scoring handler.

Loads the two trained LightGBM boosters (Curve 1: activity/survival,
Curve 2: spend) from S3, reproduces the exact categorical encoding used
at training time (pandas .astype("category") sorted-unique-value codes
over the FULL customer population -- verified byte-identical against the
official bvr.assemble_rows()/predict_proba()/predict_spend() pipeline,
max abs diff 0.0 on the full holdout set), and combines via the same
LTV_h = sum_{t=1..h} P(active_t) * E[spend_t] logic as bvr.combine.combine.
"""
from __future__ import annotations

import base64
import json
import os

import boto3
import numpy as np
import lightgbm as lgb

S3_BUCKET = os.environ.get("MODEL_BUCKET", "giggit-buyer-value-radar-models")
MODEL_PREFIX = os.environ.get("MODEL_PREFIX", "models")

# Category code maps, reproduced exactly from the full training population
# (see outputs/features_t0.parquet -- sorted unique values, verified against
# the live pandas .astype("category") pipeline with 0.0 max abs diff).
COUNTRY_GROUP_MAP = {"Europe": 0, "Rest of world": 1, "United Kingdom": 2}
FIRST_PURCHASE_QUARTER_MAP = {
    "2009Q4": 0, "2010Q1": 1, "2010Q2": 2, "2010Q3": 3, "2010Q4": 4,
}

NUMERIC_FIELDS = [
    "recency_days", "frequency", "tenure_days", "monetary_total",
    "monetary_mean_per_invoice", "monetary_last_3m", "active_months_share",
    "mean_line_quantity", "share_invoices_ge_100_units",
]
REQUIRED_FIELDS = NUMERIC_FIELDS + ["country_group", "first_purchase_quarter"]

_booster1 = None
_booster2 = None
_train_info = None


def _load_models() -> None:
    global _booster1, _booster2, _train_info
    if _booster1 is not None:
        return
    s3 = boto3.client("s3")
    for fname in ("curve1_activity.txt", "curve2_spend.txt", "train_info.json"):
        local_path = f"/tmp/{fname}"
        if not os.path.exists(local_path):
            s3.download_file(S3_BUCKET, f"{MODEL_PREFIX}/{fname}", local_path)
    _booster1 = lgb.Booster(model_file="/tmp/curve1_activity.txt")
    _booster2 = lgb.Booster(model_file="/tmp/curve2_spend.txt")
    with open("/tmp/train_info.json") as f:
        _train_info = json.load(f)


def score_customer(features: dict) -> dict:
    _load_models()
    best_iter1 = _train_info["curve1"]["best_iteration"]
    best_iter2 = _train_info["curve2"]["best_iteration"]
    smearing = _train_info["curve2"]["smearing_factor"]

    country_group = features["country_group"]
    fpq = features["first_purchase_quarter"]
    if country_group not in COUNTRY_GROUP_MAP:
        raise ValueError(
            f"Unknown country_group '{country_group}'. Must be one of {list(COUNTRY_GROUP_MAP)}"
        )
    if fpq not in FIRST_PURCHASE_QUARTER_MAP:
        raise ValueError(
            f"Unknown first_purchase_quarter '{fpq}'. Must be one of {list(FIRST_PURCHASE_QUARTER_MAP)}"
        )

    static = [float(features[f]) for f in NUMERIC_FIELDS]
    cg_code = COUNTRY_GROUP_MAP[country_group]
    fpq_code = FIRST_PURCHASE_QUARTER_MAP[fpq]

    rows = [static + [cg_code, fpq_code, t] for t in range(1, 13)]
    X = np.array(rows, dtype="float64")

    p_active = _booster1.predict(X, num_iteration=best_iter1)
    pred_log = _booster2.predict(X, num_iteration=best_iter2)
    e_spend = np.clip(smearing * np.exp(pred_log) - 1.0, a_min=0.0, a_max=None)
    contribution = p_active * e_spend

    return {
        "p_active_by_month": [round(float(x), 4) for x in p_active],
        "e_spend_by_month": [round(float(x), 2) for x in e_spend],
        "ltv_6": round(float(contribution[:6].sum()), 2),
        "ltv_12": round(float(contribution[:12].sum()), 2),
    }


_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
}


def _resp(status: int, body: dict | str) -> dict:
    return {
        "statusCode": status,
        "headers": _HEADERS,
        "body": body if isinstance(body, str) else json.dumps(body),
    }


def handler(event, context):
    method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod")
        or "GET"
    )
    path = event.get("rawPath") or event.get("path") or ""

    if method == "OPTIONS":
        return _resp(200, "")

    if path.endswith("/health"):
        return _resp(200, {"status": "ok"})

    if method != "POST":
        return _resp(405, {"error": "use POST /score"})

    try:
        raw_body = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            raw_body = base64.b64decode(raw_body).decode()
        payload = json.loads(raw_body)
        missing = [f for f in REQUIRED_FIELDS if f not in payload]
        if missing:
            return _resp(400, {"error": f"missing fields: {missing}"})
        result = score_customer(payload)
        return _resp(200, result)
    except ValueError as e:
        return _resp(400, {"error": str(e)})
    except json.JSONDecodeError:
        return _resp(400, {"error": "invalid JSON body"})
    except Exception as e:  # noqa: BLE001
        return _resp(500, {"error": "internal error", "detail": str(e)})

"""Buyer Value Radar -- Lambda scoring handler.

Loads the two trained LightGBM boosters (Curve 1: activity/survival,
Curve 2: spend) and the cross-fitted retransform file from S3, reproduces
the exact categorical encoding used at training time (pandas
.astype("category") sorted-unique-value codes over the FULL customer
population -- verified byte-identical against the official
bvr.assemble_rows()/predict_proba()/predict_spend() pipeline, max abs diff
0.0 on the full holdout set), and combines via the same
LTV_h = sum_{t=1..h} P(active_t) * E[spend_t] logic as bvr.combine.combine.

`_bin_index`, `_expected_spend` and `_summarize` below are a pure-numpy
reimplementation of bvr.retransform.{bin_index,expected_spend} and
bvr.predictive.summarize -- no `bvr` import here, since the Lambda package
does not bundle pandas/lightgbm's Python training-side dependencies.
tests/test_score_parity.py checks the two implementations give identical
output on real holdout customers.
"""
from __future__ import annotations

import base64
import json
import os

import boto3
import lightgbm as lgb
import numpy as np

S3_BUCKET = os.environ.get("MODEL_BUCKET", "giggit-buyer-value-radar-models")
MODEL_PREFIX = os.environ.get("MODEL_PREFIX", "models")

N_RESIDUAL_QUANTILES = 201
_QUANTILE_GRID = np.linspace(0, 1, N_RESIDUAL_QUANTILES)

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
_retransform = None


def _bin_index(pred_log: np.ndarray, rt: dict) -> np.ndarray:
    return np.digitize(np.asarray(pred_log, dtype=float), np.asarray(rt["edges"], dtype=float))


def _expected_spend(pred_log: np.ndarray, rt: dict) -> np.ndarray:
    pred_log = np.asarray(pred_log, dtype=float)
    b = _bin_index(pred_log, rt)
    smearing = np.asarray(rt["smearing"], dtype=float)[b]
    return np.clip(smearing * np.exp(pred_log) - 1.0, a_min=0.0, a_max=None)


def _summarize(
    p_active: np.ndarray,
    pred_log: np.ndarray,
    rt: dict,
    horizons: tuple[int, ...] = (6, 12),
    n_draws: int = 4000,
    seed: int = 26,
) -> dict:
    p_active = np.asarray(p_active, dtype=float)
    pred_log = np.asarray(pred_log, dtype=float)

    rng = np.random.default_rng(seed)
    u_active = rng.random((12, n_draws))
    u_resid = rng.random((12, n_draws))

    bin_idx = _bin_index(pred_log, rt)
    eps = np.empty((12, n_draws))
    for m in range(12):
        eps[m] = np.interp(u_resid[m], _QUANTILE_GRID, rt["residual_quantiles"][bin_idx[m]])

    draws = np.clip(np.exp(pred_log[:, None] + eps) - 1.0, a_min=0.0, a_max=None)
    draws = draws * (u_active < p_active[:, None])

    exp_spend = _expected_spend(pred_log, rt)

    out = {}
    for h in horizons:
        total_h = draws[:h].sum(axis=0)
        p10, p50, p90 = np.percentile(total_h, [10, 50, 90])
        mean_h = float(np.sum(p_active[:h] * exp_spend[:h]))
        out[h] = {"mean": mean_h, "p10": float(p10), "p50": float(p50), "p90": float(p90)}
    return out


def _load_models() -> None:
    global _booster1, _booster2, _train_info, _retransform
    if _booster1 is not None:
        return
    s3 = boto3.client("s3")
    fnames = (
        "curve1_activity.txt", "curve2_spend.txt", "train_info.json", "spend_retransform.json"
    )
    for fname in fnames:
        local_path = f"/tmp/{fname}"
        if not os.path.exists(local_path):
            s3.download_file(S3_BUCKET, f"{MODEL_PREFIX}/{fname}", local_path)
    booster1 = lgb.Booster(model_file="/tmp/curve1_activity.txt")
    booster2 = lgb.Booster(model_file="/tmp/curve2_spend.txt")
    with open("/tmp/train_info.json") as f:
        train_info = json.load(f)
    with open("/tmp/spend_retransform.json") as f:
        retransform = json.load(f)
    _booster1, _booster2, _train_info, _retransform = booster1, booster2, train_info, retransform


def score_customer(features: dict) -> dict:
    _load_models()
    best_iter1 = _train_info["curve1"]["best_iteration"]
    best_iter2 = _train_info["curve2"]["best_iteration"]

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
    e_spend = _expected_spend(pred_log, _retransform)
    summary = _summarize(p_active, pred_log, _retransform, horizons=(6, 12), seed=26)

    return {
        "p_active_by_month": [round(float(x), 4) for x in p_active],
        "e_spend_by_month": [round(float(x), 2) for x in e_spend],
        "ltv_6": round(summary[6]["mean"], 2),
        "ltv_12": round(summary[12]["mean"], 2),
        "p10_6": round(summary[6]["p10"], 2),
        "p50_6": round(summary[6]["p50"], 2),
        "p90_6": round(summary[6]["p90"], 2),
        "p10_12": round(summary[12]["p10"], 2),
        "p50_12": round(summary[12]["p50"], 2),
        "p90_12": round(summary[12]["p90"], 2),
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


def _health() -> dict:
    """Actually loads the models and the retransform file -- a shallow
    `/health` that only checks the process is up hid a 3.5-day outage on
    fraud-radar (a sibling system), where the Lambda was up but every real
    request 500'd on a missing model file."""
    try:
        _load_models()
        return {"status": "ok"}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "reason": str(e)}


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
        health = _health()
        return _resp(200 if health["status"] == "ok" else 503, health)

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

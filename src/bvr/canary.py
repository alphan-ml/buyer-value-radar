"""Live Eval canary for Buyer Value Radar.

Calls the LIVE scoring endpoint once per row in canary/rows.json, computes
WAPE at 12 months over the rows that were actually scored, and compares it
to the recorded outputs/metrics.json value within a fixed tolerance. Prints
one JSON record (the Live Eval ledger schema) to stdout.

No fallback numbers: a request that errors counts as an error, never a
guess, and forces match=false -- a broken endpoint must show up as broken,
not as a lucky WAPE computed on however many rows happened to succeed.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANARY_ROWS_PATH = ROOT / "canary" / "rows.json"
METRICS_PATH = ROOT / "outputs" / "metrics.json"

SYSTEM = "buyer-value-radar"
ENDPOINT = "https://guijt78nlb.execute-api.us-east-1.amazonaws.com/score"
METRIC = "wape_12m"
TOLERANCE = 0.060
REQUEST_TIMEOUT_S = 20


def _git_short_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _score(request: dict) -> dict:
    body = json.dumps(request).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        return json.loads(resp.read())


def _wape(pairs: list[tuple[float, float]]) -> float | None:
    """pairs of (predicted, actual). None if there is nothing to divide by."""
    den = sum(abs(actual) for _pred, actual in pairs)
    if den == 0:
        return None
    return sum(abs(pred - actual) for pred, actual in pairs) / den


def _percentile(sorted_values: list[float], p: float) -> int:
    if not sorted_values:
        return 0
    idx = min(int(len(sorted_values) * p), len(sorted_values) - 1)
    return round(sorted_values[idx])


def run() -> dict:
    rows = json.loads(CANARY_ROWS_PATH.read_text())["rows"]
    recorded_metrics = json.loads(METRICS_PATH.read_text())
    recorded_12m = recorded_metrics["wape_12m"]
    recorded_6m = recorded_metrics["wape_6m"]

    pairs_12: list[tuple[float, float]] = []
    pairs_6: list[tuple[float, float]] = []
    latencies_ms: list[float] = []
    errors = 0

    t_start = time.time()
    for row in rows:
        t0 = time.time()
        try:
            result = _score(row["request"])
            latencies_ms.append((time.time() - t0) * 1000.0)
            pairs_6.append((result["ltv_6"], row["actual_6m"]))
            pairs_12.append((result["ltv_12"], row["actual_12m"]))
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, OSError) as e:
            errors += 1
            print(f"error scoring customer {row['customer_id']}: {e}", file=sys.stderr)
    duration_s = time.time() - t_start

    observed_12m = _wape(pairs_12)
    observed_6m = _wape(pairs_6)

    match = (
        errors == 0
        and observed_12m is not None
        and abs(observed_12m - recorded_12m) <= TOLERANCE
    )

    latencies_ms.sort()
    p50_ms = _percentile(latencies_ms, 0.50)
    p95_ms = _percentile(latencies_ms, 0.95)

    release = _git_short_sha()
    lines = [
        f"$ canary buyer-value-radar --n {len(rows)} --endpoint /score",
        f"release {release} · {len(rows)} holdout rows · {errors} errors",
    ]
    if observed_12m is not None:
        lines.append(
            f"wape_12m {observed_12m:.4f} · recorded {recorded_12m:.4f} · "
            f"tolerance {TOLERANCE:.3f} · {'MATCH' if match else 'NO MATCH'}"
        )
    else:
        lines.append("wape_12m: no rows scored successfully · NO MATCH")
    if observed_6m is not None:
        lines.append(f"wape_6m {observed_6m:.4f} · recorded {recorded_6m:.4f}")
    lines.append(f"p50 {p50_ms} ms · p95 {p95_ms} ms · {duration_s:.1f}s total")

    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system": SYSTEM,
        "kind": "canary",
        "release": release,
        "endpoint": ENDPOINT,
        "n": len(rows),
        "metric": METRIC,
        "recorded": recorded_12m,
        "observed": observed_12m,
        "tolerance": TOLERANCE,
        "match": match,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "errors": errors,
        "duration_s": round(duration_s, 1),
        "lines": lines,
        "extra": {"wape_6m": observed_6m, "recorded_wape_6m": recorded_6m},
    }


def main() -> None:
    print(json.dumps(run()))


if __name__ == "__main__":
    main()

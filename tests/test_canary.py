"""Live Eval canary tests."""
from __future__ import annotations

import json

from bvr import canary


def test_canary_rows_are_all_in_the_holdout_split():
    rows = json.loads(canary.CANARY_ROWS_PATH.read_text())["rows"]
    holdout_ids = set(json.loads(canary.METRICS_PATH.parent.joinpath("split_ids.json").read_text())["holdout_ids"])

    ids = [row["customer_id"] for row in rows]
    assert len(ids) == 300
    assert len(set(ids)) == 300
    assert set(ids) <= holdout_ids


def test_wape_matches_hand_computation():
    pairs = [(150.0, 100.0), (50.0, 50.0), (0.0, 10.0)]
    # |150-100| + |50-50| + |0-10| = 60; |100|+|50|+|10| = 160
    assert canary._wape(pairs) == 60.0 / 160.0


def test_wape_is_none_when_denominator_is_zero():
    assert canary._wape([]) is None
    assert canary._wape([(5.0, 0.0)]) is None


def test_run_forces_match_false_on_any_error(tmp_path, monkeypatch):
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(json.dumps({
        "rows": [
            {"customer_id": 1, "request": {}, "actual_6m": 100.0, "actual_12m": 200.0},
            {"customer_id": 2, "request": {}, "actual_6m": 100.0, "actual_12m": 200.0},
        ]
    }))
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"wape_12m": 0.0, "wape_6m": 0.0}))

    monkeypatch.setattr(canary, "CANARY_ROWS_PATH", rows_path)
    monkeypatch.setattr(canary, "METRICS_PATH", metrics_path)

    calls = {"n": 0}

    def fake_score(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"ltv_6": 100.0, "ltv_12": 200.0}  # perfect match, would pass alone
        raise OSError("endpoint unreachable")

    monkeypatch.setattr(canary, "_score", fake_score)

    record = canary.run()

    assert record["errors"] == 1
    assert record["match"] is False
    assert record["n"] == 2
    # the one successful row's real numbers are used, not a fabricated fallback
    assert record["observed"] == 0.0
    assert record["extra"]["wape_6m"] == 0.0


def test_run_matches_when_within_tolerance(tmp_path, monkeypatch):
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(json.dumps({
        "rows": [
            {"customer_id": 1, "request": {}, "actual_6m": 100.0, "actual_12m": 200.0},
        ]
    }))
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"wape_12m": 0.5, "wape_6m": 0.5}))

    monkeypatch.setattr(canary, "CANARY_ROWS_PATH", rows_path)
    monkeypatch.setattr(canary, "METRICS_PATH", metrics_path)
    monkeypatch.setattr(canary, "_score", lambda request: {"ltv_6": 150.0, "ltv_12": 300.0})

    record = canary.run()

    assert record["errors"] == 0
    assert record["observed"] == 0.5
    assert record["match"] is True
    assert len(record["lines"]) <= 8

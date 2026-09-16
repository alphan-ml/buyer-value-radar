"""Export output-shape test (BUILD PACK Section 7)."""
from __future__ import annotations

import json

from bvr import export as export_mod


def test_export_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(export_mod, "OUT_DIR", tmp_path)

    (tmp_path / "metrics.json").write_text(json.dumps({"wape_12m": 0.5}))
    (tmp_path / "calibration.json").write_text(json.dumps({"calibration_by_month": []}))
    (tmp_path / "data_quality.json").write_text(
        json.dumps({"row_count": 10, "unique_customers": 5, "pct_rows_without_customer_id": 1.0})
    )
    (tmp_path / "clean_stats.json").write_text(json.dumps({"rows_kept": 8}))
    (tmp_path / "train_info.json").write_text(json.dumps({"curve1": {}, "curve2": {}}))

    site_data = export_mod.export()

    assert site_data["model"] == "buyer-value-radar"
    for key in ["metrics", "calibration", "data_quality_summary", "clean_stats", "train_info"]:
        assert key in site_data

    written = json.loads((tmp_path / "site_data.json").read_text())
    assert written == site_data

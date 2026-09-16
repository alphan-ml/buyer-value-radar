"""Export the site-facing data artifact (BUILD PACK Section 8, scoped for
this stage).

Fable's v2 correction (Section 1.2) moved Buyer Value Radar off its own
page: it becomes a "Score a Customer" block inside the existing
customer-lifecycle.html, served from a GCP endpoint (Section 6.3). GCP
is not set up yet (Gate 1), and no file on giggitai.com is touched without
Leon's explicit go-ahead. This export step writes the data artifact only
-- outputs/site_data.json -- so the site block can be wired up later
without re-deriving any number. No site file is read or written here.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"


def export() -> dict:
    metrics = json.loads((OUT_DIR / "metrics.json").read_text())
    calibration = json.loads((OUT_DIR / "calibration.json").read_text())
    data_quality = json.loads((OUT_DIR / "data_quality.json").read_text())
    clean_stats = json.loads((OUT_DIR / "clean_stats.json").read_text())
    train_info = json.loads((OUT_DIR / "train_info.json").read_text())

    site_data = {
        "model": "buyer-value-radar",
        "metrics": metrics,
        "calibration": calibration,
        "data_quality_summary": {
            "row_count": data_quality["row_count"],
            "unique_customers": data_quality["unique_customers"],
            "pct_rows_without_customer_id": data_quality["pct_rows_without_customer_id"],
        },
        "clean_stats": clean_stats,
        "train_info": train_info,
    }
    (OUT_DIR / "site_data.json").write_text(json.dumps(site_data, indent=2, default=str))
    return site_data


if __name__ == "__main__":
    export()

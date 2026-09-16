"""Fetch the Online Retail II dataset (UCI ML Repository, id 502).

Downloads the official zip, verifies/records its sha256, extracts the single
.xlsx (two year sheets), concatenates them, and writes a parquet file.
Resumable: skips the download when the parquet already exists and the
recorded checksum file matches the zip on disk.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
ZIP_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ZIP_PATH = RAW_DIR / "online_retail_ii.zip"
PARQUET_PATH = RAW_DIR / "online_retail_ii.parquet"
CHECKSUM_PATH = RAW_DIR / "online_retail_ii.zip.sha256"
ATTRIBUTION_PATH = Path(__file__).resolve().parents[2] / "data" / "ATTRIBUTION.md"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
        print(f"[fetch] downloaded {written:,} bytes in {time.time() - t0:.1f}s "
              f"(server reported content-length={total:,})", file=sys.stderr)


def fetch(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if PARQUET_PATH.exists() and CHECKSUM_PATH.exists() and not force:
        recorded = CHECKSUM_PATH.read_text().strip()
        if ZIP_PATH.exists() and _sha256(ZIP_PATH) == recorded:
            print(f"[fetch] parquet already present and checksum matches, skipping download: {PARQUET_PATH}")
            return PARQUET_PATH

    if not ZIP_PATH.exists() or force:
        print(f"[fetch] downloading {ZIP_URL}")
        _download(ZIP_URL, ZIP_PATH)

    sha = _sha256(ZIP_PATH)
    first_time = not CHECKSUM_PATH.exists()
    CHECKSUM_PATH.write_text(sha + "\n")
    print(f"[fetch] zip sha256: {sha} ({'recorded now' if first_time else 'matches on-disk record'})")

    print("[fetch] reading xlsx from zip (both year sheets)...")
    t0 = time.time()
    with zipfile.ZipFile(ZIP_PATH) as z:
        xlsx_names = [n for n in z.namelist() if n.lower().endswith(".xlsx")]
        if not xlsx_names:
            raise RuntimeError(f"no .xlsx found in zip; contents: {z.namelist()}")
        xlsx_name = xlsx_names[0]
        with z.open(xlsx_name) as f:
            data = io.BytesIO(f.read())

    # Both sheets share this schema in the source workbook. Invoice and
    # StockCode contain letters in some rows (e.g. cancellations "C536379",
    # non-product codes "POST", "M") so they must be read as strings, not
    # left to pandas' per-sheet type inference -- otherwise one sheet infers
    # int64 and the other object/str, and concatenation produces a mixed
    # column that pyarrow refuses to write.
    dtype_map = {
        "Invoice": "string",
        "StockCode": "string",
        "Description": "string",
        "Country": "string",
    }

    xls = pd.ExcelFile(data, engine="openpyxl")
    print(f"[fetch] sheets found: {xls.sheet_names}")
    frames = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, dtype=dtype_map)
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="raise").astype("int64")
        df["Price"] = pd.to_numeric(df["Price"], errors="raise").astype("float64")
        df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="raise").astype("Int64")
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["_source_sheet"] = sheet
        frames.append(df)
        print(f"[fetch]   sheet '{sheet}': {len(df):,} rows")

    full = pd.concat(frames, ignore_index=True)
    print(f"[fetch] total rows across sheets: {len(full):,} (took {time.time() - t0:.1f}s to parse xlsx)")
    print(f"[fetch] dtypes:\n{full.dtypes}")

    full.to_parquet(PARQUET_PATH, index=False)
    print(f"[fetch] wrote {PARQUET_PATH} ({PARQUET_PATH.stat().st_size:,} bytes)")

    ATTRIBUTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    ATTRIBUTION_PATH.write_text(
        "# Data attribution\n\n"
        "Dataset: Online Retail II\n"
        "Source: UCI Machine Learning Repository, id 502\n"
        "https://archive.ics.uci.edu/dataset/502/online+retail+ii\n"
        "License: CC BY 4.0\n\n"
        f"Zip URL: {ZIP_URL}\n"
        f"Zip sha256: {sha}\n"
        f"Zip size (bytes): {ZIP_PATH.stat().st_size}\n"
        f"Pulled at (UTC): {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
        f"Rows after concatenating both year sheets: {len(full):,}\n"
        f"Sheets: {json.dumps(xls.sheet_names)}\n"
    )
    print(f"[fetch] wrote {ATTRIBUTION_PATH}")

    return PARQUET_PATH


if __name__ == "__main__":
    fetch(force="--force" in sys.argv)

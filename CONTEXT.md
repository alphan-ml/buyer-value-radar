# CONTEXT.md — buyer-value-radar

## How to run

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"
python3 -m bvr.cli all
python3 -m pytest -q
python3 -m ruff check .
```

Mac-specific note: `lightgbm`'s wheel expects Homebrew's `libomp.dylib`, which is not installed on this machine (no sudo/Homebrew in Gate 0). Fixed once, inside this repo's own `.venv`, with:
```
install_name_tool -change @rpath/libomp.dylib /opt/anaconda3/lib/libomp.dylib \
  .venv/lib/python3.11/site-packages/lightgbm/lib/lib_lightgbm.dylib
```
Anaconda's existing `libomp.dylib` is used as the real library; nothing was installed system-wide. Not needed again unless the venv is recreated.

## Data facts from the real run (Sep 13, 2026)

- Raw: 1,067,371 rows (525,461 + 541,910 across the two year sheets), zip sha256 `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb`.
- Cleaning (`clean.py`): 243,007 rows dropped for no Customer ID; 6,207 dropped for Quantity<=0 or Price<=0 on a non-cancellation line; 3,726 dropped for a non-product StockCode; 820,567 rows kept (17,933 of them cancellation lines, kept and netted).
- Features (T0 = Dec 9, 2010 23:59:59): 4,314 customers in scope (>=1 kept purchase in the feature window), 51,768 customer-month target rows (4,314 x 12). Split 70/30 by customer, seed 26, stratified on repeat-buyer-in-feature-window: 3,019 train / 1,295 holdout.
- Curve 1 (activity): best iteration 123, fit on 2,717 customers, early-stopped on 302.
- Curve 2 (spend): best iteration 156, fit on 1,707 customers (active rows only), early-stopped on 190, Duan smearing factor 1.1541.
- Holdout results (1,295 customers): WAPE 6m 0.828, WAPE 12m 0.688, median APE 6m 0.543 (n=567 actual>0), median APE 12m 0.538 (n=797 actual>0), curve-2-alone WAPE on active rows 0.514, curve-1 ROC-AUC 0.757–0.811 across the 12 months.
- BG/NBD + Gamma-Gamma baseline: **not built** (timeboxed out; see README Design decisions). Logged here, not faked.

## Decisions

| ID | Decision | Rejected alternative |
|---|---|---|
| D1 | LightGBM for both curves | `lifetimes` (unmaintained since 2020); `pymc-marketing` (needs Python >=3.12, Bayesian fit too slow for the timebox) |
| D2 | log1p(spend) + Duan smearing for Curve 2 | Raw-scale regression — a few very large buyers dominate squared-error loss |
| D3 | Report median APE alongside mean MAPE | Winsorizing or dropping near-zero-actual holdout rows — would silently change a real, verified number instead of disclosing why the mean is unusable |
| D4 | Added StockCode "B" ("Adjust bad debt") to the non-product code list | Leaving `schema.py`'s Price sanity bound at 50,000 and treating the -53,594.36 row as a corrupt-download signal — it is a real, single bookkeeping line, not corruption |
| D5 | `export.py` writes `outputs/site_data.json` only; no site file is read or written by this repo | Building the `use-cases/customer-lifecycle.html` "Score a Customer" block now — it depends on the GCP endpoint in BUILD INSTRUCTION v2 Section 6.3, which needs Gate 1 cloud accounts not yet created, and no file on giggitai.com changes without Leon's explicit go-ahead (standing rule) |
| D6 | Public identity on this repo is `alphan-ml` / "Alpha Nyarera" | The identity line in the two 10:00 ET build-deploy packs, withdrawn by BUILD INSTRUCTION v2 Section 1.1, which wins where the packs conflict |

## Open items

- BG/NBD + Gamma-Gamma baseline not built (see D1). Build only if a later session has time left after the six-system push.
- GCP serving (BUILD INSTRUCTION v2 Section 6.3) and the `customer-lifecycle.html` "Score a Customer" block: blocked on Gate 1 (GCP project, BigQuery, Vertex AI) and on Leon's go-ahead before any site file changes.
- Kaggle "Data Access and Use" clauses for `ieee-fraud-detection` and `instacart-market-basket-analysis` (needed for Fraud Radar and Reorder Radar, not this repo): still need Leon's own logged-in view — could not be fetched from a logged-out session.
- Leon-authored commit within 7 days of push (by Sep 20, 2026): suggested edit — the country-group definition in `features.py` (which countries count as "Europe") is a judgment call worth a second pair of eyes before this goes live anywhere real.

## Task reports

### Task: build the modeling pipeline (fetch through export)
- STATUS: done, all steps run on the real, full dataset.
- BUILT: `schema.py`, `features.py`, `survival_model.py`, `monetary_model.py`, `combine.py`, `eval.py`, `export.py`, `cli.py`; `tests/test_schema.py`, `tests/test_clean.py`, `tests/test_no_leak.py`, `tests/test_combine.py`, `tests/test_export_shape.py`; `pyproject.toml`, `.gitignore`, `.env.example`, `LICENSE`, `.github/workflows/ci.yml`.
- TESTED: `python3 -m pytest -q` — 15 passed. `python3 -m ruff check .` — all checks passed. `bvr all --force` run twice end to end (once mid-build, once after the lint pass) with identical row counts, split sizes, and metrics both times — confirms the fixed seed (26) makes the pipeline reproducible.
- SPEC CHECK: BUILD PACK Sections 2–7 followed; BUILD INSTRUCTION v2 Section 1 corrections applied (identity, no separate site page).
- OPEN: see Open items above.
- NEXT: README + this file committed; QA gate (Section 9) before `gh repo create`; Drive backup; report to Leon and wait for "go" before any public push.
- TIME: 2026-09-13, ~15:30–16:00 ET.

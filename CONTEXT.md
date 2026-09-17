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
| D5 | `export.py` writes `outputs/site_data.json` only; no site file is read or written by this repo | Building the `use-cases/customer-lifecycle.html` "Score a Customer" block now — it depends on the GCP endpoint in BUILD INSTRUCTION v2 Section 6.3, which needs Gate 1 cloud accounts not yet created, and no file on giggitai.com changes without the owner's explicit go-ahead (standing rule) |
| D6 | Public identity on this repo is `alphan-ml` / "Alpha Nyarera" | The identity line in the two 10:00 ET build-deploy packs, withdrawn by BUILD INSTRUCTION v2 Section 1.1, which wins where the packs conflict |

## Open items

- BG/NBD + Gamma-Gamma baseline not built (see D1). Build only if a later session has time left after the six-system push.
- GCP serving (BUILD INSTRUCTION v2 Section 6.3) and the `customer-lifecycle.html` "Score a Customer" block: blocked on Gate 1 (GCP project, BigQuery, Vertex AI) and on the owner's go-ahead before any site file changes.
- Kaggle "Data Access and Use" clauses for `ieee-fraud-detection` and `instacart-market-basket-analysis` (needed for Fraud Radar and Reorder Radar, not this repo): still need the owner's own logged-in view — could not be fetched from a logged-out session.
- An owner-authored commit within 7 days of push (by Sep 20, 2026): suggested edit — the country-group definition in `features.py` (which countries count as "Europe") is a judgment call worth a second pair of eyes before this goes live anywhere real.

## Task reports

### Task: build the modeling pipeline (fetch through export)
- STATUS: done, all steps run on the real, full dataset.
- BUILT: `schema.py`, `features.py`, `survival_model.py`, `monetary_model.py`, `combine.py`, `eval.py`, `export.py`, `cli.py`; `tests/test_schema.py`, `tests/test_clean.py`, `tests/test_no_leak.py`, `tests/test_combine.py`, `tests/test_export_shape.py`; `pyproject.toml`, `.gitignore`, `.env.example`, `LICENSE`, `.github/workflows/ci.yml`.
- TESTED: `python3 -m pytest -q` — 15 passed. `python3 -m ruff check .` — all checks passed. `bvr all --force` run twice end to end (once mid-build, once after the lint pass) with identical row counts, split sizes, and metrics both times — confirms the fixed seed (26) makes the pipeline reproducible.
- SPEC CHECK: BUILD PACK Sections 2–7 followed; BUILD INSTRUCTION v2 Section 1 corrections applied (identity, no separate site page).
- OPEN: see Open items above.
- NEXT: README + this file committed; QA gate (Section 9) before `gh repo create`; Drive backup; report to the owner and wait for "go" before any public push.
- TIME: 2026-09-13, ~15:30–16:00 ET.

## Decisions (added)

| ID | Decision | Rejected alternative |
|---|---|---|
| D7 | AWS (Lambda + API Gateway + S3) for BVR serving, per the owner's explicit instruction ("Use AWS for Buyer Value Radar") | GCP serving per BUILD INSTRUCTION v2 Section 6.3 (superseded); SageMaker endpoint (idle cost for a stateless two-model scorer that gets occasional traffic — Lambda scales to zero) |
| D8 | Lambda scoring code (`score.py`) reconstructs LightGBM's categorical encoding manually (sorted-unique-value integer codes over the full training population, e.g. `country_group`: Europe=0, Rest of world=1, United Kingdom=2; `first_purchase_quarter`: 2009Q4=0..2010Q4=4) instead of shipping pandas | Bundling pandas to preserve `.astype("category")` at inference time — unnecessary risk and ~50MB+ extra deployment size once the encoding is verified byte-identical (see Task report below) |
| D9 | Lambda deployment uses lightgbm 3.3.5 + scipy + numpy<2 (no scikit-learn), with a scikit-learn-wheel-derived `libgomp.so.1` bundled at `lib/libgomp.so.1` | lightgbm 4.7.0 (the version actually used to train) — pulls in `narwhals` and is otherwise fine, but 3.3.5 was chosen after confirming byte-identical predictions from the real trained model files, to keep dependency surface smaller; Lambda's Amazon Linux base image has no `libgomp.so.1` preinstalled, which is why lightgbm's `import` fails without the bundled copy |

## Open items (updated)

- ~~GCP serving (BUILD INSTRUCTION v2 Section 6.3)~~ — superseded by D7; AWS serving is live and verified (see Task report below).
- `customer-lifecycle.html` "Score a Customer" UI block: still blocked on the owner's go-ahead before any file on giggitai.com changes (standing rule) — the API endpoint it would call now exists and is tested, so this is purely a front-end + explicit-approval step, not a backend blocker.
- BG/NBD + Gamma-Gamma baseline not built (see D1). Unchanged.
- Kaggle "Data Access and Use" clauses for `ieee-fraud-detection` / `instacart-market-basket-analysis`: unchanged, still needs the owner's own logged-in view.
- An owner-authored commit within 7 days of push (by Sep 20, 2026): unchanged suggestion re: `country_group` definition.

## Task reports (added)

### Task: build and deploy the AWS scoring endpoint (Lambda + API Gateway + S3)

- STATUS: done, live, tested end-to-end against the real trained models.
- BUILT: `aws-lambda/package/score.py` (Lambda handler — loads both boosters from S3 into `/tmp` on cold start, reconstructs categorical encoding, runs the same `LTV_h = sum P(active_t) * E[spend_t]` combination as `bvr.combine.combine`); `aws-lambda/package/lib/libgomp.so.1` (OpenMP runtime, sourced from a scikit-learn manylinux2014 wheel, since Lambda's base image lacks it and lightgbm's compiled library dlopens it at import time).
- INFRASTRUCTURE: S3 bucket `giggit-buyer-value-radar-models` (`models/` = model artifacts, `lambda-code/` = deployment zip); IAM role `bvr-lambda-execution-role` (AWSLambdaBasicExecutionRole + inline `s3:GetObject` on `models/*`); Lambda function `bvr-score-customer` (python3.11, 512MB, 30s timeout, ~166MB max memory used in practice); API Gateway HTTP API `bvr-score-api` — live at `https://guijt78nlb.execute-api.us-east-1.amazonaws.com` (`POST /score`, `GET /health`, CORS enabled for browser calls).
- VERIFIED: manually re-derived the categorical code maps from `outputs/features_t0.parquet` (full population) and confirmed, in an isolated Python process, that manual integer-code encoding produces **byte-identical** (`max abs diff = 0.0`) `P(active_t)` and `E[spend_t]` predictions versus the official `bvr.assemble_rows()` / `predict_proba()` / `predict_spend()` pipeline, across the entire 1,295-customer holdout set. Confirmed lightgbm 3.3.5 loads the real 4.7.0-trained model files and reproduces identical predictions to full displayed precision. End-to-end tested via the live HTTPS endpoint for a real holdout customer (ID 12352): `ltv_6=179.59`, `ltv_12=426.92`, matching the reference computed on the training machine exactly. Tested error paths (missing fields, unknown category, `/health`, CORS preflight) — all return the expected 4xx/200 responses.
- SPEC CHECK: no file on giggitai.com was touched — this is backend-only, per the standing rule that site changes need the owner's explicit go-ahead (see D5, updated Open items).
- NEXT: front-end "Score a Customer" block on `customer-lifecycle.html`, gated on the owner's explicit approval before any deploy; local-preview sign-off required first per the build instruction's own STOP-before-deploy rule.
- TIME: 2026-09-16, ~05:50–06:10 ET.

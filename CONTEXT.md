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

## Decisions (added, baselines + ranking + Tweedie variant)

| ID | Decision | Rejected alternative |
|---|---|---|
| D10 | "Repeat" baseline = each customer's `monetary_total` (the feature table's spend over the feature window ending at T0, which for every customer is the same fixed ~12-month calendar window Dec 1 2009 -- T0) carried forward as the 12-month prediction, halved for 6-month | A per-customer trailing-365-day recompute from raw transactions -- `monetary_total` already is a fixed calendar window ending at T0 for every customer, so it does not need a separate column; recomputing would just reproduce it, 8 days off (the feature window opens Dec 1, not Dec 9) |
| D11 | "Segment mean" baseline: buyer-size tercile edges computed from TRAINING customers' `monetary_total` only via `qcut(3)`; each customer's prediction is the training-tercile's mean actual spend (training rows only), computed separately at 6- and 12-month horizons (not by halving) | Halving the 12-month segment mean for the 6-month prediction, the way the repeat baseline does -- rejected because the segment mean is already an aggregate statistic re-derived per horizon from real training data at no extra cost, so halving would throw away information instead of using it |
| D12 | Tweedie spend-model variant trains on raw (untransformed) net revenue with LightGBM's `tweedie` objective and log link -- no log1p / Duan smearing, since the Tweedie mean prediction is already on the natural scale; variance power tuned over `{1.2, 1.5, 1.8}` on the validation (early-stop) split only | Applying Tweedie loss on top of the log1p-transformed target -- double-transforms the same skew the objective is meant to model, and defeats the point of comparing two different distributional assumptions for the same raw target |
| D13 | Promotion rule for the Tweedie variant: ship it only if holdout WAPE 12m improves by >= 0.02 **and** top-decile capture (12m) does not fall versus the current model; otherwise keep the current model and record the comparison | A single-metric gate (WAPE only) -- rejected because a spend model that trades WAPE for a worse top-decile ranking would be actively harmful for the site's stated use (surfacing the highest-value customers) |

## Baseline and Tweedie run (added, Sep 19, 2026)

- Same seed (26), same committed `outputs/split_ids.json` (3,019 train / 1,295 holdout) -- verified byte-identical to the previously committed file before any code changed, and again after training, so the split was never touched.
- Model vs. baselines (12m holdout): model WAPE 0.688, repeat baseline WAPE 0.902, segment-mean baseline WAPE 1.124. Model wins on WAPE, Spearman (0.612 vs 0.600 / 0.573), and top-decile capture (0.539 vs 0.536 / 0.231).
- Tweedie variant (variance power 1.2, chosen on the validation split from `{1.2, 1.5, 1.8}`): holdout WAPE 12m 0.704, worse than the current model's 0.688 (not an improvement) -- per D13, the current (log1p + Duan smearing) model stays shipped. The Tweedie booster is still written to `outputs/checkpoints/curve2_spend_tweedie.txt` (gitignored, like all model checkpoints) for the owner to inspect; nothing in `aws-lambda/` or the deployed Lambda was touched.
- Full numbers: README "Baselines and ranking metrics" / "Spend-model variant: Tweedie" sections; `outputs/metrics.json` -> `baselines`, `ranking`, `spend_model_variants`.

## Task reports (added)

### Task: build and deploy the AWS scoring endpoint (Lambda + API Gateway + S3)

- STATUS: done, live, tested end-to-end against the real trained models.
- BUILT: `aws-lambda/package/score.py` (Lambda handler — loads both boosters from S3 into `/tmp` on cold start, reconstructs categorical encoding, runs the same `LTV_h = sum P(active_t) * E[spend_t]` combination as `bvr.combine.combine`); `aws-lambda/package/lib/libgomp.so.1` (OpenMP runtime, sourced from a scikit-learn manylinux2014 wheel, since Lambda's base image lacks it and lightgbm's compiled library dlopens it at import time).
- INFRASTRUCTURE: S3 bucket `giggit-buyer-value-radar-models` (`models/` = model artifacts, `lambda-code/` = deployment zip); IAM role `bvr-lambda-execution-role` (AWSLambdaBasicExecutionRole + inline `s3:GetObject` on `models/*`); Lambda function `bvr-score-customer` (python3.11, 512MB, 30s timeout, ~166MB max memory used in practice); API Gateway HTTP API `bvr-score-api` — live at `https://guijt78nlb.execute-api.us-east-1.amazonaws.com` (`POST /score`, `GET /health`, CORS enabled for browser calls).
- VERIFIED: manually re-derived the categorical code maps from `outputs/features_t0.parquet` (full population) and confirmed, in an isolated Python process, that manual integer-code encoding produces **byte-identical** (`max abs diff = 0.0`) `P(active_t)` and `E[spend_t]` predictions versus the official `bvr.assemble_rows()` / `predict_proba()` / `predict_spend()` pipeline, across the entire 1,295-customer holdout set. Confirmed lightgbm 3.3.5 loads the real 4.7.0-trained model files and reproduces identical predictions to full displayed precision. End-to-end tested via the live HTTPS endpoint for a real holdout customer (ID 12352): `ltv_6=179.59`, `ltv_12=426.92`, matching the reference computed on the training machine exactly. Tested error paths (missing fields, unknown category, `/health`, CORS preflight) — all return the expected 4xx/200 responses.
- SPEC CHECK: no file on giggitai.com was touched — this is backend-only, per the standing rule that site changes need the owner's explicit go-ahead (see D5, updated Open items).
- NEXT: front-end "Score a Customer" block on `customer-lifecycle.html`, gated on the owner's explicit approval before any deploy; local-preview sign-off required first per the build instruction's own STOP-before-deploy rule.
- TIME: 2026-09-16, ~05:50–06:10 ET.

### Task: baselines, ranking metrics, and a Tweedie spend-model variant

- STATUS: done.
- BUILT: `repeat_baseline`, `segment_mean_baseline`, `spearman_corr`, `top_decile_capture`, `_decile_table`, `_baseline_metrics` in `eval.py`; `evaluate()` now takes `train_rows`/`train_ids` and writes `baselines` + `ranking` into `outputs/metrics.json`. `fit_tweedie`, `predict_spend_tweedie`, `load_tweedie` in `monetary_model.py`. `cli.py` `step_train` now also fits the Tweedie variant; `step_eval` evaluates both spend-model variants on the holdout, applies the promotion rule (D13), and writes `spend_model_variants`. `tests/test_baselines.py` (2 tests).
- TESTED: `python3 -m pytest -q` — 23 passed (21 previously + 2 new). `python3 -m ruff check .` — all checks passed. Re-ran `bvr all` end to end on the real dataset; confirmed the regenerated `outputs/split_ids.json` and the pre-change `outputs/metrics.json` were byte-identical to the previously committed versions before touching any code, then re-ran `train`/`eval`/`export` with the new code (train step ~1.3s, eval step ~0.4s, both far inside the 90-minute stop-if budget).
- SPEC CHECK: split/seed/holdout customers unchanged (verified byte-identical); Lambda handler, `aws-lambda/`, and anything deployed untouched; no site file touched; no raw data added to git (fetched dataset stays in `data/raw/`, gitignored).
- RESULT: Tweedie variant (best variance power 1.2) did not clear the promotion bar (holdout WAPE 12m 0.704 vs. current 0.688 — worse, not better) — current model stays shipped. See "Baseline and Tweedie run" above for full numbers.
- OPEN: none from this task.
- TIME: 2026-09-19, ~06:00–06:15 UTC.

## Decisions (added, cross-fitted retransform + predictive range + cross-validated evaluation)

| ID | Decision | Rejected alternative |
|---|---|---|
| D14 | Curve 2's Duan smearing factor is **cross-fitted, per prediction level** (`retransform.py`: 5-fold KFold over train customers, out-of-fold residuals binned into 10 quantile bins of the OOF prediction, one smearing factor per bin) instead of computed in-sample on the fit rows | Keeping the single in-sample factor — it is computed from residuals the booster has already partly memorised (1.154 in-sample vs. 1.274 cross-fitted on this run), and under-forecasts total 12-month revenue by 15.1% on average across 15 cross-validation folds (vs. 4.3% for the cross-fitted method) |
| D15 | The per-customer point forecast is **P50** (`predictive.py`: Monte Carlo draws from `Bernoulli(p_active) x lognormal-ish residual`, seed always 26), not the mean; `mean` is kept and used only for totals (LTV sums, revenue) | Reporting only the mean, as before — WAPE (the metric this whole build is scored against) is minimised by the median, not the mean, so a mean-only forecast was optimising the wrong statistic for the number actually being judged |
| D16 | The **headline evaluation is 5-fold x 3-repeat customer cross-validation over all 4,314 customers** (`eval.cross_validate`, run inside `step_eval`), not the single 30% holdout — each fold refits Curve 1, Curve 2, and the retransform exactly as production, from scratch, on that fold's own training customers | Reporting only the single holdout, as before — its bootstrap 95% interval on 12-month WAPE of P50 is [0.576, 0.795], and on this build's split its revenue error is +8.5% (over-forecast) while the 15-fold average is -4.3% (under-forecast); a single split can and did point the wrong way. The Tweedie promotion rule (D13) now reads this table, not the holdout. |

## Cross-fitted retransform, predictive range, and cross-validation run (added, Sep 24, 2026)

- Same seed (26), same committed `outputs/split_ids.json` (3,019 train / 1,295 holdout) and the same `outputs/features_t0.parquet`/`targets_monthly.parquet` — verified byte-identical to the previously committed split before any code changed. Curve 1 and Curve 2 `best_iteration` unchanged (123, 156).
- Cross-fitted retransform (`outputs/spend_retransform.json`, ~53 KB, committed): global smearing 1.2742 (vs. 1.1541 in-sample), per-bin smearing `[1.544, 1.228, 1.232, 1.175, 1.195, 1.292, 1.268, 1.174, 1.184, 1.451]`, fit on 6,509 active rows across 1,897 train customers.
- Holdout (1,295 customers): 12m WAPE P50 0.672 / mean 0.711; 6m WAPE P50 0.768 / mean 0.852; revenue error of the 12m mean +0.085; P10-P90 coverage 0.793; `wape_12m_p50_ci95` [0.576, 0.795].
- Cross-validation, mean over 15 folds (section D, the headline number): 12m WAPE P50 0.635 (SD 0.067) / mean 0.660 (SD 0.073); 6m WAPE P50 0.686 (SD 0.063); revenue error of the 12m mean -0.043 (SD 0.105), vs. -0.151 (SD 0.095) for the old in-sample-smearing method; Spearman 0.630 (SD 0.018); top-decile capture 0.613 (SD 0.057); P10-P90 coverage 0.794 (SD 0.015).
- Bug found and fixed during this build: the cross-validation fold refits (`survival_model.fit` / `monetary_model.fit`) were calling the same `booster.save_model()` path used by production training, silently overwriting the shipped `outputs/checkpoints/*.txt` on every fold. Added a `save: bool = True` parameter to both `fit()` functions; `cross_validate()` passes `save=False`. Caught by noticing `outputs/metrics.json`'s holdout numbers changed between two back-to-back `bvr eval --force` runs with no code change in between — the second run was scoring against a corrupted (last-fold) model.
- Tweedie promotion rule (D13) now reads the cross-validation table instead of the single holdout — on this run they disagree in direction (holdout: Tweedie WAPE 12m 0.704 vs. current 0.711, an apparent improvement; cross-validated: Tweedie 0.663 vs. current 0.660, not an improvement), which is exactly the single-split noise D16 exists to route around. Current model stays shipped either way (Tweedie only clears the 0.02 gate on the holdout number, which no longer decides it).
- `aws-lambda/score.py`: loads `spend_retransform.json` from S3 alongside the two boosters; re-implements `expected_spend`/`summarize` in pure numpy (no `bvr` import); `/health` now actually loads the models and retransform file and returns 503 with the failure reason instead of a shallow 200; response gains `p10_6/p50_6/p90_6/p10_12/p50_12/p90_12`, `ltv_6`/`ltv_12` are now the corrected means. `tests/test_score_parity.py` checks the pure-numpy reimplementation against `bvr.predictive.summarize` on 20 real holdout customers — skipped in CI (needs a local `bvr all` run's checkpoints, which are gitignored), runs locally.
- `canary.py`: gates on `wape_12m_p50` using the endpoint's `p50_12` (P50 is the point forecast, D15), not `ltv_12`; also logs the signed revenue error of `ltv_12` (the mean) over the 300 canary rows. Nothing in `aws-lambda/` was deployed or touched on AWS itself — this is a source change only, `.github/workflows/deploy-lambda.yml` only runs on merge to `main`.
- `metrics.json`: added `wape_{6,12}m_{p50,mean}`, `revenue_error_{6,12}m_mean`, `interval_coverage_12m_p10_p90`, `wape_12m_p50_ci95` (bootstrap, 1,000 resamples, seed 26), `cv` (the section D table), `point_forecast: "p50"`; kept `wape_6m`/`wape_12m` as aliases of the `_mean` fields for one release; removed `mape_6m_actual_gt0`/`mape_12m_actual_gt0` (means in the trillions from GBP 0.01 actuals — median APE was already reported alongside them and stays).

### Task: cross-fitted retransformation, P10/P50/P90 range, cross-validated evaluation

- STATUS: done.
- BUILT: `retransform.py` (new), `predictive.py` (new); `monetary_model.predict_spend` now takes the retransform dict instead of a scalar smearing factor; `survival_model.fit`/`monetary_model.fit`/`monetary_model.fit_tweedie` gained a `save` parameter; `cli.step_train` fits and writes `outputs/spend_retransform.json`; `cli.step_eval` builds the per-customer predictive table and runs `eval.cross_validate`; `eval.py` gained `cross_validate`, `_revenue_error`, `bootstrap_wape_ci`, and new `evaluate()` fields; `canary.py` gates on P50; `aws-lambda/score.py` gained a pure-numpy retransform/predictive reimplementation and a real `/health`. `tests/test_retransform.py`, `tests/test_predictive.py`, `tests/test_score_parity.py` (new); `tests/test_canary.py` updated for the new gate.
- TESTED: `python3 -m pytest -q` — 35 passed (23 previously + 12 new). `python3 -m ruff check .` — all checks passed. `python3 -m bvr.cli all --force` run end to end on the real, full dataset (~2m15s, fetch included); reproduced numbers above, all within ±0.001 of the issue's targets (best within ±0.0004).
- SPEC CHECK: no feature, split, seed, or LightGBM parameter changed (verified `split_ids.json` byte-identical, `best_iteration` 123/156 unchanged); nothing in `aws-lambda/` deployed or touched on AWS/S3; PR opened, not merged.
- OPEN: none from this task.
- TIME: 2026-09-24, ~00:47–01:15 UTC.

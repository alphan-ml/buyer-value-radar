# Buyer Value Radar

A two-curve model that predicts how much each repeat buyer will spend over the next 6 and 12 months. It is built and evaluated on the full public Online Retail II dataset, 1,067,371 transaction lines from a real UK-based wholesaler.

## Architecture

The model splits customer value into two questions and multiplies the answers together. Curve 1 (`survival_model.py`) is a LightGBM binary classifier that predicts the probability a customer is active in month `t` after the cutoff date. Curve 2 (`monetary_model.py`) is a LightGBM regressor, trained only on active customer-months, that predicts spend in month `t` given that the customer is active, fit on `log1p(net revenue)` with a Duan smearing correction on the back-transform -- the smearing factor is **cross-fitted, per prediction level** (`retransform.py`, CONTEXT.md D14), not computed in-sample. `combine.py` multiplies the two curves month by month and sums the result out to 6 and 12 months for the mean: `LTV_h = sum_{t=1..h} P(active_t) * E[spend_t | active]`. `predictive.py` draws from the model's own predictive distribution (per-month Bernoulli-active x log-scale residual) to also return P10/P50/P90 for each horizon; **P50 is the point forecast for a single customer** (CONTEXT.md D15) -- WAPE is minimised by the median, not the mean, which the mean-only forecast used to report. Every feature is computed as of a single cutoff date (T0 = Dec 9, 2010); nothing after that date is visible to either model, and `tests/test_no_leak.py` checks this directly.

## How to run

```bash
git clone <this repo> && cd buyer-value-radar
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"
python3 -m bvr.cli all          # fetch -> check -> features -> train -> eval -> export
python3 -m pytest -q
python3 -m ruff check .
```

Each step is resumable: rerunning `bvr all` skips any step whose `outputs/checkpoints/<step>.done` marker exists; pass `--force` to redo a step (or all of them).

## Example: one customer, one prediction

A holdout customer with `recency_days=14`, `frequency=6`, `tenure_days=310`, `monetary_total=612.40` (GBP), `country_group="United Kingdom"`, `buyer_size` in the top tercile gets a 12-month predicted value in the hundreds of GBP, built from twelve separate (P(active_t), E[spend_t]) pairs rather than a single number — the per-month detail is what `combine()` returns alongside the summed `ltv_6` / `ltv_12` columns, and is what the calibration chart is built from.

## Design decisions

1. **LightGBM over `lifetimes` or `pymc-marketing` for the main model.** `lifetimes`'s last release was 0.11.3 in Jul 2020 (unmaintained). `pymc-marketing` (BG/NBD + Gamma-Gamma) needs Python >= 3.12, and a Bayesian fit does not fit the timebox for this build. A BG/NBD + Gamma-Gamma baseline was logged as **not built** rather than faked, per the pack's rule against invented numbers.
2. **log1p + Duan smearing for the spend model, not raw-scale regression.** A small number of very large wholesale buyers dominate squared-error loss on the raw scale and starve the model of signal on everyone else. The smearing factor (1.1541, from this run) corrects the back-transform bias that a naive `expm1()` would otherwise introduce.
3. **Mean MAPE is reported alongside median APE.** BUILD PACK Section 6 asks for MAPE over `actual > 0` customers. In this data, a handful of holdout customers have an actual 6- or 12-month net revenue that is a tiny positive residual of cancellation netting (a few pence), and one such row can send the mean into the trillions of percent. The median is reported next to it so the number on the page is not misleading, rather than silently dropping or winsorizing a real value.
4. **StockCode "B" ("Adjust bad debt") added to the non-product code list.** Found during this build: one raw row (Invoice A506401) is a GBP -53,594.36 bookkeeping write-off, not a cancellation (its invoice number does not start with "C") and not a product sale. It is excluded the same way postage and bank-charge lines are.

## Data provenance

Dataset: Online Retail II, UCI Machine Learning Repository id 502 (https://archive.ics.uci.edu/dataset/502/online+retail+ii), CC BY 4.0. Downloaded zip sha256 `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb` (45,622,418 bytes), pulled 2026-09-13. 1,067,371 rows across both year sheets (525,461 + 541,910), Dec 2009 -- Dec 2011. 243,007 rows (22.8%) have no Customer ID and are excluded from the model; they are counted, not hidden. Full attribution: `data/ATTRIBUTION.md`.

## Results

**Headline: 5-fold x 3-repeat customer cross-validation over all 4,314 customers** (CONTEXT.md D16), not the single 30% holdout below -- the holdout's bootstrap 95% interval on 12-month WAPE is 0.58 to 0.79, wide enough that its point estimate is not trustworthy on its own, and a single lucky/unlucky split can hide a real change (see "Why cross-validation" below). Each of the 15 folds refits Curve 1, Curve 2, and the cross-fitted retransform exactly as production, from scratch, only on that fold's training customers.

| Metric | Mean over 15 folds | SD |
|---|---|---|
| WAPE, 12-month, P50 (point forecast) | 0.635 | 0.067 |
| WAPE, 12-month, mean (for totals) | 0.660 | 0.073 |
| WAPE, 6-month, P50 | 0.686 | 0.063 |
| WAPE, 6-month, mean | 0.752 | 0.072 |
| Revenue error, 12-month mean, signed | -0.043 | 0.105 |
| Revenue error, 12-month mean, old (in-sample smearing) | -0.151 | 0.095 |
| Spearman (predicted vs actual, 12m) | 0.630 | 0.018 |
| Top-decile capture (12m) | 0.613 | 0.057 |
| P10-P90 coverage (12m) | 0.794 | 0.015 |

The old (in-sample smearing) method under-forecasts total 12-month revenue by 15.1% on average across folds; the cross-fitted method (CONTEXT.md D14) cuts that to 4.3%. Full per-metric table: `outputs/metrics.json` -> `cv`.

### Why cross-validation, not just the 30% holdout

The single 30% holdout (1,295 customers) is still reported below, unchanged in shape, but it no longer gates anything (the Tweedie promotion rule, D13, now reads the cross-validation table above). Two numbers explain why: its bootstrap 95% interval on 12-month WAPE of P50 is [0.576, 0.795] -- wider than the entire gap between "current" and "Tweedie" below -- and on this particular split the current model's 12-month revenue error is **+8.5%** (over-forecast), while the 15-fold cross-validated average is **-4.3%** (under-forecast). Both numbers are real; the holdout capitalised on one lucky split. `outputs/metrics.json` -> `wape_12m_p50_ci95`.

## Holdout results (30% holdout, seed 26, 1,295 customers)

| Metric | Value |
|---|---|
| WAPE, 6-month, P50 | 0.768 |
| WAPE, 6-month, mean | 0.852 |
| WAPE, 12-month, P50 (point forecast) | 0.672 |
| WAPE, 12-month, mean (for totals) | 0.711 |
| Revenue error, 12-month mean, signed | +0.085 |
| Revenue error, 6-month mean, signed | +0.077 |
| P10-P90 coverage, 12-month | 0.793 |
| Median APE, 6-month (actual > 0, n=567) | 0.510 |
| Median APE, 12-month (actual > 0, n=797) | 0.520 |
| Curve 2 (spend model) WAPE, active rows only | 0.534 |
| Curve 1 (activity model) ROC-AUC, by month | 0.757 -- 0.811 |

By segment (12-month, holdout, mean-based), two numbers per segment, because they answer different questions: **WAPE** is customer-level (each customer's miss, summed, over the segment's actual revenue); **aggregate revenue error** is the miss on the segment total, where over- and under-predictions cancel. United Kingdom WAPE 0.701, aggregate error 0.103 (1,193 customers); Europe 0.713 / 0.097 (88); rest of world 1.507 / 0.641 (14). By buyer-size tercile: high-value 0.631 / 0.095, mid 1.003 / 0.096, low 1.267 / 0.095. Read together: segment totals land close, individual customers do not, and small buyers are the hardest to call (small denominators, and a few of them churn or return unpredictably). Full segment and decile tables, both columns (`wape`, `aggregate_revenue_error`): `outputs/metrics.json`.

Full metrics, calibration by month, decile table, and both segment tables are in `outputs/metrics.json` and `outputs/calibration.json` -- committed, produced by `bvr eval`, never typed by hand. `metrics.json` keeps `wape_6m` / `wape_12m` as aliases of the `_mean` fields for one release so nothing reading the old names breaks; `"point_forecast": "p50"` marks which number to show for a single customer.

## Baselines and ranking metrics

Two naive baselines are computed on the same 1,295 holdout customers, the same 6- and 12-month windows: **repeat** (each customer's own trailing spend -- `monetary_total`, roughly the twelve months before the cutoff -- carried forward, halved for the 6-month window) and **segment mean** (each customer's buyer-size tercile, with both the tercile edges and the segment's mean spend computed from the training set only, applied to holdout customers). Definitions: CONTEXT.md D10/D11. The model's column is the mean (`wape_*_mean`), for comparability with the baselines, which only produce one number each.

| Metric | Model (mean) | Repeat baseline | Segment-mean baseline |
|---|---|---|---|
| WAPE, 6-month | 0.852 | 1.251 | 1.222 |
| WAPE, 12-month | 0.711 | 0.902 | 1.124 |
| Spearman (predicted vs actual, 12m) | 0.612 | 0.600 | 0.573 |
| Top-decile capture (12m) | 0.537 | 0.536 | 0.231 |

The model beats both naive baselines on every metric here. Full baseline decile tables: `outputs/metrics.json` -> `baselines.repeat` / `baselines.segment_mean`.

## Spend-model variant: Tweedie

`monetary_model.py` also trains a Tweedie-objective LightGBM regressor directly on raw net revenue (no log1p/smearing -- the Tweedie log-link mean is already on the natural scale), with the variance power tuned on the validation split over `{1.2, 1.5, 1.8}` (chosen: 1.2). Promotion rule (CONTEXT.md D13, updated by D16): ship the Tweedie variant only if the **cross-validated** (section D, not the single holdout) 12-month WAPE of the mean improves by at least 0.02 **and** cross-validated top-decile capture does not fall; otherwise keep the current (log1p + Duan smearing, cross-fitted) model. The single holdout's comparison is still computed and recorded for reference, but no longer gates the decision -- on this build's holdout split, Tweedie looks like an improvement (WAPE 12m 0.704 vs. current 0.711, top-decile capture 0.550 vs. 0.537), the opposite of what the more reliable cross-validated comparison shows, which is exactly the kind of single-split noise D16 exists to route around.

Latest run: cross-validated current WAPE 12m mean 0.660 vs. Tweedie 0.663 (worse, not an improvement) -- **current model stays shipped**. Full comparison: `outputs/metrics.json` -> `spend_model_variants` (holdout, reference only) and `outputs/metrics.json` -> `cv.metrics.wape_12m_tweedie` / `cv.metrics.top_decile_capture_12m_tweedie` (the numbers that actually gate).

## Live Eval canary

A scheduled job (`.github/workflows/canary.yml`, cron `17 */6 * * *` plus manual `workflow_dispatch`) calls the LIVE scoring endpoint (`https://guijt78nlb.execute-api.us-east-1.amazonaws.com/score`) once for each of the 300 customers in `canary/rows.json` -- a fixed sample drawn from the 30% holdout split (`outputs/split_ids.json`), with the real 6- and 12-month spend already recorded next to each request. `tests/test_canary.py` checks every one of those 300 ids is actually in the holdout split.

`src/bvr/canary.py` computes WAPE at 12 months over the endpoint's live `p50_12` responses and compares it to the recorded `wape_12m_p50` in `outputs/metrics.json`, within a tolerance of 0.060 -- P50 is the point forecast (D15), so the canary gates on it, not on `ltv_12` (the mean). `wape_6m_p50` is computed and recorded too, alongside the recorded value, but does not gate the match; the signed revenue error of `ltv_12` (the mean) over the 300 rows is also logged, unsurprising since the mean is meant for totals, not the canary's per-customer gate. There are no fallback numbers: if a request to the endpoint errors, that counts as an error and forces `match=false` -- it never falls back to a guessed or cached score.

Each run appends one JSON record to `ledger/runs.jsonl` on the `ledger` branch of this repo and overwrites `ledger/latest.json` with the same record. giggitai.com's Live Eval tab reads both files straight from `raw.githubusercontent.com` and replays them as a terminal. `match: true` means the live endpoint's predictions are still within tolerance of the offline holdout metric; `match: false` means either the endpoint errored or its predictions have drifted from what was recorded at training time -- worth checking the Lambda deployment (`aws-lambda/score.py`, `.github/workflows/deploy-lambda.yml`) before trusting the endpoint for anything else.

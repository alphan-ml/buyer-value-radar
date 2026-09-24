"""bvr fetch | check | features | train | eval | export | all

Each step is resumable: it skips when outputs/checkpoints/<step>.done
exists, unless --force is passed. One line per step, with row counts and
wall time, to stdout and outputs/run.log (BUILD PACK Section 7).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from bvr import check as check_mod
from bvr import clean as clean_mod
from bvr import combine as combine_mod
from bvr import eval as eval_mod
from bvr import features as features_mod
from bvr import fetch as fetch_mod
from bvr import monetary_model, predictive, retransform, survival_model
from bvr.features import CATEGORICAL_FEATURES

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "outputs" / "checkpoints"
OUT_DIR = ROOT / "outputs"
LOG_PATH = OUT_DIR / "run.log"


def _log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def _done(step: str) -> bool:
    return (CKPT_DIR / f"{step}.done").exists()


def _mark_done(step: str) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    (CKPT_DIR / f"{step}.done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")


def step_fetch(force: bool = False) -> None:
    if _done("fetch") and not force:
        _log("fetch: skipped (checkpoint present)")
        return
    t0 = time.time()
    fetch_mod.fetch(force=force)
    _log(f"fetch: done in {time.time() - t0:.1f}s")
    _mark_done("fetch")


def step_check(force: bool = False) -> None:
    if _done("check") and not force:
        _log("check: skipped (checkpoint present)")
        return
    t0 = time.time()
    r = check_mod.check()
    _log(f"check: {r['row_count']:,} rows, done in {time.time() - t0:.1f}s")
    _mark_done("check")


def step_features(force: bool = False) -> None:
    if _done("features") and not force:
        _log("features: skipped (checkpoint present)")
        return
    t0 = time.time()
    raw = clean_mod.load_raw()
    cleaned, clean_stats = clean_mod.clean(raw)
    feat = features_mod.build_features(cleaned)
    targets = features_mod.build_targets(cleaned, feat["Customer ID"].tolist())
    train_ids, holdout_ids = features_mod.train_holdout_split(feat)

    feat.to_parquet(OUT_DIR / "features_t0.parquet", index=False)
    targets.to_parquet(OUT_DIR / "targets_monthly.parquet", index=False)
    (OUT_DIR / "clean_stats.json").write_text(json.dumps(clean_stats, indent=2))
    (OUT_DIR / "split_ids.json").write_text(
        json.dumps(
            {
                "train_ids": sorted(int(c) for c in train_ids),
                "holdout_ids": sorted(int(c) for c in holdout_ids),
                "seed": 26,
            },
            indent=2,
        )
    )
    _log(
        f"features: {len(feat):,} customers, {len(targets):,} customer-month rows, "
        f"train={len(train_ids):,} holdout={len(holdout_ids):,}, "
        f"done in {time.time() - t0:.1f}s"
    )
    _mark_done("features")


def _load_features_and_targets():
    feat = pd.read_parquet(OUT_DIR / "features_t0.parquet")
    targets = pd.read_parquet(OUT_DIR / "targets_monthly.parquet")
    split = json.loads((OUT_DIR / "split_ids.json").read_text())
    return feat, targets, set(split["train_ids"]), set(split["holdout_ids"])


def step_train(force: bool = False) -> None:
    if _done("train") and not force:
        _log("train: skipped (checkpoint present)")
        return
    t0 = time.time()
    feat, targets, train_ids, _holdout_ids = _load_features_and_targets()
    rows = features_mod.assemble_rows(feat, targets)

    _booster1, info1 = survival_model.fit(rows, train_ids)
    _log(
        f"train curve1: best_iter={info1['best_iteration']} "
        f"fit_customers={info1['fit_customers']} es_customers={info1['early_stop_customers']}"
    )

    _booster2, smearing_in_sample, info2 = monetary_model.fit(rows, train_ids)
    _log(
        f"train curve2: best_iter={info2['best_iteration']} "
        f"smearing_in_sample={smearing_in_sample:.4f} "
        f"fit_customers={info2['fit_customers']} es_customers={info2['early_stop_customers']}"
    )

    train_rows_active = rows[rows["Customer ID"].isin(train_ids) & (rows["active"] == 1)]
    rt = retransform.fit_retransform(
        train_rows_active,
        monetary_model.FEATURE_COLS,
        CATEGORICAL_FEATURES,
        monetary_model.LGB_PARAMS,
        num_boost_round=info2["best_iteration"],
        seed=26,
    )
    (OUT_DIR / "spend_retransform.json").write_text(json.dumps(rt, indent=2))
    _log(
        f"train retransform: global_smearing={rt['global_smearing']:.4f} "
        f"n_rows={rt['n_rows']:,} n_customers={rt['n_customers']:,}"
    )

    _booster2t, variance_power, info2t = monetary_model.fit_tweedie(rows, train_ids)
    _log(
        f"train curve2 tweedie: variance_power={variance_power} best_iter={info2t['best_iteration']} "
        f"fit_customers={info2t['fit_customers']} es_customers={info2t['early_stop_customers']}"
    )

    (OUT_DIR / "train_info.json").write_text(
        json.dumps({"curve1": info1, "curve2": info2, "curve2_tweedie": info2t}, indent=2)
    )
    _log(f"train: done in {time.time() - t0:.1f}s")
    _mark_done("train")


def _spend_variant_summary(ltv_df: pd.DataFrame, actual_6: pd.Series, actual_12: pd.Series) -> dict:
    m = (
        ltv_df.set_index("Customer ID")
        .join(actual_6.rename("actual_6"))
        .join(actual_12.rename("actual_12"))
        .fillna(0.0)
    )
    return {
        "wape_6m": eval_mod.wape(m["ltv_6"].to_numpy(), m["actual_6"].to_numpy()),
        "wape_12m": eval_mod.wape(m["ltv_12"].to_numpy(), m["actual_12"].to_numpy()),
        "top_decile_capture_12m": eval_mod.top_decile_capture(
            m["ltv_12"].to_numpy(), m["actual_12"].to_numpy()
        ),
    }


def _predictive_table(p_active: pd.DataFrame, pred_log: pd.DataFrame, rt: dict) -> pd.DataFrame:
    """p_active / pred_log: (Customer ID, t, value) frames. Returns one row
    per customer: mean_6, p10_6, p50_6, p90_6, mean_12, p10_12, p50_12, p90_12
    (bvr.predictive.summarize, seed always 26)."""
    p_active_wide = p_active.pivot(index="Customer ID", columns="t", values="value").sort_index(axis=1)
    pred_log_wide = pred_log.pivot(index="Customer ID", columns="t", values="value").sort_index(axis=1)
    records = []
    for cust in p_active_wide.index:
        summary = predictive.summarize(
            p_active_wide.loc[cust].to_numpy(), pred_log_wide.loc[cust].to_numpy(), rt,
            horizons=(6, 12), seed=26,
        )
        records.append(
            {
                "Customer ID": cust,
                "mean_6": summary[6]["mean"], "p10_6": summary[6]["p10"],
                "p50_6": summary[6]["p50"], "p90_6": summary[6]["p90"],
                "mean_12": summary[12]["mean"], "p10_12": summary[12]["p10"],
                "p50_12": summary[12]["p50"], "p90_12": summary[12]["p90"],
            }
        )
    return pd.DataFrame.from_records(records)


def step_eval(force: bool = False) -> None:
    if _done("eval") and not force:
        _log("eval: skipped (checkpoint present)")
        return
    t0 = time.time()
    feat, targets, train_ids, holdout_ids = _load_features_and_targets()
    rows = features_mod.assemble_rows(feat, targets)
    holdout_rows = rows[rows["Customer ID"].isin(holdout_ids)]
    train_rows = rows[rows["Customer ID"].isin(train_ids)]

    booster1 = survival_model.load()
    booster2 = monetary_model.load()
    booster2_tweedie = monetary_model.load_tweedie()
    train_info = json.loads((OUT_DIR / "train_info.json").read_text())
    variance_power = train_info["curve2_tweedie"]["variance_power"]
    rt = json.loads((OUT_DIR / "spend_retransform.json").read_text())

    p_active = holdout_rows[["Customer ID", "t"]].copy()
    p_active["value"] = survival_model.predict_proba(booster1, holdout_rows)

    pred_log = holdout_rows[["Customer ID", "t"]].copy()
    pred_log["value"] = booster2.predict(
        holdout_rows[monetary_model.FEATURE_COLS], num_iteration=booster2.best_iteration
    )

    e_spend_current = holdout_rows[["Customer ID", "t"]].copy()
    e_spend_current["value"] = monetary_model.predict_spend(booster2, rt, holdout_rows)

    e_spend_tweedie = holdout_rows[["Customer ID", "t"]].copy()
    e_spend_tweedie["value"] = monetary_model.predict_spend_tweedie(booster2_tweedie, holdout_rows)

    ltv_current, _ = combine_mod.combine(p_active, e_spend_current, horizons=(6, 12))
    ltv_tweedie, _ = combine_mod.combine(p_active, e_spend_tweedie, horizons=(6, 12))

    actual_6 = holdout_rows[holdout_rows["t"] <= 6].groupby("Customer ID")["net_revenue"].sum()
    actual_12 = holdout_rows.groupby("Customer ID")["net_revenue"].sum()

    summary_current = _spend_variant_summary(ltv_current, actual_6, actual_12)
    summary_tweedie = _spend_variant_summary(ltv_tweedie, actual_6, actual_12)
    summary_tweedie["variance_power"] = variance_power

    # Section D cross-validation over ALL customers -- the headline
    # evaluation (CONTEXT.md D16). The Tweedie promotion decision (D13)
    # is gated on this, not the single 30% holdout, whose bootstrap 95%
    # interval on 12-month WAPE is wide enough to hide a real gap the
    # other way (CONTEXT.md D14/D16).
    all_customer_ids = train_ids | holdout_ids
    cv_results = eval_mod.cross_validate(rows, all_customer_ids)
    cv_metrics = cv_results["metrics"]

    wape_improvement = cv_metrics["wape_12m_mean"]["mean"] - cv_metrics["wape_12m_tweedie"]["mean"]
    capture_holds = (
        cv_metrics["top_decile_capture_12m_tweedie"]["mean"]
        >= cv_metrics["top_decile_capture_12m"]["mean"]
    )
    promote_tweedie = wape_improvement >= 0.02 and capture_holds
    promoted = "tweedie" if promote_tweedie else "current"
    _log(
        f"eval: spend model comparison (cross-validated) current wape_12m_mean="
        f"{cv_metrics['wape_12m_mean']['mean']:.4f} "
        f"tweedie wape_12m_mean={cv_metrics['wape_12m_tweedie']['mean']:.4f} (vp={variance_power}) "
        f"improvement={wape_improvement:.4f} capture_holds={capture_holds} promoted={promoted}"
    )

    e_spend, ltv = (e_spend_tweedie, ltv_tweedie) if promote_tweedie else (e_spend_current, ltv_current)
    predictive_pred = _predictive_table(p_active, pred_log, rt)

    metrics, _calibration = eval_mod.evaluate(
        holdout_rows, p_active, e_spend, ltv, predictive_pred, feat, train_rows, train_ids, cv_results
    )
    metrics["spend_model_variants"] = {
        "current": summary_current,
        "tweedie": summary_tweedie,
        "promoted": promoted,
        "promotion_rule": (
            "promote tweedie only if cross-validated (section D) 12m WAPE of the mean improves "
            "by >= 0.02 and cross-validated top_decile_capture_12m does not fall; the single "
            "holdout's numbers (spend_model_variants.current/tweedie above) are kept for "
            "reference only and no longer gate the decision (CONTEXT.md D13/D16)"
        ),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    _log(
        f"eval: wape_6m_mean={metrics['wape_6m_mean']:.4f} wape_12m_mean={metrics['wape_12m_mean']:.4f} "
        f"wape_12m_p50={metrics['wape_12m_p50']:.4f} "
        f"holdout_customers={metrics['holdout_customers']:,}, done in {time.time() - t0:.1f}s"
    )
    _mark_done("eval")


def step_export(force: bool = False) -> None:
    from bvr import export as export_mod

    if _done("export") and not force:
        _log("export: skipped (checkpoint present)")
        return
    t0 = time.time()
    export_mod.export()
    _log(f"export: done in {time.time() - t0:.1f}s")
    _mark_done("export")


STEPS = {
    "fetch": step_fetch,
    "check": step_check,
    "features": step_features,
    "train": step_train,
    "eval": step_eval,
    "export": step_export,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="bvr")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in list(STEPS.keys()) + ["all"]:
        p = sub.add_parser(name)
        p.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.command == "all":
        for step_fn in STEPS.values():
            step_fn(force=args.force)
    else:
        STEPS[args.command](force=args.force)


if __name__ == "__main__":
    main()

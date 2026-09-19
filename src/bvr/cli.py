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
from bvr import monetary_model, survival_model

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

    _booster2, smearing, info2 = monetary_model.fit(rows, train_ids)
    _log(
        f"train curve2: best_iter={info2['best_iteration']} smearing={smearing:.4f} "
        f"fit_customers={info2['fit_customers']} es_customers={info2['early_stop_customers']}"
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
    smearing = train_info["curve2"]["smearing_factor"]
    variance_power = train_info["curve2_tweedie"]["variance_power"]

    p_active = holdout_rows[["Customer ID", "t"]].copy()
    p_active["value"] = survival_model.predict_proba(booster1, holdout_rows)

    e_spend_current = holdout_rows[["Customer ID", "t"]].copy()
    e_spend_current["value"] = monetary_model.predict_spend(booster2, smearing, holdout_rows)

    e_spend_tweedie = holdout_rows[["Customer ID", "t"]].copy()
    e_spend_tweedie["value"] = monetary_model.predict_spend_tweedie(booster2_tweedie, holdout_rows)

    ltv_current, _ = combine_mod.combine(p_active, e_spend_current, horizons=(6, 12))
    ltv_tweedie, _ = combine_mod.combine(p_active, e_spend_tweedie, horizons=(6, 12))

    actual_6 = holdout_rows[holdout_rows["t"] <= 6].groupby("Customer ID")["net_revenue"].sum()
    actual_12 = holdout_rows.groupby("Customer ID")["net_revenue"].sum()

    summary_current = _spend_variant_summary(ltv_current, actual_6, actual_12)
    summary_tweedie = _spend_variant_summary(ltv_tweedie, actual_6, actual_12)
    summary_tweedie["variance_power"] = variance_power

    wape_improvement = summary_current["wape_12m"] - summary_tweedie["wape_12m"]
    capture_holds = (
        summary_tweedie["top_decile_capture_12m"] >= summary_current["top_decile_capture_12m"]
    )
    promote_tweedie = wape_improvement >= 0.02 and capture_holds
    promoted = "tweedie" if promote_tweedie else "current"
    _log(
        f"eval: spend model comparison current wape_12m={summary_current['wape_12m']:.4f} "
        f"tweedie wape_12m={summary_tweedie['wape_12m']:.4f} (vp={variance_power}) "
        f"improvement={wape_improvement:.4f} capture_holds={capture_holds} promoted={promoted}"
    )

    e_spend, ltv = (e_spend_tweedie, ltv_tweedie) if promote_tweedie else (e_spend_current, ltv_current)

    metrics, _calibration = eval_mod.evaluate(
        holdout_rows, p_active, e_spend, ltv, feat, train_rows, train_ids
    )
    metrics["spend_model_variants"] = {
        "current": summary_current,
        "tweedie": summary_tweedie,
        "promoted": promoted,
        "promotion_rule": (
            "promote tweedie only if holdout wape_12m improves by >= 0.02 "
            "and top_decile_capture_12m does not fall; otherwise keep current"
        ),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    _log(
        f"eval: wape_6m={metrics['wape_6m']:.4f} wape_12m={metrics['wape_12m']:.4f} "
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

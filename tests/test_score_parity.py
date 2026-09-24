"""Endpoint / offline-pipeline parity (BUILD PACK Section 5 follow-up).

`aws-lambda/score.py` re-implements `bvr.retransform.expected_spend` and
`bvr.predictive.summarize` in pure numpy so the Lambda package does not need
to bundle the training-side dependencies. This test loads both
implementations against the same real, trained artifacts (the committed
`outputs/spend_retransform.json` and the two boosters) and checks they give
identical output on 20 real holdout customers -- no network, no S3.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bvr import features as features_mod
from bvr import monetary_model, predictive, retransform, survival_model

ROOT = Path(__file__).resolve().parents[1]
SCORE_PATH = ROOT / "aws-lambda" / "score.py"


def _load_score_module():
    spec = importlib.util.spec_from_file_location("score", SCORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _have_trained_artifacts() -> bool:
    return (
        (ROOT / "outputs" / "spend_retransform.json").exists()
        and (ROOT / "outputs" / "checkpoints" / "curve1_activity.txt").exists()
        and (ROOT / "outputs" / "checkpoints" / "curve2_spend.txt").exists()
        and (ROOT / "outputs" / "split_ids.json").exists()
        and (ROOT / "outputs" / "features_t0.parquet").exists()
    )


pytestmark = pytest.mark.skipif(
    not _have_trained_artifacts(),
    reason="requires a local `bvr all` run's outputs/ (checkpoints, split_ids, retransform file)",
)


def test_score_py_matches_bvr_predictive_on_holdout_customers():
    score = _load_score_module()

    rt = json.loads((ROOT / "outputs" / "spend_retransform.json").read_text())
    split = json.loads((ROOT / "outputs" / "split_ids.json").read_text())
    holdout_ids = sorted(split["holdout_ids"])[:20]

    feat = pd.read_parquet(ROOT / "outputs" / "features_t0.parquet")
    targets = pd.read_parquet(ROOT / "outputs" / "targets_monthly.parquet")
    rows = features_mod.assemble_rows(feat, targets)
    rows = rows[rows["Customer ID"].isin(holdout_ids)]

    booster1 = survival_model.load()
    booster2 = monetary_model.load()

    for cust in holdout_ids:
        cust_rows = rows[rows["Customer ID"] == cust].sort_values("t")
        p_active = survival_model.predict_proba(booster1, cust_rows)
        pred_log = booster2.predict(
            cust_rows[monetary_model.FEATURE_COLS], num_iteration=booster2.best_iteration
        )

        e_spend_bvr = retransform.expected_spend(pred_log, rt)
        e_spend_score = score._expected_spend(pred_log, rt)
        np.testing.assert_allclose(e_spend_bvr, e_spend_score)

        summary_bvr = predictive.summarize(p_active, pred_log, rt, seed=26)
        summary_score = score._summarize(p_active, pred_log, rt, seed=26)
        for h in (6, 12):
            for key in ("mean", "p10", "p50", "p90"):
                assert summary_bvr[h][key] == pytest.approx(summary_score[h][key])

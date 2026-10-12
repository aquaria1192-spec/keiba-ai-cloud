from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
VALIDATION_FILE = BASE / "LTR_VALIDATION_V117.json"

LTR_FEATURES = [
    "distance","horse_no","gate","age","weight","body_weight","body_weight_diff",
    "recent_avg_finish","recent_top3_rate","recent_avg_margin","recent_avg_last3f_rank",
    "recent_avg_popularity","distance_top3_rate","course_top3_rate","surface_top3_rate",
    "jockey_top3_rate","trainer_top3_rate","gate_condition_top3_rate",
    "prev_finish","prev_margin","prev_last3f_rank","distance_change","days_since_last",
    "body_weight_change_from_prev","class_change","race_class_score","running_style_score",
    "odds","popularity","implied_prob","log_odds",
    "horse_going_top3_rate","jockey_course_top3_rate","jockey_surface_top3_rate",
    "jockey_distance_top3_rate","jockey_going_top3_rate","jockey_trainer_top3_rate",
    "trainer_course_top3_rate","trainer_surface_top3_rate","trainer_distance_top3_rate",
    "field_size","market_prob_norm","market_rank_pct","popularity_pct",
    "recent_top3_rate_rank_pct","jockey_top3_rate_rank_pct","trainer_top3_rate_rank_pct",
    "distance_top3_rate_rank_pct","course_top3_rate_rank_pct","surface_top3_rate_rank_pct",
]


def load_validation_report(path: Path = VALIDATION_FILE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relevance_labels(finish):
    f = pd.to_numeric(pd.Series(finish), errors="coerce")
    return np.select(
        [f.eq(1), f.eq(2), f.eq(3), f.le(5)],
        [4, 3, 2, 1],
        default=0,
    ).astype(np.int32)


def sorted_qid(frame: pd.DataFrame):
    if "race_id" not in frame.columns:
        raise ValueError("Learning-to-Rankには race_id が必要です。")
    d = frame.sort_values(["race_id", "horse_no"]).reset_index(drop=True)
    qid = d["race_id"].astype("category").cat.codes.to_numpy()
    return d, qid


def train_ranker(history_features: pd.DataFrame, **overrides):
    """Train an XGBoost LambdaMART challenger.

    This is intentionally an offline/challenger function. Production imports
    do not require XGBoost; install requirements-ltr.txt when retraining.
    """
    try:
        from xgboost import XGBRanker
    except Exception as exc:
        raise RuntimeError(
            "LTR再学習には requirements-ltr.txt の xgboost が必要です。"
        ) from exc

    d, qid = sorted_qid(history_features)
    for col in LTR_FEATURES:
        if col not in d.columns:
            d[col] = np.nan
    x = d[LTR_FEATURES].apply(pd.to_numeric, errors="coerce")
    medians = x.median(numeric_only=True).fillna(0.0)
    x = x.fillna(medians).astype("float32")
    y = relevance_labels(d.get("finish"))

    params = dict(
        n_estimators=350,
        learning_rate=0.04,
        max_depth=4,
        min_child_weight=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="rank:ndcg",
        eval_metric="ndcg@3",
        tree_method="hist",
        lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=8,
        random_state=42,
        n_jobs=-1,
    )
    params.update(overrides)
    model = XGBRanker(**params)
    model.fit(x, y, qid=qid, verbose=False)
    return model, medians


def top_pick_metrics(frame: pd.DataFrame, scores):
    q = frame[["race_id", "finish"]].copy()
    q["score"] = np.asarray(scores, dtype=float)
    top = (
        q.sort_values(["race_id", "score"], ascending=[True, False])
        .groupby("race_id", sort=False)
        .head(1)
    )
    return {
        "races": int(top["race_id"].nunique()),
        "top_pick_win_rate": float((pd.to_numeric(top["finish"], errors="coerce") == 1).mean()),
        "top_pick_top3_rate": float((pd.to_numeric(top["finish"], errors="coerce") <= 3).mean()),
    }


def promotion_decision(report=None):
    r = report or load_validation_report()
    c = r["market_aware_candidate"]
    gate = r["promotion_gate"]

    val_win_lift = (
        float(c["challenger_2024"]["top_pick_win_rate"])
        - float(c["baseline_2024"]["top_pick_win_rate"])
    )
    val_top3_drop = (
        float(c["baseline_2024"]["top_pick_top3_rate"])
        - float(c["challenger_2024"]["top_pick_top3_rate"])
    )
    holdout_win_delta = (
        float(c["challenger_2025"]["top_pick_win_rate"])
        - float(c["baseline_2025"]["top_pick_win_rate"])
    )
    holdout_top3_delta = (
        float(c["challenger_2025"]["top_pick_top3_rate"])
        - float(c["baseline_2025"]["top_pick_top3_rate"])
    )

    checks = {
        "validation_win_lift": val_win_lift >= float(gate["minimum_validation_win_lift"]),
        "validation_top3": val_top3_drop <= float(gate["maximum_validation_top3_drop"]),
        "holdout_win": (
            holdout_win_delta >= 0
            if gate.get("require_holdout_win_non_worse", True)
            else True
        ),
        "holdout_top3": (
            holdout_top3_delta >= 0
            if gate.get("require_holdout_top3_non_worse", True)
            else True
        ),
    }
    promoted = all(checks.values())
    return {
        "promoted": promoted,
        "checks": checks,
        "validation_win_lift": val_win_lift,
        "validation_top3_delta": -val_top3_drop,
        "holdout_win_delta": holdout_win_delta,
        "holdout_top3_delta": holdout_top3_delta,
        "status": "promoted" if promoted else "not_promoted",
    }

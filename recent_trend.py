from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
VALIDATION_FILE = BASE / "TREND_VALIDATION_V118.json"
TREND_LAYER_VERSION = "recent-form-shadow-v1"


def load_validation_report(path: Path = VALIDATION_FILE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def promotion_decision(report=None):
    r = report or load_validation_report()
    b = r["baseline_2025"]
    c = r["candidate_2025"]
    gate = r["promotion_gate"]
    win_lift = float(c["top_pick_win_rate"]) - float(b["top_pick_win_rate"])
    top3_delta = float(c["top_pick_top3_rate"]) - float(b["top_pick_top3_rate"])
    checks = {
        "win_lift": win_lift >= float(gate["minimum_holdout_win_lift"]),
        "top3_guard": top3_delta >= -float(gate["maximum_holdout_top3_drop"]),
        "win_logloss": (
            float(c["win_logloss"]) <= float(b["win_logloss"])
            if gate.get("require_win_logloss_non_worse", True) else True
        ),
        "win_brier": (
            float(c["win_brier"]) <= float(b["win_brier"])
            if gate.get("require_win_brier_non_worse", True) else True
        ),
    }
    return {
        "promoted": all(checks.values()),
        "status": "promoted" if all(checks.values()) else "not_promoted",
        "checks": checks,
        "win_lift": win_lift,
        "top3_delta": top3_delta,
    }


def _num(d, col):
    if col not in d.columns:
        return pd.Series(np.nan, index=d.index, dtype=float)
    return pd.to_numeric(d[col], errors="coerce")


def attach_recent_trend(detail: pd.DataFrame) -> pd.DataFrame:
    """Attach explainable recent-form momentum without changing rank/probability.

    Positive momentum means the latest completed start was better than the
    leakage-safe recent average. The shadow layer is display/storage only until
    the historical + live promotion gate is satisfied.
    """
    if detail is None or len(detail) == 0:
        return detail.copy() if detail is not None else pd.DataFrame()

    d = detail.copy()
    finish_m = (_num(d, "recent_avg_finish") - _num(d, "prev_finish")).fillna(0.0)
    margin_m = (_num(d, "recent_avg_margin") - _num(d, "prev_margin")).fillna(0.0)
    last3f_m = (
        _num(d, "recent_avg_last3f_rank") - _num(d, "prev_last3f_rank")
    ).fillna(0.0)

    # Robust bounded display score. It is deliberately NOT used to change
    # win_prob/ranking_score while the challenger is unpromoted.
    score = (
        45.0 * np.tanh(finish_m / 4.0)
        + 30.0 * np.tanh(margin_m / 1.5)
        + 25.0 * np.tanh(last3f_m / 4.0)
    )
    score = np.clip(score, -100.0, 100.0)

    labels = np.where(
        score >= 20.0, "上昇",
        np.where(score <= -20.0, "下降", "横ばい")
    )
    d["trend_finish_momentum"] = finish_m
    d["trend_margin_momentum"] = margin_m
    d["trend_last3f_momentum"] = last3f_m
    d["trend_score"] = score
    d["trend_label"] = labels
    d["trend_layer_version"] = TREND_LAYER_VERSION
    d["trend_promoted"] = False
    d["trend_reason"] = [
        (
            f"前走着順差 {fm:+.1f} / 着差改善 {mm:+.2f} / "
            f"上がり順位差 {lm:+.1f}"
        )
        for fm, mm, lm in zip(finish_m, margin_m, last3f_m)
    ]
    return d

from __future__ import annotations

import math
import numpy as np
import pandas as pd

RANKING_LAYER_VERSION = "race-consensus-v1"

NON_MARKET_RANK_COLS = (
    "recent_top3_rate_rank_pct",
    "jockey_top3_rate_rank_pct",
    "trainer_top3_rate_rank_pct",
    "distance_top3_rate_rank_pct",
    "course_top3_rate_rank_pct",
    "surface_top3_rate_rank_pct",
)


def _num(df, col, default=np.nan):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _unit_strength(series):
    s = pd.to_numeric(series, errors="coerce")
    vmax = s.max()
    if pd.isna(vmax) or float(vmax) <= 0:
        return pd.Series(0.5, index=s.index, dtype=float)
    return (s / float(vmax)).clip(0.0, 1.0).fillna(0.5)


def _non_market_consensus(df):
    vals = []
    field = max(len(df), 1)
    neutral = 0.5
    for col in NON_MARKET_RANK_COLS:
        if col not in df.columns:
            continue
        rank_pct = _num(df, col)
        # Existing rank_pct is 1/field for the strongest horse and 1 for last.
        strength = (1.0 - rank_pct + 1.0 / field).clip(0.0, 1.0)
        vals.append(strength)
    if not vals:
        return pd.Series(neutral, index=df.index, dtype=float)
    return pd.concat(vals, axis=1).mean(axis=1, skipna=True).fillna(neutral)


def _confidence_block(df):
    if df.empty:
        return {
            "score": 0.0, "label": "D", "gap": 0.0,
            "entropy": 1.0, "reason": "予測データなし",
        }

    p = _num(df, "win_prob", 0.0).fillna(0.0).clip(lower=0.0)
    total = float(p.sum())
    if total > 0:
        p = p / total
    else:
        p = pd.Series(np.full(len(df), 1.0 / max(len(df), 1)), index=df.index)

    ranked = df.sort_values(
        ["ranking_score", "win_prob", "top3_prob", "ai_index"],
        ascending=False,
    )
    first = ranked.iloc[0]
    first_idx = first.name
    p1 = float(pd.to_numeric(pd.Series([first.get("win_prob")]), errors="coerce").fillna(0).iloc[0])
    p2 = 0.0
    if len(ranked) >= 2:
        p2 = float(pd.to_numeric(pd.Series([ranked.iloc[1].get("win_prob")]), errors="coerce").fillna(0).iloc[0])
    gap = max(0.0, p1 - p2)

    if len(p) <= 1:
        entropy = 0.0
    else:
        a = np.clip(p.to_numpy(dtype=float), 1e-12, 1.0)
        entropy = float(-(a * np.log(a)).sum() / math.log(len(a)))
        entropy = max(0.0, min(1.0, entropy))

    win_top = _num(df, "win_prob", 0.0).idxmax()
    top3_top = _num(df, "top3_prob", 0.0).idxmax()
    agreement = float(first_idx == win_top) + float(first_idx == top3_top)
    agreement /= 2.0

    score = (
        45.0 * min(max(p1, 0.0) / 0.40, 1.0)
        + 30.0 * min(gap / 0.12, 1.0)
        + 15.0 * (1.0 - entropy)
        + 10.0 * agreement
    )
    score = float(np.clip(score, 0.0, 100.0))
    if score >= 80:
        label = "S"
    elif score >= 65:
        label = "A"
    elif score >= 50:
        label = "B"
    elif score >= 35:
        label = "C"
    else:
        label = "D"

    reason = (
        f"◎勝率{p1*100:.1f}%・○との差{gap*100:.1f}pt・"
        f"予測集中度{(1.0-entropy)*100:.0f}%"
    )
    return {
        "score": score,
        "label": label,
        "gap": gap,
        "entropy": entropy,
        "reason": reason,
    }


def apply_race_ranking_layer(detail: pd.DataFrame) -> pd.DataFrame:
    """Conservative race-level reranking after all probability adjustments.

    The calibrated win probability remains dominant. The secondary layer can
    only move close contenders by using top-3 strength and non-market,
    race-relative form/condition consensus. Current odds/popularity are not
    directly used by this layer; they remain available to the betting layer.
    """
    if detail is None or len(detail) == 0:
        return detail.copy() if detail is not None else pd.DataFrame()

    d = detail.copy()
    win = _num(d, "win_prob", 0.0).fillna(0.0).clip(0.0, 1.0)
    top3_strength = _unit_strength(_num(d, "top3_prob", 0.0))
    consensus = _non_market_consensus(d)

    # Adjustment range is deliberately small (roughly +/- 1.5 percentage
    # points), so decisive win-probability differences are never overturned.
    adjustment = (
        0.018 * (top3_strength - 0.5)
        + 0.012 * (consensus - 0.5)
    )
    d["ranking_consensus"] = consensus
    d["ranking_adjustment"] = adjustment
    d["ranking_score"] = win + adjustment
    d["ranking_layer_version"] = RANKING_LAYER_VERSION

    original_top = win.idxmax()
    d = d.sort_values(
        ["ranking_score", "win_prob", "top3_prob", "ai_index"],
        ascending=False,
    ).reset_index(drop=False).rename(columns={"index": "_source_index"})
    final_source = d.iloc[0]["_source_index"] if len(d) else None
    d["ranking_changed_top"] = bool(final_source != original_top)

    conf = _confidence_block(d)
    d["confidence_score"] = conf["score"]
    d["confidence_label"] = conf["label"]
    d["confidence_gap"] = conf["gap"]
    d["confidence_entropy"] = conf["entropy"]
    d["confidence_reason"] = conf["reason"]

    return d.drop(columns=["_source_index"], errors="ignore").reset_index(drop=True)

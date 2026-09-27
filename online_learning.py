from __future__ import annotations

from datetime import datetime
from io import BytesIO
import os

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from batch_predict import mark_for_rank, mark_label
from ml_engine import FEATURE_COLS

LEARNING_ROWS_PATH = "automation_data/learning/pre_race_feature_rows.csv"
LEARNING_STATUS_PATH = "automation_data/learning/status.json"
ADAPTER_PATH = "automation_data/learning/online_adapter.joblib"
ADAPTER_METRICS_PATH = "automation_data/learning/adapter_metrics.json"
MIN_NEW_RACES = int(os.environ.get("KEIBA_LEARNING_MIN_RACES", "200"))
VALID_RATIO = float(os.environ.get("KEIBA_LEARNING_VALID_RATIO", "0.20"))
BASE_CHAMPION_VERSION = "base-1.12-calibrated"

ADAPTER_INPUTS = [
    "raw_win_prob",
    "raw_top3_prob",
    "market_prob_norm",
    "market_rank_pct",
    "popularity_pct",
    "recent_top3_rate_rank_pct",
    "jockey_top3_rate_rank_pct",
    "trainer_top3_rate_rank_pct",
    "distance_top3_rate_rank_pct",
    "course_top3_rate_rank_pct",
    "surface_top3_rate_rank_pct",
    "horse_going_top3_rate",
    "jockey_course_top3_rate",
    "jockey_surface_top3_rate",
    "jockey_distance_top3_rate",
    "jockey_going_top3_rate",
    "trainer_course_top3_rate",
    "trainer_surface_top3_rate",
    "trainer_distance_top3_rate",
    "field_size",
]
MODEL_INPUTS = [
    "logit_raw_win",
    "logit_raw_top3",
    "logit_market",
] + [c for c in ADAPTER_INPUTS if c not in {
    "raw_win_prob", "raw_top3_prob", "market_prob_norm"
}]

ID_COLS = [
    "snapshot_id", "recorded_at", "champion_version", "snapshot_type",
    "date", "course", "race_no", "race_name", "surface", "distance", "going",
    "horse_no", "horse_name", "jockey", "trainer",
    "true_date", "feature_source", "quality_flag", "missing_feature_count",
    "raw_win_prob", "raw_top3_prob", "raw_ai_index",
    "base_win_prob", "base_top3_prob", "base_ai_index", "pred_rank", "pred_mark",
    "actual_finish", "actual_win", "actual_top3", "settled_at", "result_url",
    "result_quality",
]
LEARNING_COLS = list(dict.fromkeys(ID_COLS + FEATURE_COLS))


def champion_version(adapter) -> str:
    if isinstance(adapter, dict) and str(adapter.get("version", "")).strip():
        return str(adapter["version"])
    return BASE_CHAMPION_VERSION


def blank_learning_rows():
    return pd.DataFrame(columns=LEARNING_COLS)


def normalize_learning_rows(df):
    if df is None or len(df) == 0:
        return blank_learning_rows()
    out = df.copy()
    for c in LEARNING_COLS:
        if c not in out.columns:
            out[c] = np.nan
    text_cols = {
        "snapshot_id", "recorded_at", "champion_version", "snapshot_type",
        "date", "course", "race_no", "race_name", "surface", "going",
        "horse_name", "jockey", "trainer", "feature_source", "quality_flag",
        "pred_mark", "settled_at", "result_url", "result_quality",
    }
    for c in text_cols:
        out[c] = out[c].fillna("").astype(str)
    return out[LEARNING_COLS]


def load_learning_rows(store):
    try:
        return normalize_learning_rows(store.read_csv(LEARNING_ROWS_PATH))
    except Exception:
        return blank_learning_rows()


def save_learning_rows(store, df, message):
    out = normalize_learning_rows(df)
    store.write_csv(LEARNING_ROWS_PATH, out, message)


def merge_learning_rows(*frames):
    xs = [normalize_learning_rows(x) for x in frames if x is not None and len(x)]
    if not xs:
        return blank_learning_rows()
    out = pd.concat(xs, ignore_index=True)
    return (
        out.drop_duplicates(["snapshot_id", "horse_no"], keep="last")
        .reset_index(drop=True)
    )


def _num_series(df, name):
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def learning_rows_from_detail(detail, snapshot, current_champion_version):
    if detail is None or len(detail) == 0 or snapshot is None or len(snapshot) == 0:
        return blank_learning_rows()

    d = detail.copy().reset_index(drop=True)
    snap = snapshot.copy()
    sid = str(snap.iloc[0].get("snapshot_id", ""))
    recorded_at = str(snap.iloc[0].get("recorded_at", ""))
    stype = str(snap.iloc[0].get("snapshot_type", ""))

    out = pd.DataFrame(index=d.index)
    out["snapshot_id"] = sid
    out["recorded_at"] = recorded_at
    out["champion_version"] = str(current_champion_version or BASE_CHAMPION_VERSION)
    out["snapshot_type"] = stype
    out["date"] = d.get("開催日", d.get("date", "")).astype(str)
    out["course"] = d.get("競馬場", d.get("course", "")).astype(str)
    out["race_no"] = d.get("レース", d.get("race_no", "")).astype(str)
    out["race_name"] = d.get("race_name", "").astype(str) if "race_name" in d.columns else ""
    out["surface"] = d.get("surface", "").astype(str) if "surface" in d.columns else ""
    out["distance"] = _num_series(d, "distance")
    out["going"] = d.get("going", "").astype(str) if "going" in d.columns else ""
    out["horse_no"] = _num_series(d, "horse_no")
    out["horse_name"] = d.get("horse_name", "").astype(str) if "horse_name" in d.columns else ""
    out["jockey"] = d.get("jockey", "").astype(str) if "jockey" in d.columns else ""
    out["trainer"] = d.get("trainer", "").astype(str) if "trainer" in d.columns else ""
    out["true_date"] = True
    out["feature_source"] = "automation_pre_race"
    # raw_* is the statically calibrated HistGB + race-day-adjustment output
    # before any online adapter. It is preserved across adapter generations so
    # a newly promoted adapter always
    # replaces the old correction instead of stacking on top of it.
    out["raw_win_prob"] = (
        _num_series(d, "raw_win_prob")
        if "raw_win_prob" in d.columns else _num_series(d, "win_prob")
    )
    out["raw_top3_prob"] = (
        _num_series(d, "raw_top3_prob")
        if "raw_top3_prob" in d.columns else _num_series(d, "top3_prob")
    )
    out["raw_ai_index"] = (
        _num_series(d, "raw_ai_index")
        if "raw_ai_index" in d.columns else _num_series(d, "ai_index")
    )
    # base_* is the probability actually deployed for this race after the
    # current champion adapter (or raw model when no adapter exists).
    out["base_win_prob"] = _num_series(d, "win_prob")
    out["base_top3_prob"] = _num_series(d, "top3_prob")
    out["base_ai_index"] = _num_series(d, "ai_index")
    out["pred_rank"] = _num_series(d, "順位")
    out["pred_mark"] = d.get("印", "").astype(str) if "印" in d.columns else ""

    for c in FEATURE_COLS:
        if c in d.columns:
            out[c] = d[c]
        elif c not in out.columns:
            out[c] = np.nan

    feature_view = out[[c for c in FEATURE_COLS if c in out.columns]].copy()
    for c in feature_view.columns:
        if c in {"course", "surface", "going", "sex", "running_style", "race_class"}:
            feature_view[c] = feature_view[c].replace("", np.nan)
    missing = feature_view.isna().sum(axis=1)
    out["missing_feature_count"] = missing.astype(int)
    out["quality_flag"] = np.where(missing <= 6, "ok", "feature_missing")
    out["actual_finish"] = np.nan
    out["actual_win"] = np.nan
    out["actual_top3"] = np.nan
    out["settled_at"] = ""
    out["result_url"] = ""
    out["result_quality"] = ""
    return normalize_learning_rows(out)


def learning_snapshot_needs_result(rows, snapshot_id):
    if rows is None or len(rows) == 0:
        return False
    q = rows[rows["snapshot_id"].astype(str) == str(snapshot_id)]
    if q.empty:
        return False
    return pd.to_numeric(q["actual_finish"], errors="coerce").notna().sum() < 3


def apply_learning_result(rows, snapshot_id, finish_map, result_url, settled_at):
    """
    Attach result labels only when the parsed result is complete enough for ML.

    This prevents a partial HTML parse (for example only the first three
    finishers) from turning every stored runner outside that fragment into an
    unknown/biased sample.
    """
    if rows is None or len(rows) == 0:
        return rows, False
    out = rows.copy()
    mask = out["snapshot_id"].astype(str) == str(snapshot_id)
    if not mask.any():
        return out, False

    snap = out.loc[mask].copy()
    predicted_n = int(pd.to_numeric(snap["horse_no"], errors="coerce").notna().sum())
    finishes = sorted(int(v) for v in finish_map.values() if pd.notna(v))
    matched_n = sum(
        1 for v in pd.to_numeric(snap["horse_no"], errors="coerce")
        if pd.notna(v) and int(v) in finish_map
    )
    minimum = max(5, int(np.ceil(predicted_n * 0.75)))
    complete = (
        predicted_n >= 5 and
        matched_n >= minimum and
        finishes.count(1) == 1 and
        sum(1 for x in finishes if x <= 3) >= 3
    )

    quality = "complete" if complete else "incomplete_result"
    out.loc[mask, "result_quality"] = quality
    if not complete:
        return normalize_learning_rows(out), True

    changed = False
    for idx in out.index[mask]:
        no = pd.to_numeric(
            pd.Series([out.at[idx, "horse_no"]]), errors="coerce"
        ).iloc[0]
        if pd.isna(no) or int(no) not in finish_map:
            continue
        fi = int(finish_map[int(no)])
        out.at[idx, "actual_finish"] = fi
        out.at[idx, "actual_win"] = 1 if fi == 1 else 0
        out.at[idx, "actual_top3"] = 1 if fi <= 3 else 0
        out.at[idx, "settled_at"] = settled_at
        out.at[idx, "result_url"] = str(result_url or "")
        changed = True
    return normalize_learning_rows(out), changed


def _adapter_frame(df):
    out = pd.DataFrame(index=df.index)
    bw = pd.to_numeric(df.get("raw_win_prob", df.get("base_win_prob")), errors="coerce").clip(1e-5, 1-1e-5)
    bt = pd.to_numeric(df.get("raw_top3_prob", df.get("base_top3_prob")), errors="coerce").clip(1e-5, 1-1e-5)
    mp = pd.to_numeric(df.get("market_prob_norm"), errors="coerce").clip(1e-5, 1-1e-5)
    out["logit_raw_win"] = np.log(bw / (1 - bw))
    out["logit_raw_top3"] = np.log(bt / (1 - bt))
    out["logit_market"] = np.log(mp / (1 - mp))
    for c in MODEL_INPUTS:
        if c in out.columns:
            continue
        out[c] = pd.to_numeric(df.get(c), errors="coerce")
    return out[MODEL_INPUTS]


def _new_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2500, C=0.5, solver="lbfgs", random_state=42
        )),
    ])


def _race_key_frame(df):
    if "race_key" in df.columns:
        return df["race_key"].astype(str)
    return (
        df["date"].astype(str) + "|" +
        df["course"].astype(str) + "|" +
        df["race_no"].astype(str)
    )


def _normalize_by_race(values, df):
    s = pd.Series(np.asarray(values, dtype=float), index=df.index)
    keys = _race_key_frame(df)
    sums = s.groupby(keys).transform("sum")
    sizes = s.groupby(keys).transform("count")
    return s.div(sums.where(sums > 0)).fillna(1.0 / sizes.clip(lower=1))


def _ece(y, p, bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = max(len(y), 1)
    err = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        n = int(m.sum())
        if n:
            err += (n / total) * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(err)


def _metric_block(df, win_prob, top3_prob):
    y1 = pd.to_numeric(df["actual_win"], errors="coerce").astype(int).to_numpy()
    y3 = pd.to_numeric(df["actual_top3"], errors="coerce").astype(int).to_numpy()
    p1 = np.clip(np.asarray(win_prob, dtype=float), 1e-6, 1-1e-6)
    p3 = np.clip(np.asarray(top3_prob, dtype=float), 1e-6, 1-1e-6)

    q = df[["date", "course", "race_no", "actual_win", "actual_top3"]].copy()
    q["p1"] = p1
    q["p3"] = p3
    q["race_key"] = _race_key_frame(df).values
    tops = q.sort_values(["race_key", "p1"], ascending=[True, False]).groupby("race_key", as_index=False).head(1)

    return {
        "rows": int(len(df)),
        "races": int(q["race_key"].nunique()),
        "win_logloss": float(log_loss(y1, p1, labels=[0, 1])),
        "win_brier": float(brier_score_loss(y1, p1)),
        "top3_logloss": float(log_loss(y3, p3, labels=[0, 1])),
        "top3_brier": float(brier_score_loss(y3, p3)),
        "win_ece": _ece(y1, p1),
        "top3_ece": _ece(y3, p3),
        "top_pick_win_rate": float(pd.to_numeric(tops["actual_win"], errors="coerce").mean()),
        "top_pick_top3_rate": float(pd.to_numeric(tops["actual_top3"], errors="coerce").mean()),
    }


def _predict_adapter(adapter, df):
    x = _adapter_frame(df)
    p1 = adapter["win_model"].predict_proba(x)[:, 1]
    p3 = adapter["top3_model"].predict_proba(x)[:, 1]
    p1 = _normalize_by_race(p1, df).to_numpy()
    p3 = np.maximum(np.clip(p3, 0.001, 0.995), p1)
    return p1, p3


def _champion_predictions(df):
    """Return the probabilities that were actually deployed for each race."""
    p1 = _normalize_by_race(
        pd.to_numeric(df["base_win_prob"], errors="coerce").fillna(0), df
    ).to_numpy()
    p3 = (
        pd.to_numeric(df["base_top3_prob"], errors="coerce")
        .fillna(0).clip(0.001, 0.995).to_numpy()
    )
    p3 = np.maximum(p3, p1)
    return p1, p3


def _promotion_decision(champion_metrics, challenger_metrics):
    c = champion_metrics
    n = challenger_metrics
    ratios = []
    for key in ["win_logloss", "win_brier", "top3_logloss", "top3_brier"]:
        denom = max(float(c[key]), 1e-9)
        ratios.append(float(n[key]) / denom)
    mean_ratio = float(np.mean(ratios))
    cal_old = float(c["win_ece"] + c["top3_ece"]) / 2.0
    cal_new = float(n["win_ece"] + n["top3_ece"]) / 2.0

    ok = (
        n["win_logloss"] <= c["win_logloss"] and
        n["win_brier"] <= c["win_brier"] and
        n["top3_logloss"] <= c["top3_logloss"] and
        n["top3_brier"] <= c["top3_brier"] and
        mean_ratio <= 0.995 and
        cal_new <= cal_old + 0.01 and
        n["top_pick_win_rate"] >= c["top_pick_win_rate"] - 0.02
    )
    return bool(ok), {
        "mean_loss_ratio": mean_ratio,
        "calibration_old": cal_old,
        "calibration_new": cal_new,
    }


def _ordered_race_keys(df):
    meta = df[["date", "course", "race_no"]].drop_duplicates().copy()
    meta["_dt"] = pd.to_datetime(meta["date"], errors="coerce")
    meta["_rn"] = pd.to_numeric(meta["race_no"].astype(str).str.replace("R", "", regex=False), errors="coerce").fillna(999)
    meta = meta.sort_values(["_dt", "course", "_rn"])
    return (meta["date"].astype(str) + "|" + meta["course"].astype(str) + "|" + meta["race_no"].astype(str)).tolist()


def _canonical_settled_rows(rows):
    """
    Keep one leakage-safe pre-race snapshot per race.

    If more snapshot types are stored in the future, use the same evaluation
    priority as the app: course_batch > pre_race > morning > manual, then the
    latest snapshot within the selected type.
    """
    q = normalize_learning_rows(rows)
    q = q[q["result_quality"].astype(str) == "complete"].copy()
    q = q[pd.to_numeric(q["actual_finish"], errors="coerce").notna()].copy()
    q = q[pd.to_numeric(q["actual_win"], errors="coerce").notna()].copy()
    q = q[pd.to_numeric(q["actual_top3"], errors="coerce").notna()].copy()
    if q.empty:
        return q

    q["race_key"] = _race_key_frame(q)
    pri = {"course_batch": 4, "pre_race": 3, "morning": 2, "manual": 1}
    q["_pri"] = q["snapshot_type"].astype(str).map(pri).fillna(0)
    q["_dt"] = pd.to_datetime(q["recorded_at"], errors="coerce")

    meta = (
        q[["race_key", "snapshot_id", "_pri", "_dt"]]
        .drop_duplicates()
        .sort_values(["race_key", "_pri", "_dt"])
        .groupby("race_key", as_index=False)
        .tail(1)
    )
    keep = set(meta["snapshot_id"].astype(str))
    q = q[q["snapshot_id"].astype(str).isin(keep)].copy()
    return q.drop(columns=["_pri", "_dt"], errors="ignore")


def load_adapter_from_store(store):
    try:
        raw, _ = store.read_bytes(ADAPTER_PATH)
        return joblib.load(BytesIO(raw)) if raw else None
    except Exception:
        return None


def load_adapter_public(
    repo="aquaria1192-spec/keiba-ai-cloud",
    branch="prediction-history",
    timeout=8,
):
    url = f"https://raw.githubusercontent.com/{repo}/{branch}/{ADAPTER_PATH}"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return joblib.load(BytesIO(r.content))
    except Exception:
        return None


def apply_online_adapter(detail, adapter):
    if detail is None or len(detail) == 0:
        return detail
    d = detail.copy()

    # Always freeze the unadapted probability first. This is the invariant
    # that prevents recursive/double correction after future promotions.
    d["raw_win_prob"] = pd.to_numeric(d.get("win_prob"), errors="coerce")
    d["raw_top3_prob"] = pd.to_numeric(d.get("top3_prob"), errors="coerce")
    d["raw_ai_index"] = pd.to_numeric(d.get("ai_index"), errors="coerce")
    if adapter is None:
        d["online_adapter_version"] = ""
        return d

    p1, p3 = _predict_adapter(adapter, d)
    d["win_prob"] = p1
    d["top3_prob"] = p3

    parts = []
    if "競馬場" in d.columns and "レース" in d.columns:
        group_cols = ["競馬場", "レース"]
    else:
        group_cols = ["course", "race_no"]
    for _, g in d.groupby(group_cols, sort=False, dropna=False):
        g = g.copy()
        max_win = max(float(pd.to_numeric(g["win_prob"], errors="coerce").max()), 1e-9)
        recent = pd.to_numeric(g.get("recent_top3_rate"), errors="coerce").fillna(0).clip(0, 1)
        g["ai_index"] = np.clip(
            100 * (
                0.65 * (pd.to_numeric(g["win_prob"], errors="coerce") / max_win) +
                0.25 * pd.to_numeric(g["top3_prob"], errors="coerce") +
                0.10 * recent
            ),
            0, 100,
        )
        g["prediction_score"] = g["win_prob"]
        odds = pd.to_numeric(g.get("odds"), errors="coerce")
        implied = pd.to_numeric(g.get("implied_prob"), errors="coerce").fillna(0)
        g["expected_value"] = g["win_prob"] * odds
        g["value_gap"] = g["win_prob"] - implied
        g = g.sort_values(["prediction_score", "top3_prob", "ai_index"], ascending=False).reset_index(drop=True)
        g["順位"] = range(1, len(g) + 1)
        g["印"] = [mark_for_rank(i, len(g)) for i in g["順位"]]
        g["評価"] = g["印"].map(mark_label)
        g["online_adapter_version"] = str(adapter.get("version", ""))
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else d


def run_online_learning(store, min_new_races=MIN_NEW_RACES):
    rows = load_learning_rows(store)
    current_adapter = load_adapter_from_store(store)
    current_version = champion_version(current_adapter)
    q = _canonical_settled_rows(rows)

    status = store.read_json(LEARNING_STATUS_PATH) or {}
    race_keys = _ordered_race_keys(q) if len(q) else []
    total_races = len(race_keys)
    last_evaluated = int(status.get("last_evaluated_race_count", 0) or 0)
    new_races = total_races - last_evaluated

    result = {
        "checked_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "champion_version": current_version,
        "total_settled_races": total_races,
        "new_races_since_last_evaluation": new_races,
        "min_new_races": int(min_new_races),
        "status": "waiting",
    }

    if total_races < max(80, int(min_new_races)) or new_races < int(min_new_races):
        status.update({
            "last_seen_race_count": total_races,
            "last_status": "waiting",
            "last_checked_at": result["checked_at"],
            "latest": result,
        })
        store.write_json(LEARNING_STATUS_PATH, status, "Update online learning wait status")
        return result

    new_block = race_keys[last_evaluated:]
    valid_n = max(40, int(round(len(new_block) * VALID_RATIO)))
    valid_n = min(valid_n, max(1, len(new_block) - 40))
    valid_keys = set(new_block[-valid_n:])
    train_keys = set(race_keys) - valid_keys
    train = q[q["race_key"].isin(train_keys)].copy()
    valid = q[q["race_key"].isin(valid_keys)].copy()

    if train["race_key"].nunique() < 40 or valid["race_key"].nunique() < 20:
        result["status"] = "insufficient_split"
        status.update({
            "last_seen_race_count": total_races,
            "last_status": result["status"],
            "last_checked_at": result["checked_at"],
            "latest": result,
        })
        store.write_json(
            LEARNING_STATUS_PATH,status,
            "Update online learning insufficient split status"
        )
        return result

    x_train = _adapter_frame(train)
    win_model = _new_model()
    top3_model = _new_model()
    win_model.fit(x_train, pd.to_numeric(train["actual_win"], errors="coerce").astype(int))
    top3_model.fit(x_train, pd.to_numeric(train["actual_top3"], errors="coerce").astype(int))

    candidate = {
        "version": "candidate",
        "created_at": result["checked_at"],
        "parent_version": current_version,
        "win_model": win_model,
        "top3_model": top3_model,
        "model_inputs": MODEL_INPUTS,
    }

    champ_p1, champ_p3 = _champion_predictions(valid)
    cand_p1, cand_p3 = _predict_adapter(candidate, valid)
    champ_metrics = _metric_block(valid, champ_p1, champ_p3)
    cand_metrics = _metric_block(valid, cand_p1, cand_p3)
    promote, decision = _promotion_decision(champ_metrics, cand_metrics)

    result.update({
        "status": "promoted" if promote else "rejected",
        "train_races": int(train["race_key"].nunique()),
        "validation_races": int(valid["race_key"].nunique()),
        "champion_metrics": champ_metrics,
        "challenger_metrics": cand_metrics,
        "decision": decision,
    })

    status.update({
        "last_evaluated_race_count": total_races,
        "last_seen_race_count": total_races,
        "last_status": result["status"],
        "last_checked_at": result["checked_at"],
    })

    if promote:
        full_x = _adapter_frame(q)
        final_win = _new_model()
        final_top3 = _new_model()
        final_win.fit(full_x, pd.to_numeric(q["actual_win"], errors="coerce").astype(int))
        final_top3.fit(full_x, pd.to_numeric(q["actual_top3"], errors="coerce").astype(int))
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        promoted = {
            "version": f"ol-{stamp}",
            "created_at": result["checked_at"],
            "parent_version": current_version,
            "trained_races": total_races,
            "win_model": final_win,
            "top3_model": final_top3,
            "model_inputs": MODEL_INPUTS,
            "validation": result,
        }
        bio = BytesIO()
        joblib.dump(promoted, bio, compress=3)
        store.write_bytes(ADAPTER_PATH, bio.getvalue(), f"Promote online adapter {promoted['version']}")
        result["promoted_version"] = promoted["version"]

    status.update({"latest": result})
    store.write_json(LEARNING_STATUS_PATH, status, f"Update online learning status {result['status']}")
    store.write_json(ADAPTER_METRICS_PATH, result, f"Save online learning metrics {result['status']}")
    return result

from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from feature_engineering import normalize_columns, validate_entries, _class_score, _style_score

BASE = Path(__file__).resolve().parent
STORE_FILE = BASE/"data"/"cloud_feature_store.joblib"

def load_feature_store():
    return joblib.load(STORE_FILE)

def enrich_entries_cloud(entries, store=None):
    store = store or load_feature_store()
    ent = normalize_columns(entries).copy()
    missing = validate_entries(ent)
    if missing:
        raise ValueError("出走表に必要な列がありません: " + ", ".join(missing))

    ent["_dt"] = pd.to_datetime(ent["date"], errors="coerce")
    for c in ["distance","body_weight","body_weight_diff","popularity","odds","gate"]:
        if c not in ent.columns:
            ent[c] = np.nan
        ent[c] = pd.to_numeric(ent[c], errors="coerce")
    for c in ["race_class","running_style"]:
        if c not in ent.columns:
            ent[c] = np.nan

    ent["race_class_score"] = ent["race_class"].map(_class_score)
    ent["distance_bucket"] = (ent["distance"] // 400 * 400).astype("Int64")

    latest = store["latest"].copy()
    recent = store["recent"].copy()
    hd = store["horse_distance"].copy()
    hc = store["horse_course"].copy()
    hs = store["horse_surface"].copy()
    jr = store["jockey"].copy()
    tr = store["trainer"].copy()
    gr = store["gate"].copy()
    glob = store["global"]

    # Merge compact lookups instead of loading 464k historical rows.
    out = ent.merge(latest, on="horse_name", how="left", suffixes=("","_prev"))
    out = out.merge(recent, on="horse_name", how="left")

    hd = hd.rename(columns={"rate":"distance_top3_rate"})
    hc = hc.rename(columns={"rate":"course_top3_rate"})
    hs = hs.rename(columns={"rate":"surface_top3_rate"})
    jr = jr.rename(columns={"rate":"jockey_top3_rate"})
    tr = tr.rename(columns={"rate":"trainer_top3_rate"})
    gr = gr.rename(columns={"rate":"gate_condition_top3_rate"})

    out = out.merge(hd, on=["horse_name","distance"], how="left")
    out = out.merge(hc, on=["horse_name","course"], how="left")
    out = out.merge(hs, on=["horse_name","surface"], how="left")
    out = out.merge(jr, on="jockey", how="left")
    out = out.merge(tr, on="trainer", how="left")
    out = out.merge(gr, on=["course","surface","distance_bucket","gate"], how="left")

    prev_dt = pd.to_datetime(out["_dt_prev"], errors="coerce")
    cur_dt = pd.to_datetime(out["_dt"], errors="coerce")

    out["prev_finish"] = pd.to_numeric(out["finish"], errors="coerce")
    out["prev_margin"] = pd.to_numeric(out["margin"], errors="coerce")
    out["prev_last3f_rank"] = pd.to_numeric(out["last3f_rank"], errors="coerce")
    prev_distance = pd.to_numeric(out["distance_prev"], errors="coerce")
    prev_bw = pd.to_numeric(out["body_weight_prev"], errors="coerce")
    prev_class = pd.to_numeric(out["race_class_score_prev"], errors="coerce")

    out["distance_change"] = (
        pd.to_numeric(out["distance"], errors="coerce") - prev_distance
    ).fillna(0.0)
    out["days_since_last"] = (cur_dt - prev_dt).dt.days

    supplied_diff = pd.to_numeric(out.get("body_weight_diff"), errors="coerce")
    calc_bw = pd.to_numeric(out["body_weight"], errors="coerce") - prev_bw
    out["body_weight_change_from_prev"] = calc_bw.where(calc_bw.notna(), supplied_diff).fillna(0.0)

    current_class = pd.to_numeric(out["race_class_score"], errors="coerce")
    out["class_change"] = (current_class - prev_class).fillna(0.0)
    out["running_style_score"] = out["running_style"].map(_style_score)

    odds = pd.to_numeric(out["odds"], errors="coerce")
    out["implied_prob"] = 1 / odds.replace(0, np.nan)
    out["log_odds"] = np.log(odds.clip(lower=1.01))

    fill = {
        "recent_avg_finish": glob["finish"],
        "recent_top3_rate": glob["top3"],
        "recent_avg_margin": glob["margin"],
        "recent_avg_last3f_rank": glob["last3f_rank"],
        "recent_avg_popularity": glob["popularity"],
        "distance_top3_rate": glob["top3"],
        "course_top3_rate": glob["top3"],
        "surface_top3_rate": glob["top3"],
        "jockey_top3_rate": glob["top3"],
        "trainer_top3_rate": glob["top3"],
        "gate_condition_top3_rate": glob["top3"],
    }
    for c, v in fill.items():
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(v)

    # Drop lookup-only columns while keeping the entry's original columns/features.
    out = out.drop(
        columns=[
            "_dt","_dt_prev","finish","margin","last3f_rank",
            "distance_prev","body_weight_prev","race_class_score_prev",
        ],
        errors="ignore",
    )
    return out

from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from feature_engineering import normalize_columns, validate_entries, _class_score, _style_score, _add_race_relative_features

BASE = Path(__file__).resolve().parent
STORE_FILE = BASE/"data"/"cloud_feature_store.joblib"

def load_feature_store(): return joblib.load(STORE_FILE)

def _merge_rate(out, table, keys, rate_name, starts_name=None):
    if table is None or len(table)==0:
        out[rate_name]=np.nan
        if starts_name: out[starts_name]=0
        return out
    q=table.copy()
    ren={"rate":rate_name}
    if starts_name and "starts" in q.columns: ren["starts"]=starts_name
    q=q.rename(columns=ren)
    return out.merge(q,on=keys,how="left")

def enrich_entries_cloud(entries, store=None):
    store=store or load_feature_store()
    ent=normalize_columns(entries).copy(); missing=validate_entries(ent)
    if missing: raise ValueError("出走表に必要な列がありません: "+", ".join(missing))
    ent["_dt"]=pd.to_datetime(ent["date"],errors="coerce")
    for c in ["distance","body_weight","body_weight_diff","popularity","odds","gate"]:
        if c not in ent.columns: ent[c]=np.nan
        ent[c]=pd.to_numeric(ent[c],errors="coerce")
    for c in ["race_class","running_style"]:
        if c not in ent.columns: ent[c]=""
    # Merge keys must keep string dtype even when a live column is entirely blank/NaN.
    for c in ["course","surface","going","horse_name","jockey","trainer","sex","race_class","running_style"]:
        if c not in ent.columns: ent[c]=""
        ent[c]=ent[c].fillna("").astype(str).str.strip()
    ent["race_class_score"]=ent["race_class"].map(_class_score)
    ent["distance_bucket"]=(ent["distance"]//400*400).astype("Int64")

    latest=store["latest"].copy(); recent=store["recent"].copy(); glob=store["global"]
    out=ent.merge(latest,on="horse_name",how="left",suffixes=("","_prev"))
    out=out.merge(recent,on="horse_name",how="left")

    out=_merge_rate(out,store.get("horse_distance"),["horse_name","distance"],"distance_top3_rate","distance_starts")
    out=_merge_rate(out,store.get("horse_course"),["horse_name","course"],"course_top3_rate","course_starts")
    out=_merge_rate(out,store.get("horse_surface"),["horse_name","surface"],"surface_top3_rate","surface_starts")
    out=_merge_rate(out,store.get("jockey"),["jockey"],"jockey_top3_rate","jockey_starts")
    out=_merge_rate(out,store.get("trainer"),["trainer"],"trainer_top3_rate","trainer_starts")
    out=_merge_rate(out,store.get("gate"),["course","surface","distance_bucket","gate"],"gate_condition_top3_rate","gate_condition_starts")

    out=_merge_rate(out,store.get("horse_going"),["horse_name","surface","going"],"horse_going_top3_rate","going_starts")
    # Legacy UI names the same horse-going feature going_top3_rate.
    out["going_top3_rate"]=pd.to_numeric(out["horse_going_top3_rate"],errors="coerce")
    gg=store.get("global_going")
    if gg is not None and len(gg):
        q=gg.rename(columns={"rate":"going_global_top3_rate","starts":"going_global_starts"})
        out=out.merge(q,on=["surface","going"],how="left")
    else:
        out["going_global_top3_rate"]=glob["top3"];out["going_global_starts"]=0

    for key,keys,rate,starts in [
        ("jockey_course",["jockey","course"],"jockey_course_top3_rate","jockey_course_starts"),
        ("jockey_surface",["jockey","surface"],"jockey_surface_top3_rate","jockey_surface_starts"),
        ("jockey_distance",["jockey","distance_bucket"],"jockey_distance_top3_rate","jockey_distance_starts"),
        ("jockey_going",["jockey","surface","going"],"jockey_going_top3_rate","jockey_going_starts"),
        ("jockey_trainer",["jockey","trainer"],"jockey_trainer_top3_rate","jockey_trainer_starts"),
        ("trainer_course",["trainer","course"],"trainer_course_top3_rate","trainer_course_starts"),
        ("trainer_surface",["trainer","surface"],"trainer_surface_top3_rate","trainer_surface_starts"),
        ("trainer_distance",["trainer","distance_bucket"],"trainer_distance_top3_rate","trainer_distance_starts"),
    ]:
        out=_merge_rate(out,store.get(key),keys,rate,starts)

    hstyle=store.get("horse_style")
    if hstyle is not None and len(hstyle): out=out.merge(hstyle,on="horse_name",how="left")
    else: out["usual_running_style"]=""
    live=out["running_style"].fillna("").astype(str).str.strip(); usual=out["usual_running_style"].fillna("").astype(str).str.strip()
    out["running_style"]=live.where(live!="",usual)

    prev_dt=pd.to_datetime(out["_dt_prev"],errors="coerce");cur_dt=pd.to_datetime(out["_dt"],errors="coerce")
    out["prev_finish"]=pd.to_numeric(out["finish"],errors="coerce")
    out["prev_margin"]=pd.to_numeric(out["margin"],errors="coerce")
    out["prev_last3f_rank"]=pd.to_numeric(out["last3f_rank"],errors="coerce")
    prev_distance=pd.to_numeric(out["distance_prev"],errors="coerce")
    prev_bw=pd.to_numeric(out["body_weight_prev"],errors="coerce")
    prev_class=pd.to_numeric(out["race_class_score_prev"],errors="coerce")
    out["distance_change"]=(pd.to_numeric(out["distance"],errors="coerce")-prev_distance).fillna(0.0)
    out["days_since_last"]=(cur_dt-prev_dt).dt.days
    supplied=pd.to_numeric(out.get("body_weight_diff"),errors="coerce")
    calc=pd.to_numeric(out["body_weight"],errors="coerce")-prev_bw
    out["body_weight_change_from_prev"]=calc.where(calc.notna(),supplied).fillna(0.0)
    out["class_change"]=(pd.to_numeric(out["race_class_score"],errors="coerce")-prev_class).fillna(0.0)
    out["running_style_score"]=out["running_style"].map(_style_score)
    odds=pd.to_numeric(out["odds"],errors="coerce")
    out["implied_prob"]=1/odds.replace(0,np.nan);out["log_odds"]=np.log(odds.clip(lower=1.01))

    # Evidence-aware fallback. Condition interactions fall back to the person's
    # overall rate rather than an arbitrary global constant.
    out["going_global_top3_rate"]=pd.to_numeric(out["going_global_top3_rate"],errors="coerce").fillna(glob["top3"])
    out["horse_going_top3_rate"]=pd.to_numeric(out["horse_going_top3_rate"],errors="coerce").fillna(out["going_global_top3_rate"])
    out["going_top3_rate"]=out["horse_going_top3_rate"]
    for c,v in {
        "recent_avg_finish":glob["finish"],"recent_top3_rate":glob["top3"],"recent_avg_margin":glob["margin"],
        "recent_avg_last3f_rank":glob["last3f_rank"],"recent_avg_popularity":glob["popularity"],
        "distance_top3_rate":glob["top3"],"course_top3_rate":glob["top3"],"surface_top3_rate":glob["top3"],
        "jockey_top3_rate":glob["top3"],"trainer_top3_rate":glob["top3"],"gate_condition_top3_rate":glob["top3"],
    }.items(): out[c]=pd.to_numeric(out[c],errors="coerce").fillna(v)
    for c in ["jockey_course_top3_rate","jockey_surface_top3_rate","jockey_distance_top3_rate","jockey_going_top3_rate","jockey_trainer_top3_rate"]:
        out[c]=pd.to_numeric(out[c],errors="coerce").fillna(out["jockey_top3_rate"])
    for c in ["trainer_course_top3_rate","trainer_surface_top3_rate","trainer_distance_top3_rate"]:
        out[c]=pd.to_numeric(out[c],errors="coerce").fillna(out["trainer_top3_rate"])
    for c in [x for x in out.columns if x.endswith("_starts")]: out[c]=pd.to_numeric(out[c],errors="coerce").fillna(0)

    out=_add_race_relative_features(out)
    return out.drop(columns=["_dt","_dt_prev","finish","margin","last3f_rank","distance_prev","body_weight_prev","race_class_score_prev"],errors="ignore")

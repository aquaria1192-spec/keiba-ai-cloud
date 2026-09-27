import pandas as pd
import numpy as np

BASE_REQUIRED=["date","race_id","course","surface","distance","going","horse_no","gate","horse_name","jockey","trainer","age","sex","weight","body_weight","body_weight_diff","odds","popularity"]
ADVANCED_OPTIONAL=["race_class","finish_time","margin","last3f","last3f_rank","running_style"]
JP_ALIASES={
    "日付":"date","開催日":"date","レースID":"race_id","競馬場":"course","開催地":"course",
    "芝ダ":"surface","馬場種別":"surface","距離":"distance","馬場":"going","馬場状態":"going",
    "馬番":"horse_no","枠":"gate","枠番":"gate","馬名":"horse_name","騎手":"jockey","調教師":"trainer",
    "年齢":"age","性":"sex","斤量":"weight","馬体重":"body_weight","馬体重増減":"body_weight_diff",
    "単勝":"odds","単勝オッズ":"odds","人気":"popularity","着順":"finish","レース":"race_no","R":"race_no",
    "レース名":"race_name","クラス":"race_class","競走クラス":"race_class","タイム":"finish_time","着差":"margin",
    "上がり":"last3f","上り":"last3f","上がり3F":"last3f","上がり順位":"last3f_rank","上り順位":"last3f_rank","脚質":"running_style"
}
CLASS_MAP={"新馬":1,"未勝利":2,"1勝":3,"1勝クラス":3,"500万":3,"2勝":4,"2勝クラス":4,"1000万":4,"3勝":5,"3勝クラス":5,"1600万":5,"OP":6,"オープン":6,"L":7,"リステッド":7,"G3":8,"GIII":8,"G2":9,"GII":9,"G1":10,"GI":10}
STYLE_MAP={"逃げ":1,"先行":2,"差し":3,"追込":4,"追い込み":4,"front":1,"pace":2,"stalker":2,"closer":4}

def normalize_columns(df):
    out=df.copy(); out.columns=[str(c).strip() for c in out.columns]
    return out.rename(columns={c:JP_ALIASES[c] for c in out.columns if c in JP_ALIASES})
def _num(s): return pd.to_numeric(s,errors="coerce")
def _class_score(v):
    if pd.isna(v): return np.nan
    s=str(v).strip()
    if s in CLASS_MAP:return CLASS_MAP[s]
    for k,val in CLASS_MAP.items():
        if k.lower() in s.lower(): return val
    try:return float(s)
    except:return np.nan
def _style_score(v):
    if pd.isna(v): return np.nan
    s=str(v).strip()
    if s in STYLE_MAP:return STYLE_MAP[s]
    for k,val in STYLE_MAP.items():
        if k.lower() in s.lower(): return val
    return np.nan

def validate_raw_history(df):
    df=normalize_columns(df); return [c for c in BASE_REQUIRED+["finish"] if c not in df.columns]
def validate_entries(df):
    df=normalize_columns(df); return [c for c in BASE_REQUIRED if c!="race_id" and c not in df.columns]

def _prior_mean(df,keys,col="_top3"):
    gr=df.groupby(keys,sort=False,dropna=False)
    cnt=gr.cumcount().astype(float)
    csum=gr[col].cumsum()-df[col]
    return csum.div(cnt.where(cnt>0))
def _rolling_prior(df,key,col,window=5):
    shifted=df.groupby(key,sort=False)[col].shift(1)
    out=(shifted.groupby(df[key],sort=False).rolling(window,min_periods=1).mean().reset_index(level=0,drop=True))
    return out.sort_index()

def _add_race_relative_features(df):
    out=df.copy()
    if "race_id" in out.columns:
        key=out["race_id"].astype(str)
    else:
        parts=[]
        for c in ["date","course","race_no"]:
            if c in out.columns: parts.append(out[c].astype(str))
        if parts:
            key=parts[0]
            for p in parts[1:]: key=key+"|"+p
        else:
            key=pd.Series(np.arange(len(out)),index=out.index).astype(str)
    out["field_size"]=out.groupby(key)["horse_no"].transform("count").astype(float)
    implied=pd.to_numeric(out.get("implied_prob"),errors="coerce")
    isum=implied.groupby(key).transform("sum")
    out["market_prob_norm"]=implied.div(isum.where(isum>0))
    odds=pd.to_numeric(out.get("odds"),errors="coerce")
    out["market_rank_pct"]=odds.groupby(key).rank(method="average",ascending=True).div(out["field_size"])
    pop=pd.to_numeric(out.get("popularity"),errors="coerce")
    out["popularity_pct"]=pop.div(out["field_size"])
    for c in [
        "recent_top3_rate","jockey_top3_rate","trainer_top3_rate",
        "distance_top3_rate","course_top3_rate","surface_top3_rate",
    ]:
        s=pd.to_numeric(out.get(c),errors="coerce")
        out[c+"_rank_pct"]=s.groupby(key).rank(method="average",ascending=False).div(out["field_size"])
    return out

def build_history_features(raw_history):
    """Leakage-aware features. Every historical rate uses only prior starts."""
    df=normalize_columns(raw_history).copy(); missing=validate_raw_history(df)
    if missing: raise ValueError("過去結果CSVに必要な列がありません: "+", ".join(missing))
    df["date"]=pd.to_datetime(df["date"],errors="coerce")
    if df["date"].isna().any(): raise ValueError("date に日付として解釈できない値があります。")
    nums=["distance","horse_no","gate","age","weight","body_weight","body_weight_diff","odds","popularity","finish","margin","last3f","last3f_rank"]
    for c in nums:
        if c not in df.columns: df[c]=np.nan
        df[c]=_num(df[c])
    for c in ["race_class","running_style"]:
        if c not in df.columns: df[c]=np.nan
    df["race_class_score"]=df["race_class"].map(_class_score)
    df["running_style_score"]=df["running_style"].map(_style_score)
    sort_cols=["date"]+(["race_day_seq"] if "race_day_seq" in df.columns else [])+["race_id","horse_no"]
    df=df.sort_values(sort_cols).reset_index(drop=True)
    df["_top3"]=(df["finish"]<=3).astype(float)

    gh=df.groupby("horse_name",sort=False)
    for src,dst in [("finish","prev_finish"),("margin","prev_margin"),("last3f_rank","prev_last3f_rank"),("distance","prev_distance"),("body_weight","prev_body_weight"),("race_class_score","prev_class_score"),("surface","prev_surface"),("date","prev_date")]:
        df[dst]=gh[src].shift(1)
    df["distance_change"]=df["distance"]-df["prev_distance"]
    df["days_since_last"]=(df["date"]-df["prev_date"]).dt.days
    df["body_weight_change_from_prev"]=df["body_weight"]-df["prev_body_weight"]
    df["class_change"]=df["race_class_score"]-df["prev_class_score"]

    for src,dst in [("finish","recent_avg_finish"),("_top3","recent_top3_rate"),("margin","recent_avg_margin"),("last3f_rank","recent_avg_last3f_rank"),("popularity","recent_avg_popularity")]:
        df[dst]=_rolling_prior(df,"horse_name",src,5)

    df["distance_top3_rate"]=_prior_mean(df,["horse_name","distance"])
    df["course_top3_rate"]=_prior_mean(df,["horse_name","course"])
    df["surface_top3_rate"]=_prior_mean(df,["horse_name","surface"])
    df["jockey_top3_rate"]=_prior_mean(df,["jockey"])
    df["trainer_top3_rate"]=_prior_mean(df,["trainer"])
    df["distance_bucket"]=(df["distance"]//400*400).astype("Int64")
    df["gate_condition_top3_rate"]=_prior_mean(df,["course","surface","distance_bucket","gate"])

    # Ver.1.11: learned condition interactions rather than heuristic-only use.
    df["horse_going_top3_rate"]=_prior_mean(df,["horse_name","surface","going"])
    df["jockey_course_top3_rate"]=_prior_mean(df,["jockey","course"])
    df["jockey_surface_top3_rate"]=_prior_mean(df,["jockey","surface"])
    df["jockey_distance_top3_rate"]=_prior_mean(df,["jockey","distance_bucket"])
    df["jockey_going_top3_rate"]=_prior_mean(df,["jockey","surface","going"])
    df["jockey_trainer_top3_rate"]=_prior_mean(df,["jockey","trainer"])
    df["trainer_course_top3_rate"]=_prior_mean(df,["trainer","course"])
    df["trainer_surface_top3_rate"]=_prior_mean(df,["trainer","surface"])
    df["trainer_distance_top3_rate"]=_prior_mean(df,["trainer","distance_bucket"])

    df["implied_prob"]=1/df["odds"].replace(0,np.nan)
    df["log_odds"]=np.log(df["odds"].clip(lower=1.01))
    df=_add_race_relative_features(df)

    df["date"]=df["date"].dt.date.astype(str)
    df["prev_date"]=pd.to_datetime(df["prev_date"],errors="coerce").dt.date.astype(str)
    return df.drop(columns=["_top3"])

def enrich_entries(raw_history,entries):
    """Compatibility path. Cloud uses cloud_features.enrich_entries_cloud."""
    hist=build_history_features(raw_history)
    ent=normalize_columns(entries).copy()
    missing=validate_entries(ent)
    if missing: raise ValueError("出走表に必要な列がありません: "+", ".join(missing))
    # Minimal compatibility: caller should prefer cloud feature store for production.
    for c in [
        "recent_avg_finish","recent_top3_rate","recent_avg_margin","recent_avg_last3f_rank","recent_avg_popularity",
        "distance_top3_rate","course_top3_rate","surface_top3_rate","jockey_top3_rate","trainer_top3_rate",
        "gate_condition_top3_rate","horse_going_top3_rate","jockey_course_top3_rate","jockey_surface_top3_rate",
        "jockey_distance_top3_rate","jockey_going_top3_rate","jockey_trainer_top3_rate","trainer_course_top3_rate",
        "trainer_surface_top3_rate","trainer_distance_top3_rate","prev_finish","prev_margin","prev_last3f_rank",
        "distance_change","days_since_last","body_weight_change_from_prev","class_change","race_class_score",
        "running_style_score"
    ]:
        if c not in ent: ent[c]=np.nan
    ent["implied_prob"]=1/pd.to_numeric(ent.get("odds"),errors="coerce").replace(0,np.nan)
    ent["log_odds"]=np.log(pd.to_numeric(ent.get("odds"),errors="coerce").clip(lower=1.01))
    return _add_race_relative_features(ent)

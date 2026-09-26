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

def build_history_features(raw_history):
    """Large-data optimized leakage-aware feature generation."""
    df=normalize_columns(raw_history).copy(); missing=validate_raw_history(df)
    if missing: raise ValueError("過去結果CSVに必要な列がありません: "+", ".join(missing))
    df["date"]=pd.to_datetime(df["date"],errors="coerce")
    if df["date"].isna().any(): raise ValueError("date に日付として解釈できない値があります。")
    nums=["distance","horse_no","gate","age","weight","body_weight","body_weight_diff","odds","popularity","finish","margin","last3f","last3f_rank"]
    for c in nums:
        if c not in df.columns: df[c]=np.nan
        df[c]=_num(df[c])
    if "race_class" not in df.columns: df["race_class"]=np.nan
    if "running_style" not in df.columns: df["running_style"]=np.nan
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
    df["implied_prob"]=1/df["odds"].replace(0,np.nan)
    df["log_odds"]=np.log(df["odds"].clip(lower=1.01))
    df["date"]=df["date"].dt.date.astype(str)
    df["prev_date"]=pd.to_datetime(df["prev_date"],errors="coerce").dt.date.astype(str)
    return df.drop(columns=["_top3"])

def enrich_entries(raw_history,entries):
    """Optimized for a large history and a small current entry table."""
    hist=normalize_columns(raw_history).copy(); ent=normalize_columns(entries).copy()
    mh=validate_raw_history(hist); me=validate_entries(ent)
    if mh: raise ValueError("過去結果CSVに必要な列がありません: "+", ".join(mh))
    if me: raise ValueError("出走表CSVに必要な列がありません: "+", ".join(me))
    hist["_dt"]=pd.to_datetime(hist["date"],errors="coerce"); ent["_dt"]=pd.to_datetime(ent["date"],errors="coerce")
    for c in ["finish","distance","body_weight","popularity","odds","margin","last3f_rank","gate"]:
        if c not in hist.columns: hist[c]=np.nan
        hist[c]=_num(hist[c])
    for c in ["distance","body_weight","body_weight_diff","popularity","odds","gate"]:
        if c not in ent.columns: ent[c]=np.nan
        ent[c]=_num(ent[c])
    for c in ["race_class","running_style"]:
        if c not in hist.columns: hist[c]=np.nan
        if c not in ent.columns: ent[c]=np.nan
    hist["race_class_score"]=hist["race_class"].map(_class_score); ent["race_class_score"]=ent["race_class"].map(_class_score)
    hist["_top3"]=(hist["finish"]<=3).astype(float)
    hist["distance_bucket"]=(hist["distance"]//400*400).astype("Int64"); ent["distance_bucket"]=(ent["distance"]//400*400).astype("Int64")
    result=[]
    for dt,eg in ent.groupby("_dt",dropna=False,sort=False):
        h=hist[hist["_dt"]<dt].copy() if pd.notna(dt) else hist.copy()
        global_top3=float(h["_top3"].mean()) if len(h) else .20
        global_finish=float(h["finish"].mean()) if len(h) else 8.0
        global_margin=float(h["margin"].dropna().mean()) if h["margin"].notna().any() else 0.0
        global_l3=float(h["last3f_rank"].dropna().mean()) if h["last3f_rank"].notna().any() else 8.0
        global_pop=float(h["popularity"].dropna().mean()) if h["popularity"].notna().any() else 8.0
        horses=set(eg["horse_name"].astype(str)); jockeys=set(eg["jockey"].astype(str)); trainers=set(eg["trainer"].astype(str))
        hp=h[h["horse_name"].astype(str).isin(horses)].sort_values(["_dt","race_id","horse_no"])
        prev_rows=hp.groupby(hp["horse_name"].astype(str),sort=False).tail(1) if len(hp) else hp
        prev={str(r["horse_name"]):r for _,r in prev_rows.iterrows()}
        last5=hp.groupby(hp["horse_name"].astype(str),sort=False,group_keys=False).tail(5) if len(hp) else hp
        recent_finish=last5.groupby(last5["horse_name"].astype(str))["finish"].mean().to_dict() if len(last5) else {}
        recent_margin=last5.groupby(last5["horse_name"].astype(str))["margin"].mean().to_dict() if len(last5) else {}
        recent_l3=last5.groupby(last5["horse_name"].astype(str))["last3f_rank"].mean().to_dict() if len(last5) else {}
        recent_pop=last5.groupby(last5["horse_name"].astype(str))["popularity"].mean().to_dict() if len(last5) else {}
        recent_top3=last5.groupby(last5["horse_name"].astype(str))["_top3"].mean().to_dict() if len(last5) else {}
        hd=hp.groupby([hp["horse_name"].astype(str),"distance"])["_top3"].mean().to_dict() if len(hp) else {}
        hc=hp.groupby([hp["horse_name"].astype(str),hp["course"].astype(str)])["_top3"].mean().to_dict() if len(hp) else {}
        hs=hp.groupby([hp["horse_name"].astype(str),hp["surface"].astype(str)])["_top3"].mean().to_dict() if len(hp) else {}
        hj_src=h[h["jockey"].astype(str).isin(jockeys)]
        ht_src=h[h["trainer"].astype(str).isin(trainers)]
        hj=hj_src.groupby(hj_src["jockey"].astype(str))["_top3"].mean().to_dict() if len(hj_src) else {}
        ht=ht_src.groupby(ht_src["trainer"].astype(str))["_top3"].mean().to_dict() if len(ht_src) else {}
        hg=h.groupby(["course","surface","distance_bucket","gate"],dropna=False)["_top3"].mean().to_dict() if len(h) else {}
        for _,r in eg.iterrows():
            horse=str(r["horse_name"]); jockey=str(r["jockey"]); trainer=str(r["trainer"]); course=str(r["course"]); surface=str(r["surface"])
            distance=r["distance"]; gate=r["gate"]; bucket=r["distance_bucket"]; pr=prev.get(horse)
            prev_distance=float(pr["distance"]) if pr is not None and pd.notna(pr["distance"]) else np.nan
            prev_bw=float(pr["body_weight"]) if pr is not None and pd.notna(pr["body_weight"]) else np.nan
            prev_class=float(pr["race_class_score"]) if pr is not None and pd.notna(pr["race_class_score"]) else np.nan
            prev_dt=pr["_dt"] if pr is not None else pd.NaT
            supplied_diff=pd.to_numeric(pd.Series([r.get("body_weight_diff")]),errors="coerce").iloc[0]
            d=r.to_dict(); d.update({
                "prev_finish":float(pr["finish"]) if pr is not None and pd.notna(pr["finish"]) else np.nan,
                "prev_margin":float(pr["margin"]) if pr is not None and pd.notna(pr["margin"]) else np.nan,
                "prev_last3f_rank":float(pr["last3f_rank"]) if pr is not None and pd.notna(pr["last3f_rank"]) else np.nan,
                "distance_change":float(distance-prev_distance) if pd.notna(prev_distance) else 0.0,
                "days_since_last":float((dt-prev_dt).days) if pd.notna(dt) and pd.notna(prev_dt) else np.nan,
                "body_weight_change_from_prev":float(r["body_weight"]-prev_bw) if pd.notna(prev_bw) and pd.notna(r["body_weight"]) else (float(supplied_diff) if pd.notna(supplied_diff) else 0.0),
                "class_change":float(r["race_class_score"]-prev_class) if pd.notna(r["race_class_score"]) and pd.notna(prev_class) else 0.0,
                "recent_avg_finish":recent_finish.get(horse,np.nan),"recent_top3_rate":recent_top3.get(horse,np.nan),"recent_avg_margin":recent_margin.get(horse,np.nan),"recent_avg_last3f_rank":recent_l3.get(horse,np.nan),"recent_avg_popularity":recent_pop.get(horse,np.nan),
                "distance_top3_rate":hd.get((horse,distance),np.nan),"course_top3_rate":hc.get((horse,course),np.nan),"surface_top3_rate":hs.get((horse,surface),np.nan),"jockey_top3_rate":hj.get(jockey,np.nan),"trainer_top3_rate":ht.get(trainer,np.nan),"gate_condition_top3_rate":hg.get((course,surface,bucket,gate),np.nan),
                "running_style_score":_style_score(r.get("running_style")),"implied_prob":1/float(r["odds"]) if pd.notna(r["odds"]) and float(r["odds"])>0 else np.nan,"log_odds":np.log(max(float(r["odds"]),1.01)) if pd.notna(r["odds"]) else np.nan
            }); result.append(d)
        fill={"recent_avg_finish":global_finish,"recent_top3_rate":global_top3,"recent_avg_margin":global_margin,"recent_avg_last3f_rank":global_l3,"recent_avg_popularity":global_pop,"distance_top3_rate":global_top3,"course_top3_rate":global_top3,"surface_top3_rate":global_top3,"jockey_top3_rate":global_top3,"trainer_top3_rate":global_top3,"gate_condition_top3_rate":global_top3}
    out=pd.DataFrame(result).drop(columns=["_dt"],errors="ignore")
    for c,v in fill.items(): out[c]=pd.to_numeric(out[c],errors="coerce").fillna(v)
    return out

from __future__ import annotations

from datetime import datetime
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo
import base64
import hashlib
import json
import re

import numpy as np
import pandas as pd
import requests

from github_storage_config import (
    resolve_secret, resolve_token, resolve_history_repo, resolve_history_branch
)
from auto_data_builder import _session, VENUE_CODE, _flatten_columns, _find_col
from race_day_context import (
    _result_url_from_entry, parse_jra_result_html,
    collect_official_entry_urls
)
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
LOCAL_HISTORY = DATA_DIR / "prediction_history.csv"
JST = ZoneInfo("Asia/Tokyo")
EVALUATION_STORE_API_VERSION = 5

DEFAULT_AUTO_HISTORY_REPO = "aquaria1192-spec/keiba-ai-cloud"
DEFAULT_AUTO_HISTORY_BRANCH = "prediction-history"
DEFAULT_AUTO_HISTORY_PATH = "automation_data/pre_race_predictions.csv"
DEFAULT_HISTORY_REPO = DEFAULT_AUTO_HISTORY_REPO
DEFAULT_HISTORY_BRANCH = DEFAULT_AUTO_HISTORY_BRANCH

def load_public_auto_history(
    repo=DEFAULT_AUTO_HISTORY_REPO,
    branch=DEFAULT_AUTO_HISTORY_BRANCH,
    path=DEFAULT_AUTO_HISTORY_PATH,
):
    """
    Read scheduled pre-race predictions from the public prediction-history branch.
    No token is required for a public GitHub repository.
    """
    if not repo:
        return blank_history()
    url=f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
    try:
        r=requests.get(url,timeout=20)
        if r.status_code==404:
            return blank_history()
        r.raise_for_status()
        return norm_history(pd.read_csv(StringIO(r.text),low_memory=False))
    except Exception:
        return blank_history()

def merge_histories(*frames):
    xs=[norm_history(x) for x in frames if x is not None and len(x)]
    if not xs:
        return blank_history()
    out=pd.concat(xs,ignore_index=True)
    return (
        out.drop_duplicates(["snapshot_id","horse_no"],keep="last")
        .reset_index(drop=True)
    )

COLS = [
    "snapshot_id","recorded_at","app_version",
    "date","course","race_no","race_name","surface","distance","going",
    "horse_no","horse_name","mark","rank",
    "win_prob","top3_prob","base_ai_index","day_adjustment","ai_index",
    "odds","popularity","body_weight","body_weight_diff",
    "snapshot_type","post_time","minutes_before_post","auto_generated",
    "bet_style","bet_budget","bet_plan_json",
    "bet_stake","bet_payout","bet_profit","bet_roi",
    "bet_hit_count","bet_ticket_count","bet_refund_count",
    "bet_status","bet_ticket_results_json","bet_settled_at",
    "actual_finish","actual_win","actual_top3","settled_at","result_url",
]

def blank_history():
    return pd.DataFrame(columns=COLS)

def norm_history(df):
    if df is None or len(df)==0:
        return blank_history()
    out=df.copy()
    text_cols=[
        "snapshot_id","recorded_at","app_version","date","course","race_no",
        "race_name","surface","going","horse_name","mark","snapshot_type","post_time",
        "bet_style","bet_plan_json","bet_status","bet_ticket_results_json","bet_settled_at",
        "settled_at","result_url",
    ]
    for c in COLS:
        if c not in out.columns:
            out[c]="" if c in text_cols else np.nan
    for c in text_cols:
        out[c]=out[c].fillna("").astype(str)
    return out[COLS]

def secret_get(secrets,key,default=""):
    value,_=resolve_secret(secrets,key,default)
    return value

class HistoryBackend:
    """
    Persistent mode writes to the dedicated prediction-history branch by
    default. Keeping runtime data off main avoids Streamlit redeploys while
    allowing the app and automation to share the same persistent store.
    """
    def __init__(self,secrets=None):
        self.token,self.token_source=resolve_token(secrets)
        self.repo,self.repo_source=resolve_history_repo(secrets)
        self.branch,self.branch_source=resolve_history_branch(secrets)
        self.path=str(secret_get(secrets,"GITHUB_HISTORY_PATH","prediction_history.csv") or "prediction_history.csv").strip()
        self.mode="github" if self.token and self.repo else "local"

    @property
    def persistent(self):
        return self.mode=="github"

    @property
    def label(self):
        return "履歴専用GitHubへ永続保存" if self.persistent else "一時保存＋CSVバックアップ"

    def headers(self):
        return {
            "Authorization":f"Bearer {self.token}",
            "Accept":"application/vnd.github+json",
            "X-GitHub-Api-Version":"2022-11-28",
        }

    def gh_get(self):
        url=f"https://api.github.com/repos/{self.repo}/contents/{self.path}"
        r=requests.get(url,headers=self.headers(),params={"ref":self.branch},timeout=20)
        if r.status_code==404:
            return blank_history(),None
        r.raise_for_status()
        js=r.json()
        raw=base64.b64decode(js.get("content",""))
        df=pd.read_csv(StringIO(raw.decode("utf-8-sig")),low_memory=False) if raw else blank_history()
        return norm_history(df),js.get("sha")

    def _local_store_path(self,path):
        p=DATA_DIR/str(path)
        p.parent.mkdir(parents=True,exist_ok=True)
        return p

    def read_bytes(self,path):
        """Generic store API used by online-learning artifacts."""
        if self.persistent:
            url=f"https://api.github.com/repos/{self.repo}/contents/{path}"
            r=requests.get(
                url,headers=self.headers(),params={"ref":self.branch},timeout=20
            )
            if r.status_code==404:
                return b"",None
            r.raise_for_status()
            js=r.json()
            return base64.b64decode(js.get("content","")),js.get("sha")
        p=self._local_store_path(path)
        return (p.read_bytes(),None) if p.exists() else (b"",None)

    def write_bytes(self,path,data,message):
        """Generic persistent writer used by online-learning artifacts."""
        p=self._local_store_path(path)
        p.write_bytes(data)
        if not self.persistent:
            return {"mode":"local","path":str(p)}
        _,sha=self.read_bytes(path)
        payload={
            "message":message,
            "content":base64.b64encode(data).decode("ascii"),
            "branch":self.branch,
        }
        if sha:
            payload["sha"]=sha
        url=f"https://api.github.com/repos/{self.repo}/contents/{path}"
        r=requests.put(url,headers=self.headers(),json=payload,timeout=30)
        r.raise_for_status()
        return {"mode":"github","path":path}

    def read_csv(self,path):
        raw,_=self.read_bytes(path)
        if not raw:
            return pd.DataFrame()
        return pd.read_csv(StringIO(raw.decode("utf-8-sig")),low_memory=False)

    def write_csv(self,path,df,message):
        raw=df.to_csv(index=False).encode("utf-8-sig")
        return self.write_bytes(path,raw,message)

    def read_json(self,path):
        raw,_=self.read_bytes(path)
        return json.loads(raw.decode("utf-8")) if raw else None

    def write_json(self,path,obj,message):
        raw=json.dumps(obj,ensure_ascii=False,indent=2).encode("utf-8")
        return self.write_bytes(path,raw,message)

    def load(self):
        if self.persistent:
            try:
                df,_=self.gh_get()
                return df
            except Exception:
                pass
        if LOCAL_HISTORY.exists():
            try:
                return norm_history(pd.read_csv(LOCAL_HISTORY,low_memory=False))
            except Exception:
                pass
        return blank_history()

    def save(self,df):
        df=norm_history(df)
        DATA_DIR.mkdir(parents=True,exist_ok=True)
        df.to_csv(LOCAL_HISTORY,index=False,encoding="utf-8-sig")

        if not self.persistent:
            return {"mode":"local"}

        _,sha=self.gh_get()
        raw=df.to_csv(index=False).encode("utf-8-sig")
        payload={
            "message":"Update keiba AI evaluation history",
            "content":base64.b64encode(raw).decode("ascii"),
            "branch":self.branch,
        }
        if sha:
            payload["sha"]=sha
        url=f"https://api.github.com/repos/{self.repo}/contents/{self.path}"
        r=requests.put(url,headers=self.headers(),json=payload,timeout=30)
        r.raise_for_status()
        return {"mode":"github"}

    def merge_import(self,df):
        base=self.load()
        inc=norm_history(df)
        out=pd.concat([base,inc],ignore_index=True)
        out=out.drop_duplicates(["snapshot_id","horse_no"],keep="last")
        self.save(out)
        return out

def numtext(v):
    x=pd.to_numeric(pd.Series([v]),errors="coerce").iloc[0]
    return "" if pd.isna(x) else f"{float(x):.8f}"

def prediction_snapshot(detail,app_version="1.11"):
    if detail is None or len(detail)==0:
        return blank_history()

    d=detail.copy().sort_values("horse_no")
    f=d.iloc[0]
    date=str(f.get("開催日",f.get("date","")))
    course=str(f.get("競馬場",f.get("course","")))
    race_no=str(f.get("レース",f.get("race_no","")))
    race_name=str(f.get("race_name","")).strip()

    hash_rows=[]
    for _,r in d.iterrows():
        hash_rows.append("|".join([
            str(r.get("horse_no","")),str(r.get("horse_name","")),
            str(r.get("印","")),numtext(r.get("win_prob")),
            numtext(r.get("top3_prob")),numtext(r.get("ai_index")),
            numtext(r.get("odds")),numtext(r.get("body_weight")),
            str(r.get("going","")),
            str(r.get("snapshot_type","manual")),
            str(r.get("bet_style","")),str(r.get("bet_budget","")),
            str(r.get("bet_plan_json","")),
        ]))
    digest=hashlib.sha256("\n".join(hash_rows).encode("utf-8")).hexdigest()[:16]
    sid=f"{date}|{course}|{race_no}|{digest}"
    now=datetime.now(JST).isoformat(timespec="seconds")

    rows=[]
    for _,r in d.iterrows():
        rows.append({
            "snapshot_id":sid,"recorded_at":now,"app_version":app_version,
            "date":date,"course":course,"race_no":race_no,"race_name":race_name,
            "surface":r.get("surface",""),"distance":r.get("distance",np.nan),
            "going":r.get("going",""),"horse_no":r.get("horse_no",np.nan),
            "horse_name":r.get("horse_name",""),"mark":r.get("印",""),
            "rank":r.get("順位",np.nan),"win_prob":r.get("win_prob",np.nan),
            "top3_prob":r.get("top3_prob",np.nan),
            "base_ai_index":r.get("基礎AI指数",r.get("ai_index",np.nan)),
            "day_adjustment":r.get("当日補正",0.0),"ai_index":r.get("ai_index",np.nan),
            "odds":r.get("odds",np.nan),"popularity":r.get("popularity",np.nan),
            "body_weight":r.get("body_weight",np.nan),
            "body_weight_diff":r.get("body_weight_diff",np.nan),
            "snapshot_type":r.get("snapshot_type","manual"),
            "post_time":r.get("post_time",""),
            "minutes_before_post":r.get("minutes_before_post",np.nan),
            "auto_generated":r.get("auto_generated",False),
            "bet_style":r.get("bet_style",""),
            "bet_budget":r.get("bet_budget",np.nan),
            "bet_plan_json":r.get("bet_plan_json",""),
            "bet_stake":np.nan,"bet_payout":np.nan,"bet_profit":np.nan,
            "bet_roi":np.nan,"bet_hit_count":np.nan,"bet_ticket_count":np.nan,
            "bet_refund_count":np.nan,"bet_status":"",
            "bet_ticket_results_json":"","bet_settled_at":"",
            "actual_finish":np.nan,"actual_win":np.nan,"actual_top3":np.nan,
            "settled_at":"","result_url":"",
        })
    return norm_history(pd.DataFrame(rows))

def save_prediction_if_new(backend,detail,app_version="1.11"):
    snap=prediction_snapshot(detail,app_version)
    if snap.empty:
        return {"saved":False,"message":"予想データなし"}
    sid=str(snap.iloc[0]["snapshot_id"])
    hist=backend.load()
    if sid in set(hist["snapshot_id"].astype(str)):
        return {"saved":False,"message":"記録済み","snapshot_id":sid}
    out=pd.concat([hist,snap],ignore_index=True)
    backend.save(out)
    return {"saved":True,"message":"記録しました","snapshot_id":sid}

def save_course_batch_predictions(
    backend,
    detail,
    app_version="1.11",
    snapshot_type="course_batch",
    learning_store=None,
    current_champion_version="",
):
    """
    Save one immutable venue-wide prediction snapshot per race.

    When learning_store is supplied, the exact pre-result feature rows used for
    each newly saved course_batch prediction are frozen at the same time. Those
    rows are later labelled by settle_day_snapshots after the official result is
    known, preventing post-race feature leakage.
    """
    if detail is None or len(detail)==0:
        return {
            "saved_races":0,"existing_races":0,"saved_rows":0,
            "learning_saved_rows":0,
            "message":"一括予想データなし",
        }

    d=detail.copy()
    if "開催日" not in d and "date" in d:
        d["開催日"]=d["date"]
    if "競馬場" not in d and "course" in d:
        d["競馬場"]=d["course"]
    if "レース" not in d and "race_no" in d:
        d["レース"]=d["race_no"]

    d["snapshot_type"]=snapshot_type
    d["auto_generated"]=False

    hist=backend.load()
    hist=norm_history(hist)

    existing=set()
    if len(hist):
        h=hist[hist["snapshot_type"].astype(str)==snapshot_type].copy()
        for _,r in h[["date","course","race_no"]].drop_duplicates().iterrows():
            existing.add((
                str(r["date"]),
                str(r["course"]),
                str(r["race_no"]).replace("R",""),
            ))

    new=[]
    learning_new=[]
    existing_count=0
    group_cols=["開催日","競馬場","レース"]
    for (_,_,_),g in d.groupby(group_cols,sort=False,dropna=False):
        f=g.iloc[0]
        key=(
            str(f.get("開催日","")),
            str(f.get("競馬場","")),
            str(f.get("レース","")).replace("R",""),
        )
        if key in existing:
            existing_count+=1
            continue
        snap=prediction_snapshot(g,app_version)
        if len(snap):
            new.append(snap)
            existing.add(key)
            if learning_store is not None:
                try:
                    from online_learning import learning_rows_from_detail
                    learn=learning_rows_from_detail(
                        g,snap,current_champion_version
                    )
                    if len(learn):
                        learn["feature_source"]="streamlit_course_batch"
                        learning_new.append(learn)
                except Exception:
                    # Prediction history is still valuable even if a learning
                    # artifact cannot be prepared; the caller gets a warning
                    # from the learning save step below.
                    pass

    if not new:
        return {
            "saved_races":0,
            "existing_races":existing_count,
            "saved_rows":0,
            "learning_saved_rows":0,
            "message":"一括予想はすでに答え合わせ用に保存済みです。",
        }

    out=pd.concat([hist]+new,ignore_index=True)
    out=out.drop_duplicates(["snapshot_id","horse_no"],keep="last")
    save_info=backend.save(out) or {}

    learning_saved_rows=0
    learning_error=""
    if learning_store is not None and learning_new:
        try:
            from online_learning import (
                load_learning_rows, merge_learning_rows, save_learning_rows
            )
            learning_existing=load_learning_rows(learning_store)
            learning_merged=merge_learning_rows(
                learning_existing,*learning_new
            )
            save_learning_rows(
                learning_store,
                learning_merged,
                f"Freeze course-batch learning features {d.iloc[0].get('開催日','')}",
            )
            learning_saved_rows=int(sum(len(x) for x in learning_new))
        except Exception as e:
            learning_error=str(e)

    return {
        "saved_races":len(new),
        "existing_races":existing_count,
        "saved_rows":int(sum(len(x) for x in new)),
        "save_mode":save_info.get("mode",""),
        "learning_saved_rows":learning_saved_rows,
        "learning_error":learning_error,
        "message":f"一括予想 {len(new)}R を答え合わせ用に固定保存しました。",
    }

def generic_result_parse(html):
    parsed=parse_jra_result_html(html)
    if len(parsed.get("rows",pd.DataFrame())):
        return parsed
    try:
        tabs=pd.read_html(StringIO(html))
    except Exception:
        return {"rows":pd.DataFrame()}
    for tab in tabs:
        t=_flatten_columns(tab)
        cols=list(t.columns)
        cf=_find_col(cols,["着順","着"])
        cn=_find_col(cols,["馬番","馬 番"])
        if not cf or not cn:
            continue
        rows=[]
        for _,r in t.iterrows():
            fm=re.search(r"^\s*(\d+)",str(r.get(cf,"")))
            nm=re.search(r"(\d+)",str(r.get(cn,"")))
            if fm and nm:
                rows.append({"finish":int(fm.group(1)),"horse_no":int(nm.group(1))})
        if rows:
            return {"rows":pd.DataFrame(rows)}
    return {"rows":pd.DataFrame()}

def fetch_race_result(date_iso,course,race_no,official_entry_urls=None,race_id_map=None):
    s=_session()
    rn=int(str(race_no).upper().replace("R","").strip())
    errors=[]

    # JRA official first. Older session state may not have official URLs,
    # so rebuild them at answer-check time.
    official=dict(official_entry_urls or {})
    if not official:
        try:
            official=collect_official_entry_urls(
                pd.Timestamp(date_iso).date(), [str(course)], s
            )
        except Exception as e:
            errors.append(f"JRA公式リンク再探索={e}")

    # Even if some URLs were supplied, ensure the selected race is present.
    has_selected=False
    for rid in official:
        try:
            if VENUE_CODE.get(rid[4:6],"")==str(course) and int(rid[-2:])==rn:
                has_selected=True
                break
        except Exception:
            pass
    if not has_selected:
        try:
            refreshed=collect_official_entry_urls(
                pd.Timestamp(date_iso).date(), [str(course)], s
            )
            official.update(refreshed)
        except Exception as e:
            errors.append(f"JRA公式リンク補完={e}")

    for rid,entry in official.items():
        try:
            if VENUE_CODE.get(rid[4:6],"")!=str(course) or int(rid[-2:])!=rn:
                continue
            u=_result_url_from_entry(entry,s)
            if not u:
                errors.append(f"JRA公式 {rid}: 出馬表から結果リンクを発見できません")
                continue
            r=s.get(u,timeout=20)
            r.raise_for_status()
            p=generic_result_parse(r.text)
            if len(p.get("rows",pd.DataFrame())):
                p.update({"result_url":u,"source":"JRA公式"})
                return p
            errors.append(f"JRA公式 {rid}: 結果表を解析できません")
        except Exception as e:
            errors.append(f"JRA公式取得={e}")

    # Public fallback by race_id.
    rid=None
    for k,v in (race_id_map or {}).items():
        try:
            kc,kr=k.split("|",1)
            if kc==str(course) and int(str(kr).replace("R",""))==rn:
                rid=v
                break
        except Exception:
            pass
    if rid:
        try:
            u=f"https://race.netkeiba.com/race/result.html?race_id={rid}"
            r=s.get(u,timeout=20)
            r.raise_for_status()
            p=generic_result_parse(r.text)
            if len(p.get("rows",pd.DataFrame())):
                p.update({"result_url":u,"source":"公開レース結果"})
                return p
            errors.append(f"予備結果 {rid}: 結果表を解析できません")
        except Exception as e:
            errors.append(f"予備結果取得={e}")

    detail=" / ".join(errors[-4:]) if errors else "結果リンクを作成できません"
    raise RuntimeError(
        "結果ページを取得できませんでした。"
        " JRA公式結果が公開済みの場合は「最新データに更新」を1回押してから再試行してください。"
        f" 詳細: {detail}"
    )

def settle_latest_snapshot(backend,date_iso,course,race_no,official_entry_urls=None,race_id_map=None):
    hist=backend.load()
    if hist.empty:
        raise RuntimeError("保存済み予想がありません。")
    target=hist[
        (hist["date"].astype(str)==str(date_iso)) &
        (hist["course"].astype(str)==str(course)) &
        (hist["race_no"].astype(str).str.replace("R","",regex=False)
         ==str(race_no).replace("R",""))
    ].copy()
    if target.empty:
        raise RuntimeError("このレースの保存済み予想がありません。")

    target["_dt"]=pd.to_datetime(target["recorded_at"],errors="coerce")
    snap_times=target.groupby("snapshot_id")["_dt"].max().sort_values()
    sid=str(snap_times.index[-1])

    result=fetch_race_result(
        date_iso,course,race_no,
        official_entry_urls=official_entry_urls,
        race_id_map=race_id_map,
    )
    res=result["rows"].copy()
    finish_map={}
    for _,r in res.iterrows():
        no=pd.to_numeric(pd.Series([r.get("horse_no")]),errors="coerce").iloc[0]
        fi=pd.to_numeric(pd.Series([r.get("finish")]),errors="coerce").iloc[0]
        if pd.notna(no) and pd.notna(fi):
            finish_map[int(no)]=int(fi)

    mask=hist["snapshot_id"].astype(str)==sid
    now=datetime.now(JST).isoformat(timespec="seconds")
    matched=0
    for idx in hist.index[mask]:
        no=pd.to_numeric(pd.Series([hist.at[idx,"horse_no"]]),errors="coerce").iloc[0]
        if pd.isna(no) or int(no) not in finish_map:
            continue
        fi=finish_map[int(no)]
        hist.at[idx,"actual_finish"]=fi
        hist.at[idx,"actual_win"]=1 if fi==1 else 0
        hist.at[idx,"actual_top3"]=1 if fi<=3 else 0
        hist.at[idx,"settled_at"]=now
        hist.at[idx,"result_url"]=result.get("result_url","")
        matched+=1

    if matched<3:
        raise RuntimeError("結果と予想の照合数が不足しています。")
    backend.save(hist)
    return {"matched":matched,"source":result.get("source",""),"snapshot_id":sid}

def _race_no_int(v):
    try:
        return int(str(v).upper().replace("R","").strip())
    except Exception:
        return 999

def _latest_snapshot_id_for_race(hist, date_iso, course, race_no):
    q=hist[
        (hist["date"].astype(str)==str(date_iso)) &
        (hist["course"].astype(str)==str(course)) &
        (hist["race_no"].astype(str).str.replace("R","",regex=False)
         ==str(race_no).replace("R",""))
    ].copy()
    if q.empty:
        return ""

    # User-facing venue-wide batch prediction is the answer-check baseline.
    # Other snapshots are retained only as fallback when no batch snapshot exists.
    pri={"course_batch":4,"pre_race":3,"morning":2,"manual":1}
    q["_pri"]=q["snapshot_type"].astype(str).map(pri).fillna(0)
    best=q["_pri"].max()
    q=q[q["_pri"]==best].copy()
    q["_dt"]=pd.to_datetime(q["recorded_at"],errors="coerce")
    times=q.groupby("snapshot_id")["_dt"].max().sort_values()
    return str(times.index[-1]) if len(times) else ""

def _mark_answer_fields(snap, marks=("◎","○","▲")):
    """Return saved mark horses and their exact 1st/2nd/3rd result fields."""
    out={}
    if snap is None or len(snap)==0:
        for mark in marks:
            out.update({
                f"{mark}馬":"",
                f"{mark}着順":"",
                f"{mark}1着":"判定不可",
                f"{mark}2着":"判定不可",
                f"{mark}3着":"判定不可",
                f"{mark}3着内":"判定不可",
            })
        return out

    ranked=snap.copy()
    ranked["_rank_num"]=pd.to_numeric(ranked.get("rank"),errors="coerce")
    rank_fallback={"◎":1,"○":2,"▲":3}
    for mark in marks:
        q=ranked[ranked["mark"].astype(str)==mark].copy()
        if q.empty and mark in rank_fallback:
            q=ranked[ranked["_rank_num"]==rank_fallback[mark]].copy()
        if q.empty:
            out.update({
                f"{mark}馬":"",
                f"{mark}着順":"",
                f"{mark}1着":"判定不可",
                f"{mark}2着":"判定不可",
                f"{mark}3着":"判定不可",
                f"{mark}3着内":"判定不可",
            })
            continue
        r=q.iloc[0]
        no=pd.to_numeric(pd.Series([r.get("horse_no")]),errors="coerce").iloc[0]
        no_text="" if pd.isna(no) else str(int(no))
        name=str(r.get("horse_name","") or "").strip()
        horse_text=(no_text+" "+name).strip()
        fi=pd.to_numeric(pd.Series([r.get("actual_finish")]),errors="coerce").iloc[0]
        if pd.isna(fi):
            out.update({
                f"{mark}馬":horse_text,
                f"{mark}着順":"",
                f"{mark}1着":"未判定",
                f"{mark}2着":"未判定",
                f"{mark}3着":"未判定",
                f"{mark}3着内":"未判定",
            })
            continue
        fi=int(fi)
        out.update({
            f"{mark}馬":horse_text,
            f"{mark}着順":fi,
            f"{mark}1着":"的中" if fi==1 else "不的中",
            f"{mark}2着":"的中" if fi==2 else "不的中",
            f"{mark}3着":"的中" if fi==3 else "不的中",
            f"{mark}3着内":"的中" if fi<=3 else "不的中",
        })
    return out


def _main_answer_fields(snap):
    """Backward-compatible ◎ fields plus ○/▲ exact finishing fields."""
    out=_mark_answer_fields(snap)
    out["◎本命"]=out.get("◎馬","")
    out["◎単勝"]=out.get("◎1着","判定不可")
    return out

def settle_day_snapshots(
    backend,
    date_iso,
    official_entry_urls=None,
    race_id_map=None,
    expected_races=None,
    progress_callback=None,
    learning_store=None,
    run_learning=True,
):
    """
    Check all races for one day, persist the official results, and optionally
    attach those results to the frozen pre-race learning rows.

    The course_batch snapshot is evaluated first; other prediction types are
    fallback only. Prediction history and learning rows are each written once
    after the whole day is processed.
    """
    hist=backend.load()
    if hist.empty and not expected_races:
        raise RuntimeError("保存済み予想がありません。")

    date_iso=str(date_iso)

    learning=None
    learning_changed=False
    learning_error=""
    if learning_store is not None:
        try:
            from online_learning import load_learning_rows
            learning=load_learning_rows(learning_store)
        except Exception as e:
            learning_error=f"学習データ読込失敗: {e}"

    saved=[]
    if not hist.empty:
        q=hist[hist["date"].astype(str)==date_iso].copy()
        if len(q):
            saved=list(
                q[["course","race_no"]]
                .drop_duplicates()
                .itertuples(index=False,name=None)
            )

    pairs=[]
    seen=set()
    for item in list(expected_races or []) + saved:
        try:
            course,race_no=item[0],item[1]
        except Exception:
            continue
        key=(str(course),str(race_no))
        if key not in seen:
            seen.add(key)
            pairs.append(key)

    if not pairs:
        raise RuntimeError("この開催日の予想履歴がありません。")

    pairs=sorted(pairs,key=lambda x:(x[0],_race_no_int(x[1])))

    rows=[]
    changed=False
    settled_sids=[]
    total=len(pairs)
    now=datetime.now(JST).isoformat(timespec="seconds")

    def _apply_learning_labels(snapshot_id,finish_map,result_url):
        nonlocal learning,learning_changed,learning_error
        if learning_store is None or learning is None:
            return
        try:
            from online_learning import apply_learning_result
            learning,learn_changed=apply_learning_result(
                learning,snapshot_id,finish_map,result_url,now
            )
            learning_changed=learning_changed or bool(learn_changed)
        except Exception as e:
            learning_error=f"学習ラベル反映失敗: {e}"

    for i,(course,race_no) in enumerate(pairs, start=1):
        if progress_callback:
            try:
                progress_callback(
                    i,total,
                    f"{course} {race_no} を確認中… {i}/{total}"
                )
            except Exception:
                pass

        sid=_latest_snapshot_id_for_race(
            hist,date_iso,course,race_no
        ) if not hist.empty else ""

        if not sid:
            rows.append({
                "競馬場":course,"レース":race_no,
                "状態":"予想履歴なし","照合頭数":0,
                "取得元":"","メッセージ":"事前に保存された予想がありません。",
            })
            continue

        mask=hist["snapshot_id"].astype(str)==sid
        snap=hist.loc[mask].copy()
        actual=pd.to_numeric(snap["actual_finish"],errors="coerce")
        finish_done=int(actual.notna().sum()) >= 3

        if finish_done:
            finish_map={}
            for _,r in snap.iterrows():
                no=pd.to_numeric(
                    pd.Series([r.get("horse_no")]),errors="coerce"
                ).iloc[0]
                fi=pd.to_numeric(
                    pd.Series([r.get("actual_finish")]),errors="coerce"
                ).iloc[0]
                if pd.notna(no) and pd.notna(fi):
                    finish_map[int(no)]=int(fi)
            _apply_learning_labels(
                sid,finish_map,str(snap.iloc[0].get("result_url","") or "")
            )
            answer=_main_answer_fields(snap)
            rows.append({
                "競馬場":course,"レース":race_no,
                "状態":"照合済み","照合頭数":int(actual.notna().sum()),
                **answer,
                "取得元":"保存済み",
                "メッセージ":"着順を照合済みです。",
            })
            continue

        try:
            result=fetch_race_result(
                date_iso,course,race_no,
                official_entry_urls=official_entry_urls,
                race_id_map=race_id_map,
            )
            res=result["rows"].copy()
            finish_map={}
            for _,r in res.iterrows():
                no=pd.to_numeric(
                    pd.Series([r.get("horse_no")]),errors="coerce"
                ).iloc[0]
                fi=pd.to_numeric(
                    pd.Series([r.get("finish")]),errors="coerce"
                ).iloc[0]
                if pd.notna(no) and pd.notna(fi):
                    finish_map[int(no)]=int(fi)

            matched=0
            for idx in hist.index[mask]:
                no=pd.to_numeric(
                    pd.Series([hist.at[idx,"horse_no"]]),errors="coerce"
                ).iloc[0]
                if pd.isna(no) or int(no) not in finish_map:
                    continue
                fi=finish_map[int(no)]
                hist.at[idx,"actual_finish"]=fi
                hist.at[idx,"actual_win"]=1 if fi==1 else 0
                hist.at[idx,"actual_top3"]=1 if fi<=3 else 0
                hist.at[idx,"settled_at"]=now
                hist.at[idx,"result_url"]=result.get("result_url","")
                matched+=1

            final_matched=int(
                pd.to_numeric(
                    hist.loc[mask,"actual_finish"],errors="coerce"
                ).notna().sum()
            )
            if final_matched < 3:
                rows.append({
                    "競馬場":course,"レース":race_no,
                    "状態":"取得失敗","照合頭数":final_matched,
                    "取得元":result.get("source",""),
                    "メッセージ":"結果は取得できましたが、予想との照合数が不足しています。",
                })
                continue

            changed=True
            settled_sids.append(sid)
            _apply_learning_labels(
                sid,finish_map,result.get("result_url","")
            )
            answer=_main_answer_fields(hist.loc[mask])
            rows.append({
                "競馬場":course,"レース":race_no,
                "状態":"照合完了","照合頭数":final_matched,
                **answer,
                "取得元":result.get("source",""),
                "メッセージ":"",
            })

        except Exception as e:
            rows.append({
                "競馬場":course,"レース":race_no,
                "状態":"未公開・取得失敗","照合頭数":0,
                "取得元":"",
                "メッセージ":str(e),
            })

    save_info={"mode":getattr(backend,"mode","")}
    if changed:
        save_info=backend.save(hist) or save_info

        # Do not merely assume the write worked. Reload the persistent backend
        # and confirm that every newly settled snapshot still has result rows.
        verified=backend.load()
        for sid in settled_sids:
            q=verified[verified["snapshot_id"].astype(str)==str(sid)]
            n=int(pd.to_numeric(q["actual_finish"],errors="coerce").notna().sum())
            if n < 3:
                raise RuntimeError(
                    f"答え合わせ結果の保存確認に失敗しました: {sid}"
                )

    learning_saved=False
    if learning_store is not None and learning is not None and learning_changed:
        try:
            from online_learning import save_learning_rows
            save_learning_rows(
                learning_store,learning,
                f"Save answer-check learning labels {date_iso}"
            )
            learning_saved=True
        except Exception as e:
            learning_error=f"学習データ保存失敗: {e}"

    learning_result={}
    if (
        learning_store is not None and
        learning_saved and
        run_learning
    ):
        try:
            from online_learning import run_online_learning
            learning_result=run_online_learning(learning_store) or {}
        except Exception as e:
            learning_error=f"AI強化判定失敗: {e}"

    report=pd.DataFrame(rows)
    counts=report["状態"].value_counts().to_dict() if len(report) else {}
    evaluated=report[
        report["状態"].isin(["照合完了","照合済み"])
    ].copy() if len(report) else pd.DataFrame()
    win_hits=int((evaluated.get("◎単勝",pd.Series(dtype=object))=="的中").sum()) if len(evaluated) else 0
    second_hits=int((evaluated.get("◎2着",pd.Series(dtype=object))=="的中").sum()) if len(evaluated) else 0
    top3_hits=int((evaluated.get("◎3着内",pd.Series(dtype=object))=="的中").sum()) if len(evaluated) else 0
    evaluated_races=int(len(evaluated))
    mark_position_stats={}
    for mark in ("◎","○","▲"):
        stats={}
        for pos in (1,2,3):
            col=f"{mark}{pos}着"
            hits=int((evaluated.get(col,pd.Series(dtype=object))=="的中").sum()) if len(evaluated) else 0
            stats[f"{pos}着数"]=hits
            stats[f"{pos}着率"]=hits/evaluated_races if evaluated_races else np.nan
        top3_col=f"{mark}3着内"
        top3_mark_hits=int((evaluated.get(top3_col,pd.Series(dtype=object))=="的中").sum()) if len(evaluated) else 0
        stats["3着内数"]=top3_mark_hits
        stats["3着内率"]=top3_mark_hits/evaluated_races if evaluated_races else np.nan
        mark_position_stats[mark]=stats
    return {
        "date":date_iso,
        "total":int(len(report)),
        "completed":int(counts.get("照合完了",0)),
        "already":int(counts.get("照合済み",0)),
        "failed":int(counts.get("未公開・取得失敗",0)+counts.get("取得失敗",0)),
        "no_prediction":int(counts.get("予想履歴なし",0)),
        "evaluated_races":evaluated_races,
        "main_win_hits":win_hits,
        "main_second_hits":second_hits,
        "main_top3_hits":top3_hits,
        "main_win_rate":win_hits/evaluated_races if evaluated_races else np.nan,
        "main_second_rate":second_hits/evaluated_races if evaluated_races else np.nan,
        "main_top3_rate":top3_hits/evaluated_races if evaluated_races else np.nan,
        "mark_position_stats":mark_position_stats,
        "save_mode":save_info.get("mode",getattr(backend,"mode","")),
        "saved_result_races":len(settled_sids),
        "save_verified":bool(not changed or settled_sids),
        "learning_saved":learning_saved,
        "learning_error":learning_error,
        "learning_result":learning_result,
        "report":report,
    }

def settled_latest(history):
    h=norm_history(history)
    h["actual_finish_num"]=pd.to_numeric(h["actual_finish"],errors="coerce")
    h=h[h["actual_finish_num"].notna()].copy()
    if h.empty:
        return h
    pri={"course_batch":4,"pre_race":3,"morning":2,"manual":1}
    h["_pri"]=h["snapshot_type"].map(pri).fillna(0)
    h["_dt"]=pd.to_datetime(h["recorded_at"],errors="coerce")
    racecols=["date","course","race_no"]
    snap=h[
        ["snapshot_id","recorded_at","snapshot_type","_pri","_dt"]+racecols
    ].drop_duplicates()
    snap=snap.sort_values(["date","course","race_no","_pri","_dt"])
    snap=snap.groupby(racecols,as_index=False).tail(1)
    keep=set(snap["snapshot_id"].astype(str))
    return h[h["snapshot_id"].astype(str).isin(keep)].copy()
def evaluation_metrics(history):
    h=settled_latest(history)
    if h.empty:
        return {
            "races":0,"horses":0,
            "main_win_hits":0,"main_second_hits":0,"main_top3_hits":0,
            "main_win_rate":np.nan,"main_second_rate":np.nan,"main_top3_rate":np.nan,
            "brier_win":np.nan,"brier_top3":np.nan,"log_loss":np.nan,
        },h

    for c in ["win_prob","top3_prob","actual_win","actual_top3","odds"]:
        h[c]=pd.to_numeric(h[c],errors="coerce")
    races=int(h[["date","course","race_no"]].drop_duplicates().shape[0])
    main=h[h["mark"].astype(str)=="◎"].copy()
    if len(main):
        main["actual_finish_num"]=pd.to_numeric(
            main["actual_finish"],errors="coerce"
        )
    eps=1e-9
    wp=h["win_prob"].clip(eps,1-eps)
    tp=h["top3_prob"].clip(eps,1-eps)
    winner=h[h["actual_win"]==1]
    ll=float((-np.log(winner["win_prob"].clip(eps,1))).mean()) if len(winner) else np.nan

    return {
        "races":races,"horses":int(len(h)),
        "main_win_hits":int(pd.to_numeric(main["actual_win"],errors="coerce").fillna(0).sum()) if len(main) else 0,
        "main_second_hits":int((main["actual_finish_num"]==2).sum()) if len(main) else 0,
        "main_top3_hits":int(pd.to_numeric(main["actual_top3"],errors="coerce").fillna(0).sum()) if len(main) else 0,
        "main_win_rate":float(main["actual_win"].mean()) if len(main) else np.nan,
        "main_second_rate":float((main["actual_finish_num"]==2).mean()) if len(main) else np.nan,
        "main_top3_rate":float(main["actual_top3"].mean()) if len(main) else np.nan,
        "brier_win":float(np.mean((wp-h["actual_win"])**2)),
        "brier_top3":float(np.mean((tp-h["actual_top3"])**2)),
        "log_loss":ll,
    },h
def mark_summary(history):
    h=settled_latest(history)
    if h.empty:
        return pd.DataFrame()
    for c in ["actual_win","actual_top3","actual_finish","win_prob"]:
        h[c]=pd.to_numeric(h[c],errors="coerce")
    h["_first"]=(h["actual_finish"]==1).astype(int)
    h["_second"]=(h["actual_finish"]==2).astype(int)
    h["_third"]=(h["actual_finish"]==3).astype(int)
    out=h.groupby("mark",dropna=False).agg({
        "horse_no":"count",
        "_first":"sum",
        "_second":"sum",
        "_third":"sum",
        "actual_top3":"sum",
        "win_prob":"mean",
    }).reset_index()
    out=out.rename(columns={
        "mark":"印","horse_no":"出走数",
        "_first":"1着数","_second":"2着数","_third":"3着数",
        "actual_top3":"3着内数","win_prob":"平均予測勝率",
    })
    for pos in (1,2,3):
        out[f"{pos}着率"]=out[f"{pos}着数"]/out["出走数"]
    out["勝率"]=out["1着率"]
    out["3着内率"]=out["3着内数"]/out["出走数"]
    order={"◎":1,"○":2,"▲":3,"△":4,"☆":5,"注":6,"×":7}
    out["_o"]=out["印"].map(order).fillna(99)
    cols=[
        "印","出走数","1着数","2着数","3着数","3着内数",
        "1着率","2着率","3着率","3着内率","平均予測勝率",
    ]
    out=out.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    return out[[col for col in cols if col in out.columns]]

def condition_summary(history):
    h=settled_latest(history)
    if h.empty:
        return pd.DataFrame()
    for c in ["actual_win","actual_top3","win_prob"]:
        h[c]=pd.to_numeric(h[c],errors="coerce")
    h=h[h["mark"].astype(str)=="◎"].copy()
    if h.empty:
        return pd.DataFrame()
    out=h.groupby(["course","surface","distance"],dropna=False).agg({
        "snapshot_id":"nunique","actual_win":"mean","actual_top3":"mean","win_prob":"mean"
    }).reset_index()
    return out.rename(columns={
        "course":"競馬場","surface":"馬場","distance":"距離","snapshot_id":"レース数",
        "actual_win":"◎勝率","actual_top3":"◎3着内率","win_prob":"◎平均予測勝率",
    }).sort_values(["レース数","◎3着内率"],ascending=[False,False]).reset_index(drop=True)

def calibration_summary(history):
    h=settled_latest(history)
    if h.empty:
        return pd.DataFrame()
    h["win_prob"]=pd.to_numeric(h["win_prob"],errors="coerce")
    h["actual_win"]=pd.to_numeric(h["actual_win"],errors="coerce")
    bins=[0,.05,.10,.15,.20,.30,.40,.60,1.01]
    labels=["0-5%","5-10%","10-15%","15-20%","20-30%","30-40%","40-60%","60%+"]
    h["勝率帯"]=pd.cut(h["win_prob"],bins=bins,labels=labels,right=False)
    out=h.groupby("勝率帯",observed=True).agg({
        "horse_no":"count","win_prob":"mean","actual_win":"mean"
    }).reset_index()
    out=out.rename(columns={"horse_no":"頭数","win_prob":"平均予測勝率","actual_win":"実勝率"})
    out["差"]=out["実勝率"]-out["平均予測勝率"]
    return out

def training_candidate_csv(history):
    h=settled_latest(history)
    if h.empty:
        return pd.DataFrame()
    cols=[
        "date","course","race_no","race_name","surface","distance","going",
        "horse_no","horse_name","odds","popularity","body_weight","body_weight_diff",
        "mark","rank","win_prob","top3_prob","base_ai_index","day_adjustment","ai_index",
        "actual_finish","actual_win","actual_top3",
    ]
    return h[[c for c in cols if c in h.columns]].copy()

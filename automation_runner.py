from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from auto_data_builder import (
    _session, discover_race_ids, fetch_race_entries, VENUE_CODE,
    unsupported_race_ids,
)
from cloud_features import load_feature_store, enrich_entries_cloud
from batch_predict import batch_predict_day
from race_day_context import (
    fetch_day_contexts, apply_official_going, merge_entry_conditions,
    apply_day_adjustments, fetch_same_day_bias, collect_official_entry_urls
)
from evaluation_store import (
    prediction_snapshot, norm_history, merge_histories,
    apply_bet_settlement
)
from betting_tools import race_bet_plan
from payout_tools import plan_to_json
from github_branch_store import GitHubBranchStore

BASE=Path(__file__).resolve().parent
MODEL_FILE=BASE/"data"/"cloud_model.joblib"
JST=ZoneInfo("Asia/Tokyo")

HISTORY_PATH="automation_data/pre_race_predictions.csv"
SCHEDULE_PREFIX="automation_data/schedules"
AUTO_BET_STYLE=os.environ.get("KEIBA_BET_STYLE","標準")
try:
    AUTO_BET_BUDGET=max(
        100,int(os.environ.get("KEIBA_BET_BUDGET","2000"))//100*100
    )
except Exception:
    AUTO_BET_BUDGET=2000

def now_jst():
    return datetime.now(JST)

def clean(v):
    return re.sub(r"\s+"," ",str(v).replace("\u3000"," ")).strip()

def discover_schedule(target_date,session=None):
    """
    Discover race_id + post time from the public daily race-list page.
    Falls back to discovered race IDs with blank times when the page layout
    changes, so a morning snapshot can still be created.
    """
    s=session or _session()
    ymd=target_date.strftime("%Y%m%d")
    url=f"https://race.netkeiba.com/top/race_list.html?kaisai_date={ymd}"
    schedule={}
    try:
        r=s.get(url,timeout=20)
        r.raise_for_status()
        html=r.content.decode(r.apparent_encoding or "utf-8",errors="replace")
        soup=BeautifulSoup(html,"lxml")
        for a in soup.find_all("a",href=True):
            m=re.search(r"race_id=(\d{12})",a["href"])
            if not m:
                continue
            rid=m.group(1)
            if rid[:4] != target_date.strftime("%Y") or rid[4:6] not in VENUE_CODE:
                continue
            try:
                rn=int(rid[-2:])
            except Exception:
                continue
            if not 1 <= rn <= 12:
                continue

            tm=""
            node=a
            # The time is normally in the race-list item surrounding the link.
            for _ in range(6):
                node=node.parent if node else None
                if node is None:
                    break
                txt=clean(node.get_text(" ",strip=True))
                mm=re.search(r"(?<!\d)(\d{1,2}:\d{2})(?!\d)",txt)
                if mm:
                    tm=mm.group(1)
                    break
            schedule[rid]={
                "race_id":rid,
                "course":VENUE_CODE.get(rid[4:6],""),
                "race_no":f"{rn}R",
                "post_time":tm,
            }
    except Exception:
        pass

    for rid in discover_race_ids(target_date,s):
        if rid not in schedule:
            schedule[rid]={
                "race_id":rid,
                "course":VENUE_CODE.get(rid[4:6],""),
                "race_no":f"{int(rid[-2:])}R",
                "post_time":"",
            }

    unsupported=set(unsupported_race_ids(target_date))
    return [v for k,v in sorted(schedule.items()) if k not in unsupported]

def dt_for_post(target_date,hhmm):
    if not hhmm:
        return None
    try:
        hh,mm=map(int,hhmm.split(":"))
        return datetime(
            target_date.year,target_date.month,target_date.day,
            hh,mm,tzinfo=JST
        )
    except Exception:
        return None

def load_assets():
    pkg=joblib.load(MODEL_FILE)
    store=load_feature_store()
    return pkg,store

def load_existing(store):
    df=store.read_csv(HISTORY_PATH)
    return norm_history(df) if len(df) else norm_history(pd.DataFrame())

def save_history(store,df,message):
    df=norm_history(df)
    store.write_csv(HISTORY_PATH,df,message)

def snapshot_race(
    race_id,target_date,model_pkg,feature_store,
    contexts,snapshot_type,post_time="",minutes_before=np.nan,
    official_entry_urls=None,
):
    s=_session()
    entries=fetch_race_entries(race_id,target_date,s,timeout=20)
    if entries is None or len(entries)==0:
        raise RuntimeError("出走表が空です。")

    contexts=merge_entry_conditions(contexts,entries)
    entries,_=apply_official_going(entries,contexts,target_date)
    features=enrich_entries_cloud(entries,feature_store)
    detail,_=batch_predict_day(
        features,target_date.isoformat(),
        model_pkg["trained_models"],model_pkg["weights"]
    )

    course=str(entries.iloc[0]["course"])
    context=contexts.get(course,{})
    race_no=int(str(entries.iloc[0].get("race_no","0R")).replace("R",""))
    surface=str(entries.iloc[0].get("surface",""))
    bias=fetch_same_day_bias(
        target_date,course,race_no,surface,official_entry_urls or {}
    ) if official_entry_urls else {
        "races_used":0,"inner_score":0.0,"front_score":0.0,
        "summary":"終了済みレースなし","jockey_stats":{}
    }
    detail=apply_day_adjustments(detail,context,bias,enabled=True)
    detail["snapshot_type"]=snapshot_type
    detail["post_time"]=post_time
    detail["minutes_before_post"]=minutes_before
    detail["auto_generated"]=True

    plan,meta=race_bet_plan(
        detail,style=AUTO_BET_STYLE,budget_yen=AUTO_BET_BUDGET
    )
    detail["bet_style"]=AUTO_BET_STYLE
    detail["bet_budget"]=int(meta.get("予算",AUTO_BET_BUDGET))
    detail["bet_plan_json"]=plan_to_json(plan)

    snap=prediction_snapshot(detail,app_version="1.8")
    # prediction_snapshot hashes prediction state; include snapshot type/post time
    # in ID so morning and near-post records can coexist even if probabilities match.
    if len(snap):
        suffix=f"|{snapshot_type}|{post_time or 'NA'}"
        snap["snapshot_id"]=snap["snapshot_id"].astype(str)+suffix
        snap["snapshot_type"]=snapshot_type
        snap["post_time"]=post_time
        snap["minutes_before_post"]=minutes_before
        snap["auto_generated"]=True
    return snap

def backfill_missing_bet_plans(existing):
    """
    Ver.1.7 may already have created a morning snapshot without bet_plan_json.
    Build the plan only from that frozen pre-race prediction, never from results.
    """
    if existing is None or len(existing)==0:
        return existing,False

    out=existing.copy()
    changed=False
    for sid,g in out.groupby("snapshot_id",sort=False):
        stype=str(g.iloc[0].get("snapshot_type",""))
        if stype not in ("morning","pre_race"):
            continue
        current=str(g.iloc[0].get("bet_plan_json","") or "").strip()
        if current not in ("","nan","[]"):
            continue

        detail=g.copy().rename(columns={
            "date":"開催日","course":"競馬場","race_no":"レース",
            "mark":"印","rank":"順位",
        })
        # betting_tools expects these fields.
        if "評価" not in detail:
            detail["評価"]=""
        plan,meta=race_bet_plan(
            detail,style=AUTO_BET_STYLE,budget_yen=AUTO_BET_BUDGET
        )
        pj=plan_to_json(plan)
        mask=out["snapshot_id"].astype(str)==str(sid)
        out.loc[mask,"bet_style"]=AUTO_BET_STYLE
        out.loc[mask,"bet_budget"]=int(meta.get("予算",AUTO_BET_BUDGET))
        out.loc[mask,"bet_plan_json"]=pj
        changed=True
    return out,changed

def choose_due(schedule,target_date,now,mode,existing):
    if mode=="morning":
        done=set(
            zip(
                existing.loc[
                    (existing["date"].astype(str)==target_date.isoformat()) &
                    (existing["snapshot_type"].astype(str)=="morning"),
                    "course"
                ].astype(str),
                existing.loc[
                    (existing["date"].astype(str)==target_date.isoformat()) &
                    (existing["snapshot_type"].astype(str)=="morning"),
                    "race_no"
                ].astype(str),
            )
        ) if len(existing) else set()
        return [
            x for x in schedule
            if (str(x["course"]),str(x["race_no"])) not in done
        ]

    if mode=="pre_race":
        done=set(
            zip(
                existing.loc[
                    (existing["date"].astype(str)==target_date.isoformat()) &
                    (existing["snapshot_type"].astype(str)=="pre_race"),
                    "course"
                ].astype(str),
                existing.loc[
                    (existing["date"].astype(str)==target_date.isoformat()) &
                    (existing["snapshot_type"].astype(str)=="pre_race"),
                    "race_no"
                ].astype(str),
            )
        ) if len(existing) else set()

        due=[]
        for x in schedule:
            key=(str(x["course"]),str(x["race_no"]))
            if key in done:
                continue
            post=dt_for_post(target_date,x.get("post_time",""))
            if post is None:
                continue
            mins=(post-now).total_seconds()/60.0
            # Actions is scheduled every 20 minutes. This window produces one
            # snapshot roughly 15-65 minutes before post, even with moderate delay.
            if 15 <= mins <= 65:
                y=dict(x)
                y["minutes_before_post"]=round(mins,1)
                due.append(y)
        return due

    return []

def run_predictions(mode,target_date,store):
    schedule_path=f"{SCHEDULE_PREFIX}/race_schedule_{target_date.strftime('%Y%m%d')}.json"
    schedule=store.read_json(schedule_path)
    if not schedule:
        schedule=discover_schedule(target_date)
        if schedule:
            store.write_json(
                schedule_path,schedule,
                f"Save race schedule {target_date.isoformat()}"
            )
    if not schedule:
        print("No races discovered; exiting.")
        return

    existing=load_existing(store)
    existing,backfilled=backfill_missing_bet_plans(existing)
    if backfilled:
        save_history(
            store,existing,
            f"Backfill saved bet plans {target_date.isoformat()}"
        )
        print("Backfilled bet plans for existing Ver.1.7 snapshots.")
    now=now_jst()
    due=choose_due(schedule,target_date,now,mode,existing)
    if not due:
        print(f"No due races for {mode}; exiting.")
        return

    courses=sorted(set(x["course"] for x in due if x.get("course")))
    official_urls=collect_official_entry_urls(target_date,courses)
    contexts=fetch_day_contexts(target_date,courses,official_urls)
    model_pkg,feature_store=load_assets()

    new=[]
    for i,x in enumerate(due,1):
        rid=x["race_id"]
        print(f"[{i}/{len(due)}] {x['course']} {x['race_no']} {mode}")
        try:
            mins=x.get("minutes_before_post",np.nan)
            snap=snapshot_race(
                rid,target_date,model_pkg,feature_store,contexts,
                snapshot_type=mode,
                post_time=x.get("post_time",""),
                minutes_before=mins,
                official_entry_urls=official_urls,
            )
            if len(snap):
                new.append(snap)
        except Exception as e:
            print(f"SKIP {rid}: {e}")

    if not new:
        print("No snapshots created.")
        return

    merged=merge_histories(existing,*new)
    save_history(
        store,merged,
        f"Auto-save {mode} predictions {target_date.isoformat()}"
    )
    print(f"Saved {sum(len(x) for x in new)} horse rows / {len(new)} races.")

def settle_all(target_date,store):
    from evaluation_store import fetch_race_result

    hist=load_existing(store)
    hist,backfilled=backfill_missing_bet_plans(hist)
    if backfilled:
        save_history(
            store,hist,
            f"Backfill saved bet plans {target_date.isoformat()}"
        )
    if hist.empty:
        print("No pre-race history.")
        return

    day=hist[hist["date"].astype(str)==target_date.isoformat()].copy()
    if day.empty:
        print("No history for date.")
        return

    # Latest preferred pre-race snapshot for each race:
    # pre_race > morning, then latest recorded_at.
    pri={"pre_race":2,"morning":1}
    day["_pri"]=day["snapshot_type"].map(pri).fillna(0)
    day["_dt"]=pd.to_datetime(day["recorded_at"],errors="coerce")
    snap_meta=day[
        ["snapshot_id","course","race_no","snapshot_type","_pri","_dt"]
    ].drop_duplicates()
    snap_meta=snap_meta.sort_values(["course","race_no","_pri","_dt"])
    chosen=snap_meta.groupby(["course","race_no"],as_index=False).tail(1)

    changed=False
    settled_at=now_jst().isoformat(timespec="seconds")
    for _,m in chosen.iterrows():
        sid=str(m["snapshot_id"])
        mask=hist["snapshot_id"].astype(str)==sid
        finish_done=(
            pd.to_numeric(hist.loc[mask,"actual_finish"],errors="coerce")
            .notna().sum() >= 3
        )
        bet_done=(hist.loc[mask,"bet_status"].astype(str)=="確定").any()
        if finish_done and bet_done:
            continue
        course=str(m["course"]); race_no=str(m["race_no"])
        try:
            result=fetch_race_result(
                target_date.isoformat(),course,race_no,
                official_entry_urls={},race_id_map={}
            )
            res=result["rows"].copy()
            fmap={}
            for _,r in res.iterrows():
                no=pd.to_numeric(pd.Series([r.get("horse_no")]),errors="coerce").iloc[0]
                fi=pd.to_numeric(pd.Series([r.get("finish")]),errors="coerce").iloc[0]
                if pd.notna(no) and pd.notna(fi):
                    fmap[int(no)]=int(fi)
            matched=0
            for idx in hist.index[mask]:
                no=pd.to_numeric(pd.Series([hist.at[idx,"horse_no"]]),errors="coerce").iloc[0]
                if pd.isna(no) or int(no) not in fmap:
                    continue
                fi=fmap[int(no)]
                hist.at[idx,"actual_finish"]=fi
                hist.at[idx,"actual_win"]=1 if fi==1 else 0
                hist.at[idx,"actual_top3"]=1 if fi<=3 else 0
                hist.at[idx,"settled_at"]=settled_at
                hist.at[idx,"result_url"]=result.get("result_url","")
                matched+=1
            if matched>=3 or finish_done:
                bet=apply_bet_settlement(
                    hist,mask,result,settled_at
                )
                changed=True
                roi=bet.get("roi",np.nan)
                roi_text="-" if pd.isna(roi) else f"{float(roi):.1f}%"
                print(
                    f"SETTLED {course} {race_no}: "
                    f"stake={bet.get('stake',0)} "
                    f"payout={bet.get('payout',0)} ROI={roi_text}"
                )
            else:
                print(f"INSUFFICIENT {course} {race_no}")
        except Exception as e:
            print(f"UNSETTLED {course} {race_no}: {e}")

    if changed:
        save_history(
            store,hist,
            f"Auto-settle race results {target_date.isoformat()}"
        )

def determine_mode(requested,now):
    if requested!="auto":
        return requested
    h=now.hour
    if h < 9:
        return "morning"
    if h < 17:
        return "pre_race"
    return "settle"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["auto","morning","pre_race","settle"],default="auto")
    ap.add_argument("--date",default="")
    args=ap.parse_args()

    now=now_jst()
    target_date=pd.Timestamp(args.date).date() if args.date else now.date()
    mode=determine_mode(args.mode,now)

    store=GitHubBranchStore(
        branch=os.environ.get("KEIBA_HISTORY_BRANCH","prediction-history")
    )
    print(f"Mode={mode} Date={target_date} JST={now.isoformat()}")

    if mode in ("morning","pre_race"):
        run_predictions(mode,target_date,store)
    else:
        settle_all(target_date,store)

if __name__=="__main__":
    main()

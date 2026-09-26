from __future__ import annotations

from datetime import date, datetime
from io import StringIO
from urllib.parse import urljoin, unquote
import re
import math
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from auto_data_builder import (
    _session, _jra_signature_for_race_id, all_known_race_ids,
    VENUE_CODE, _flatten_columns, _find_col
)
from batch_predict import mark_for_rank, mark_label

# 2026-09-27 official entry seeds. One official page per venue is enough:
# links for the other races are collected from the page itself.
OFFICIAL_ENTRY_SEEDS = {
    # 2026-09-26: verified JRA official race-card pages.
    # From one page per venue, links for the other races are collected.
    ("2026-09-26","中山"):
        "https://www.jra.go.jp/JRADB/accessD.html?CNAME=pw01dde0106202604080820260926%2FCB",
    ("2026-09-26","阪神"):
        "https://www.jra.go.jp/JRADB/accessD.html?CNAME=pw01dde0109202604080120260926%2FB6",

    # 2026-09-27
    ("2026-09-27","中山"):
        "https://www.jra.go.jp/JRADB/accessD.html?CNAME=pw01dde0106202604091120260927%2F5A",
    ("2026-09-27","阪神"):
        "https://www.jra.go.jp/JRADB/accessD.html?CNAME=pw01dde0109202604090420260927%2F05",
}

# JRA track-information tabs. The course name is parsed from each page,
# so order changes are harmless.
JRA_BABA_URLS = [
    "https://www.jra.go.jp/keiba/baba/",
    "https://www.jra.go.jp/keiba/baba/index2.html",
    "https://www.jra.go.jp/keiba/baba/index3.html",
]

# Approximate racecourse coordinates. Used only for local weather forecast.
COURSE_COORDS = {
    "札幌": (43.076, 141.322),
    "函館": (41.782, 140.775),
    "福島": (37.758, 140.474),
    "新潟": (37.949, 139.187),
    "東京": (35.664, 139.485),
    "中山": (35.726, 139.963),
    "中京": (35.066, 136.965),
    "京都": (34.993, 135.749),
    "阪神": (34.780, 135.363),
    "小倉": (33.887, 130.879),
}

WEATHER_CODES = {
    0:"快晴",1:"晴",2:"一部曇",3:"曇",
    45:"霧",48:"霧",
    51:"弱い霧雨",53:"霧雨",55:"強い霧雨",
    61:"弱い雨",63:"雨",65:"強い雨",
    71:"弱い雪",73:"雪",75:"強い雪",
    80:"弱いにわか雨",81:"にわか雨",82:"強いにわか雨",
    95:"雷雨",96:"雷雨",99:"強い雷雨",
}

def _clean(v):
    return re.sub(r"\s+", " ", str(v).replace("\u3000"," ")).strip()

def _number(v):
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(v))
    return float(m.group()) if m else np.nan

def _next_value_after_heading(soup, label, allowed=None):
    for tag in soup.find_all(["h3","h4","h5","dt","th"]):
        if _clean(tag.get_text(" ",strip=True)) == label:
            for node in tag.find_all_next(limit=12):
                txt = _clean(node.get_text(" ",strip=True)) if hasattr(node,"get_text") else _clean(node)
                if not txt or txt == label:
                    continue
                if allowed is None:
                    return txt
                if txt in allowed:
                    return txt
    return ""

def _section_text(soup, label, max_len=260):
    for tag in soup.find_all(["h2","h3","h4","h5"]):
        if _clean(tag.get_text(" ",strip=True)) == label:
            texts=[]
            for node in tag.find_all_next(limit=10):
                if node is tag:
                    continue
                if getattr(node, "name", "") in ("h2","h3","h4","h5"):
                    break
                txt=_clean(node.get_text(" ",strip=True)) if hasattr(node,"get_text") else ""
                if txt and txt not in texts:
                    texts.append(txt)
                if len(" ".join(texts)) >= max_len:
                    break
            return " ".join(texts)[:max_len]
    return ""

def parse_jra_baba_html(html: str, source_url="") -> dict:
    soup = BeautifulSoup(html, "lxml")
    text = _clean(soup.get_text(" ",strip=True))
    title = _clean(soup.title.get_text(" ",strip=True) if soup.title else "")

    cm = re.search(r"馬場情報（(.+?)競馬場）", title + " " + text[:800])
    course = cm.group(1) if cm else ""

    dm = re.search(r"（(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    race_date = ""
    if dm:
        race_date = f"{int(dm.group(1)):04d}-{int(dm.group(2)):02d}-{int(dm.group(3)):02d}"

    um = re.search(r"馬場状態（(.{1,40}?現在)）", text)
    status_time = um.group(1) if um else ""

    wm = re.search(r"天候[:：]\s*([^\s]+)", text)
    jra_weather = wm.group(1) if wm else ""

    going_values = {"良","稍重","重","不良"}
    turf_going = _next_value_after_heading(soup, "芝", going_values)
    dirt_going = _next_value_after_heading(soup, "ダート", going_values)

    # Fallback for flattened text.
    if not turf_going or not dirt_going:
        m = re.search(r"芝\s*(良|稍重|重|不良).*?ダート\s*(良|稍重|重|不良)", text)
        if m:
            turf_going = turf_going or m.group(1)
            dirt_going = dirt_going or m.group(2)

    moist = {
        "turf_goal":np.nan, "turf_corner":np.nan,
        "dirt_goal":np.nan, "dirt_corner":np.nan,
    }
    try:
        for tab in pd.read_html(StringIO(html)):
            t = _flatten_columns(tab)
            cols = list(t.columns)
            if not any("ゴール前" in str(c) for c in cols):
                continue
            if not any("4コーナー" in str(c) for c in cols):
                continue
            goal_col = next(c for c in cols if "ゴール前" in str(c))
            corner_col = next(c for c in cols if "4コーナー" in str(c))
            first_col = cols[0]
            for _, r in t.iterrows():
                label = _clean(r.get(first_col,""))
                if "芝" in label:
                    moist["turf_goal"] = _number(r.get(goal_col))
                    moist["turf_corner"] = _number(r.get(corner_col))
                elif "ダート" in label:
                    moist["dirt_goal"] = _number(r.get(goal_col))
                    moist["dirt_corner"] = _number(r.get(corner_col))
            if any(pd.notna(v) for v in moist.values()):
                break
    except Exception:
        pass

    # Cushion value is only accepted when JRA explicitly marks a pure numeric
    # option as selected. If the site structure does not expose it safely,
    # leave it blank instead of guessing from the reference scale (12/10/8/7).
    cushion = np.nan
    for sel in soup.find_all("select"):
        opt = sel.find("option", selected=True)
        if opt:
            txt = _clean(opt.get_text(" ",strip=True))
            if re.fullmatch(r"\d{1,2}(?:\.\d)?", txt):
                val = float(txt)
                if 5.0 <= val <= 15.0:
                    parent_txt = _clean(sel.find_parent().get_text(" ",strip=True)) if sel.find_parent() else ""
                    if "クッション" in parent_txt:
                        cushion = val
                        break

    used_course = _section_text(soup, "使用コース", 180)
    turf_state = _section_text(soup, "芝の状態", 260)

    return {
        "course":course,
        "race_date":race_date,
        "status_time":status_time,
        "jra_weather":jra_weather,
        "turf_going":turf_going,
        "dirt_going":dirt_going,
        "cushion":cushion,
        **moist,
        "used_course":used_course,
        "turf_state":turf_state,
        "jra_baba_url":source_url,
    }

def fetch_jra_baba_contexts(target_date: date, courses=None, session=None) -> dict:
    s = session or _session()
    wanted = set(courses or [])
    out = {}
    for url in JRA_BABA_URLS:
        try:
            r=s.get(url,timeout=15)
            r.raise_for_status()
            ctx=parse_jra_baba_html(r.text,url)
            course=ctx.get("course","")
            if not course:
                continue
            if wanted and course not in wanted:
                continue
            ctx["is_target_date"] = (ctx.get("race_date") == target_date.isoformat())
            out[course]=ctx
        except Exception:
            continue
    return out

def _weather_pick(hourly, target_date):
    times = hourly.get("time") or []
    if not times:
        return {}
    # 14:00 gives a representative race-day condition for the main part of card.
    target = f"{target_date.isoformat()}T14:00"
    idx = min(range(len(times)), key=lambda i: abs(
        (pd.Timestamp(times[i]) - pd.Timestamp(target)).total_seconds()
    ))
    def at(name, default=np.nan):
        arr=hourly.get(name) or []
        return arr[idx] if idx < len(arr) else default
    code=at("weather_code",np.nan)
    return {
        "forecast_time":times[idx],
        "temperature_c":_number(at("temperature_2m")),
        "precipitation_mm":_number(at("precipitation")),
        "rain_mm":_number(at("rain")),
        "precip_probability":_number(at("precipitation_probability")),
        "wind_kmh":_number(at("wind_speed_10m")),
        "wind_gust_kmh":_number(at("wind_gusts_10m")),
        "weather_code":int(code) if pd.notna(_number(code)) else None,
        "weather_text":WEATHER_CODES.get(int(code),"") if pd.notna(_number(code)) else "",
    }

def fetch_weather_context(course: str, target_date: date, session=None) -> dict:
    coords=COURSE_COORDS.get(course)
    if not coords:
        return {}
    s=session or requests.Session()
    lat,lon=coords
    params={
        "latitude":lat,"longitude":lon,"timezone":"Asia/Tokyo",
        "start_date":target_date.isoformat(),"end_date":target_date.isoformat(),
        "hourly":",".join([
            "temperature_2m","precipitation","rain","precipitation_probability",
            "weather_code","wind_speed_10m","wind_gusts_10m"
        ])
    }
    try:
        r=s.get("https://api.open-meteo.com/v1/forecast",params=params,timeout=15)
        r.raise_for_status()
        js=r.json()
        out=_weather_pick(js.get("hourly") or {},target_date)
        out["source"]="Open-Meteo"
        return out
    except Exception as e:
        return {"error":str(e),"source":"Open-Meteo"}

def fetch_day_contexts(target_date: date, courses) -> dict:
    courses=list(dict.fromkeys([str(x) for x in courses if str(x)]))
    s=_session()
    jra=fetch_jra_baba_contexts(target_date,courses,s)
    out={}
    for course in courses:
        ctx=dict(jra.get(course,{course:course}))
        ctx["course"]=course
        ctx["weather"]=fetch_weather_context(course,target_date,s)
        out[course]=ctx
    return out

def apply_official_going(entries: pd.DataFrame, contexts: dict, target_date: date):
    out=entries.copy()
    if "going" not in out.columns:
        out["going"]=""
    # Keep string values safe even when a source CSV inferred this column as float.
    out["going"]=out["going"].fillna("").astype(str)
    out["going_source"]="出走表"
    changed=0
    for i,r in out.iterrows():
        course=str(r.get("course",""))
        surface=str(r.get("surface",""))
        ctx=contexts.get(course) or {}
        # Never inject stale official going from a different race date.
        if not ctx.get("is_target_date"):
            continue
        val = ctx.get("turf_going") if surface=="芝" else ctx.get("dirt_going")
        if val in ("良","稍重","重","不良"):
            old=str(r.get("going","")).strip()
            out.at[i,"going"]=val
            out.at[i,"going_source"]="JRA馬場情報"
            if old != val:
                changed += 1
    return out, changed

def collect_official_entry_urls(target_date: date, courses=None, session=None) -> dict:
    s=session or _session()
    wanted=set(courses or [])
    known=all_known_race_ids(target_date)
    by_course={}
    for rid in known:
        course=VENUE_CODE.get(rid[4:6],"")
        if wanted and course not in wanted:
            continue
        by_course.setdefault(course,[]).append(rid)

    result={}
    for course,rids in by_course.items():
        seed=OFFICIAL_ENTRY_SEEDS.get((target_date.isoformat(),course))
        if not seed:
            continue
        try:
            r=s.get(seed,timeout=20)
            r.raise_for_status()
            soup=BeautifulSoup(r.text,"lxml")
            urls=[urljoin(seed,a["href"]) for a in soup.find_all("a",href=True)]
            urls.append(seed)
            pairs=[(u,unquote(u)) for u in dict.fromkeys(urls)]
            for rid in rids:
                sig=_jra_signature_for_race_id(rid,target_date)
                for raw,uq in pairs:
                    if "accessD.html" in uq and sig in uq:
                        result[rid]=raw
                        break
        except Exception:
            continue
    return result

def _result_url_from_entry(entry_url: str, session):
    """
    Resolve the JRA official result page from an official entry page.
    JRA may include accessibility text/whitespace around "レース結果",
    so match by both href and text instead of exact-text equality only.
    """
    try:
        r=session.get(entry_url,timeout=20)
        r.raise_for_status()
        soup=BeautifulSoup(r.text,"lxml")

        candidates=[]
        for a in soup.find_all("a",href=True):
            txt=_clean(a.get_text(" ",strip=True))
            href=urljoin(entry_url,a["href"])
            if "accessS.html" not in href:
                continue
            if "レース結果" in txt:
                return href
            candidates.append(href)

        # Fallback: if there is exactly one accessS result-style link on
        # the race page, use it rather than failing because anchor text changed.
        uniq=list(dict.fromkeys(candidates))
        if len(uniq)==1:
            return uniq[0]
    except Exception:
        pass
    return ""

def parse_jra_result_html(html: str) -> dict:
    soup=BeautifulSoup(html,"lxml")
    text=_clean(soup.get_text(" ",strip=True))
    sm=re.search(r"コース[:：]\s*[\d,]+メートル（(芝|ダート)",text)
    surface=sm.group(1) if sm else ""
    going=""
    gm=re.search(r"(芝|ダート)(良|稍重|重|不良)",text)
    if gm:
        going=gm.group(2)

    table=None
    try:
        for t in pd.read_html(StringIO(html)):
            tt=_flatten_columns(t)
            cols=list(tt.columns)
            joined=" ".join(map(str,cols))
            if "着順" in joined and ("馬番" in joined or "馬 番" in joined):
                table=tt; break
    except Exception:
        table=None
    if table is None:
        return {"surface":surface,"going":going,"rows":pd.DataFrame()}

    cols=list(table.columns)
    c_finish=_find_col(cols,["着順"])
    c_no=_find_col(cols,["馬番","馬 番"])
    c_pass=_find_col(cols,["コーナー 通過順位","通過順位","コーナー"])
    rows=[]
    for _,r in table.iterrows():
        fm=re.search(r"^\s*(\d+)",str(r.get(c_finish,"")))
        nm=re.search(r"(\d+)",str(r.get(c_no,"")))
        if not fm or not nm:
            continue
        finish=int(fm.group(1)); horse_no=int(nm.group(1))
        passage=[]
        if c_pass:
            passage=[int(x) for x in re.findall(r"\d+",str(r.get(c_pass,"")))]
        rows.append({"finish":finish,"horse_no":horse_no,"first_pos":passage[0] if passage else np.nan})
    return {"surface":surface,"going":going,"rows":pd.DataFrame(rows)}

def fetch_same_day_bias(
    target_date: date,
    course: str,
    current_race_no: int,
    surface: str,
    entry_url_map: dict,
    session=None,
) -> dict:
    s=session or _session()
    prior=[]
    for rid,url in (entry_url_map or {}).items():
        if VENUE_CODE.get(rid[4:6],"") != course:
            continue
        rn=int(rid[-2:])
        if rn < int(current_race_no):
            prior.append((rn,rid,url))
    prior.sort()

    top3_norm_no=[]
    top3_norm_pos=[]
    used=0
    sources=[]
    for rn,rid,entry_url in prior:
        result_url=_result_url_from_entry(entry_url,s)
        if not result_url:
            continue
        try:
            rr=s.get(result_url,timeout=15)
            rr.raise_for_status()
            parsed=parse_jra_result_html(rr.text)
            if parsed.get("surface") != surface:
                continue
            df=parsed.get("rows")
            if df is None or len(df)<3:
                continue
            field_size=max(int(df["horse_no"].max()),len(df))
            top=df[df["finish"]<=3].copy()
            if top.empty:
                continue
            denom=max(field_size-1,1)
            top3_norm_no.extend(((top["horse_no"]-1)/denom).clip(0,1).tolist())
            pos=pd.to_numeric(top["first_pos"],errors="coerce")
            pos=pos[pos.notna()]
            if len(pos):
                top3_norm_pos.extend(((pos-1)/denom).clip(0,1).tolist())
            used += 1
            sources.append(result_url)
        except Exception:
            continue

    # Positive inner_score = inside numbers performing better.
    inner_score=0.0
    if top3_norm_no:
        inner_score=float(np.clip((0.5-float(np.mean(top3_norm_no)))*2.0,-1,1))
    # Positive front_score = horses near the front at first recorded corner performing better.
    front_score=0.0
    if top3_norm_pos:
        front_score=float(np.clip((0.5-float(np.mean(top3_norm_pos)))*2.0,-1,1))

    # Small samples should have smaller influence.
    reliability=min(1.0,used/4.0)
    inner_score*=reliability
    front_score*=reliability

    if used == 0:
        summary="終了済み同条件レースがなく、当日バイアス補正なし"
    else:
        gate_txt="内寄り" if inner_score>0.12 else ("外寄り" if inner_score<-0.12 else "内外ほぼ中立")
        pace_txt="前有利" if front_score>0.12 else ("差し有利" if front_score<-0.12 else "脚質ほぼ中立")
        summary=f"{used}レース集計：{gate_txt}／{pace_txt}"

    return {
        "races_used":used,
        "inner_score":inner_score,
        "front_score":front_score,
        "summary":summary,
        "source_urls":sources,
    }

def _surface_moisture(ctx, surface):
    if not ctx:
        return np.nan
    if surface=="芝":
        vals=[ctx.get("turf_goal"),ctx.get("turf_corner")]
    else:
        vals=[ctx.get("dirt_goal"),ctx.get("dirt_corner")]
    vals=[float(v) for v in vals if pd.notna(pd.to_numeric(pd.Series([v]),errors="coerce").iloc[0])]
    return float(np.mean(vals)) if vals else np.nan

def apply_day_adjustments(detail: pd.DataFrame, context: dict, bias: dict, enabled=True):
    d=detail.copy()
    if d.empty:
        return d
    d["基礎AI指数"]=pd.to_numeric(d["ai_index"],errors="coerce")
    d["基礎勝率"]=pd.to_numeric(d["win_prob"],errors="coerce")
    d["当日補正"]=0.0
    d["補正理由"]=""

    if not enabled:
        return d

    field_size=max(len(d),2)
    surface=str(d.iloc[0].get("surface",""))
    official_going=str(d.iloc[0].get("going",""))
    moisture=_surface_moisture(context,surface)
    cushion=pd.to_numeric(pd.Series([context.get("cushion")]),errors="coerce").iloc[0]
    weather=(context or {}).get("weather") or {}
    precip=pd.to_numeric(pd.Series([weather.get("precipitation_mm")]),errors="coerce").iloc[0]
    precip_prob=pd.to_numeric(pd.Series([weather.get("precip_probability")]),errors="coerce").iloc[0]

    inner=float((bias or {}).get("inner_score",0) or 0)
    front=float((bias or {}).get("front_score",0) or 0)

    adjs=[]
    reasons=[]
    for _,r in d.iterrows():
        adj=0.0
        rs=[]

        # 1) Historical aptitude for today's official going.
        hgr=pd.to_numeric(pd.Series([r.get("going_top3_rate")]),errors="coerce").iloc[0]
        ggr=pd.to_numeric(pd.Series([r.get("going_global_top3_rate")]),errors="coerce").iloc[0]
        starts=pd.to_numeric(pd.Series([r.get("going_starts")]),errors="coerce").iloc[0]
        if pd.notna(hgr) and pd.notna(ggr):
            n=max(float(starts) if pd.notna(starts) else 0.0,0.0)
            shrink=(n*hgr+5.0*ggr)/(n+5.0)
            fit=shrink-ggr
            amp=1.0
            if surface=="芝" and pd.notna(cushion) and cushion<8.0:
                amp*=1.12
            if pd.notna(precip) and precip>=0.5:
                amp*=1.08
            elif pd.notna(precip_prob) and precip_prob>=60:
                amp*=1.04
            a=float(np.clip(fit*15.0*amp,-2.4,2.4))
            if abs(a)>=0.15:
                adj += a
                rs.append(f"{official_going or '当日馬場'}適性 {a:+.1f}")

        # 2) Same-day inside/outside bias.
        no=pd.to_numeric(pd.Series([r.get("horse_no")]),errors="coerce").iloc[0]
        if pd.notna(no) and abs(inner)>0.02:
            pos=(float(no)-1.0)/max(field_size-1,1)
            affinity=1.0-2.0*pos  # inner +1, outer -1
            a=float(np.clip(affinity*inner*2.3,-1.8,1.8))
            if abs(a)>=0.12:
                adj += a
                rs.append(f"内外傾向 {a:+.1f}")

        # 3) Same-day running-style bias.
        style=str(r.get("running_style","")).strip() or str(r.get("usual_running_style","")).strip()
        style_aff={"逃げ":1.0,"先行":0.55,"差し":-0.45,"追込":-1.0,"追い込み":-1.0}.get(style,0.0)
        if style_aff and abs(front)>0.02:
            a=float(np.clip(style_aff*front*2.3,-1.8,1.8))
            if abs(a)>=0.12:
                adj += a
                rs.append(f"脚質傾向 {a:+.1f}")

        adj=float(np.clip(adj,-5.0,5.0))
        adjs.append(adj)
        reasons.append("／".join(rs) if rs else "補正なし")

    d["当日補正"]=adjs
    d["補正理由"]=reasons

    # A modest transparent post-model adjustment. It does not pretend that
    # cushion/moisture/bias were part of the original 2016-2025 training set.
    base_win=pd.to_numeric(d["基礎勝率"],errors="coerce").fillna(0).clip(lower=1e-9)
    mult=np.exp(d["当日補正"]/18.0)
    adj_win=base_win*mult
    total=float(adj_win.sum())
    d["win_prob"]=adj_win/total if total>0 else base_win

    base_top3=pd.to_numeric(d["top3_prob"],errors="coerce").fillna(0)
    d["top3_prob"]=np.clip(base_top3*np.exp(d["当日補正"]/30.0),0.001,0.995)
    d["ai_index"]=np.clip(d["基礎AI指数"]+d["当日補正"],0,100)

    odds=pd.to_numeric(d.get("odds"),errors="coerce")
    implied=pd.to_numeric(d.get("implied_prob"),errors="coerce").fillna(0)
    d["expected_value"]=d["win_prob"]*odds
    d["value_gap"]=d["win_prob"]-implied

    d=d.sort_values(["ai_index","win_prob"],ascending=False).reset_index(drop=True)
    d["順位"]=range(1,len(d)+1)
    d["印"]=[mark_for_rank(i,len(d)) for i in d["順位"]]
    d["評価"]=d["印"].map(mark_label)
    return d

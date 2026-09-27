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
    _session, _decode_html, _jra_signature_for_race_id, all_known_race_ids,
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

def _decode_jra_response(response):
    """
    JRA pages may arrive without a reliable charset in the HTTP header.
    requests.Response.text can therefore mojibake Japanese labels such as
    天候 / 芝 / 稍重. Decode raw bytes with several candidates and select the
    version that best preserves known JRA Japanese terms.
    """
    raw=getattr(response,"content",b"") or b""
    text_hint=getattr(response,"text","") or ""

    candidates=[]
    seen=set()
    encs=[
        getattr(response,"encoding",None),
        getattr(response,"apparent_encoding",None),
        "cp932","shift_jis","utf-8","utf-8-sig",
    ]
    for enc in encs:
        if not enc:
            continue
        key=str(enc).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            if raw:
                candidates.append((key,raw.decode(enc,errors="replace")))
        except Exception:
            pass
    if text_hint:
        candidates.append(("response.text",text_hint))
    if not candidates:
        return ""

    keywords={
        "出馬表":8,"天候":8,"競馬場":6,"発走時刻":6,
        "中山":5,"阪神":5,"芝":3,"ダート":3,
        "稍重":7,"不良":5,"良":2,"重":2,
        "馬場状態":7,"含水率":5,"クッション":5,
    }
    def score(txt):
        s=sum(txt.count(k)*w for k,w in keywords.items())
        s-=txt.count("�")*4
        s-=txt.count("譁")*2
        s-=txt.count("陦")*2
        return s
    return max(candidates,key=lambda x:score(x[1]))[1]

def _get_jra_html(session,url,timeout=20):
    r=session.get(url,timeout=timeout)
    r.raise_for_status()
    return _decode_jra_response(r),r

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


def _parse_going_token(text, surface):
    # Accept both "芝稍重" and "芝：稍重", plus known mojibake fallback.
    m=re.search(
        re.escape(surface)+r"\s*[:：]?\s*(不良|稍重|良|重|�.?重)",
        str(text)
    )
    if not m:
        return ""
    v=m.group(1)
    return "稍重" if "�" in v else v

def parse_jra_entry_condition_html(html: str, source_url="") -> dict:
    """Parse live weather/going from a JRA official race-card page."""
    soup=BeautifulSoup(html or "","lxml")
    text=_clean(soup.get_text(" ",strip=True))
    # Only inspect the race header. Previous-race form later on the page also
    # contains many going strings and must not be mistaken for today's going.
    cut=len(text)
    for token in ("本賞金", "枠 馬番", "枠 馬 番"):
        p=text.find(token)
        if p>=0:
            cut=min(cut,p)
    head=text[:min(cut,3500)]

    dm=re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日",head)
    race_date=""
    if dm:
        race_date=f"{int(dm.group(1)):04d}-{int(dm.group(2)):02d}-{int(dm.group(3)):02d}"
    cm=re.search(r"\d+回\s*([^\s\d]+?)\s*\d+日",head)
    course=cm.group(1) if cm else ""
    rm=re.search(r"(\d+)レース",head)
    race_no=int(rm.group(1)) if rm else None
    tm=re.search(r"発走時刻[:：]?\s*(\d{1,2})時(\d{2})分",head)
    post_time=f"{int(tm.group(1)):02d}:{int(tm.group(2)):02d}" if tm else ""
    wm=re.search(r"天候\s*[:：]?\s*(晴|曇|雨|小雨|雪|小雪)",head)
    weather=wm.group(1) if wm else ""
    turf=_parse_going_token(head,"芝")
    dirt=_parse_going_token(head,"ダート")
    return {
        "course":course,"race_date":race_date,"race_no":race_no,
        "post_time":post_time,"jra_weather":weather,
        "turf_going":turf,"dirt_going":dirt,
        "source_url":source_url,
    }

def fetch_jra_live_entry_contexts(target_date: date, courses=None, entry_url_map=None, session=None) -> dict:
    """
    Get live weather/going from JRA official race-card headers.

    Robustness:
    - use any already-discovered official race URLs;
    - also use a known same-day seed directly, so URL collection failure does
      not prevent current going/weather acquisition;
    - decode JRA bytes explicitly (CP932/Shift-JIS/UTF-8 scoring).
    """
    s=session or _session()
    wanted=set(str(x) for x in (courses or []) if str(x))
    urls=dict(entry_url_map or {})
    errors=[]

    if not urls:
        try:
            urls=collect_official_entry_urls(target_date,courses,s)
        except Exception as e:
            errors.append(f"公式URL一覧={e}")
            urls={}

    by_course={}
    for rid,url in urls.items():
        course=VENUE_CODE.get(str(rid)[4:6],"")
        if not course or (wanted and course not in wanted):
            continue
        try:
            rn=int(str(rid)[-2:])
        except Exception:
            rn=0
        by_course.setdefault(course,[]).append((rn,url,"公式URL一覧"))

    # Critical fallback: a single official race card is enough to obtain the
    # course-level live weather/going. Do not depend on collecting all 12 URLs.
    seed_courses=wanted or {
        VENUE_CODE.get(str(rid)[4:6],"") for rid in urls
        if VENUE_CODE.get(str(rid)[4:6],"")
    }
    for course in seed_courses:
        seed=OFFICIAL_ENTRY_SEEDS.get((target_date.isoformat(),course))
        if seed:
            existing={u for _,u,_ in by_course.get(course,[])}
            if seed not in existing:
                by_course.setdefault(course,[]).append((99,seed,"当日公式シード"))

    out={}
    for course in sorted(seed_courses):
        items=by_course.get(course,[])
        merged={
            "course":course,"race_date":"","jra_weather":"",
            "turf_going":"","dirt_going":"","source_urls":[],
            "live_condition_ok":False,"condition_errors":[],
        }
        if not items:
            merged["condition_errors"].append("JRA公式出馬表URLを取得できません")
            out[course]=merged
            continue

        # Prefer direct seed / later race because its header reflects current
        # course conditions and requires only one successful page.
        items=sorted(items,key=lambda x:x[0],reverse=True)
        for _,url,source_kind in items:
            try:
                html,_=_get_jra_html(s,url,timeout=20)
                q=parse_jra_entry_condition_html(html,url)
                if q.get("race_date") and q.get("race_date")!=target_date.isoformat():
                    merged["condition_errors"].append(
                        f"{source_kind}: 日付不一致 {q.get('race_date')}"
                    )
                    continue

                if q.get("race_date"):
                    merged["race_date"]=q["race_date"]
                if q.get("jra_weather"):
                    merged["jra_weather"]=q["jra_weather"]
                if q.get("turf_going"):
                    merged["turf_going"]=q["turf_going"]
                if q.get("dirt_going"):
                    merged["dirt_going"]=q["dirt_going"]
                merged["source_urls"].append(url)

                if (
                    q.get("jra_weather") or
                    q.get("turf_going") or
                    q.get("dirt_going")
                ):
                    merged["live_condition_ok"]=True

                # One current-condition value is already useful; keep trying
                # only if both surfaces/weather are still blank.
                if (
                    merged["jra_weather"] and
                    (merged["turf_going"] or merged["dirt_going"])
                ):
                    break
            except Exception as e:
                merged["condition_errors"].append(
                    f"{source_kind}: {type(e).__name__}: {e}"
                )

        merged["is_target_date"]=(
            merged.get("race_date")==target_date.isoformat()
        )
        out[course]=merged

    # Preserve discovery-level errors in each requested course for UI diagnosis.
    if errors:
        for course in seed_courses:
            out.setdefault(course,{"course":course})
            out[course].setdefault("condition_errors",[]).extend(errors)
    return out
def parse_public_race_condition_html(html: str, race_id: str, source_url="") -> dict:
    """
    Parse the current race header from the already-used public race-card source.

    Typical header:
      15:40発走 / 芝1200m (...) / 天候:曇 / 馬場:稍

    "稍" is normalized to the model's historical label "稍重".
    """
    soup=BeautifulSoup(html or "","lxml")
    rd1=soup.select_one("div.RaceData01")
    rd2=soup.select_one("div.RaceData02")
    bits=[]
    if rd1:
        bits.append(_clean(rd1.get_text(" ",strip=True)))
    if rd2:
        bits.append(_clean(rd2.get_text(" ",strip=True)))
    if not bits:
        bits.append(_clean(soup.get_text(" ",strip=True))[:2500])
    text=" ".join(bits)

    sm=re.search(r"(芝|ダ(?:ート)?)\s*(\d{3,4})m",text)
    raw_surface=sm.group(1) if sm else ""
    surface="ダート" if raw_surface.startswith("ダ") else ("芝" if raw_surface=="芝" else "")

    wm=re.search(r"天候\s*[:：]\s*(晴|曇|雨|小雨|雪|小雪)",text)
    weather=wm.group(1) if wm else ""

    gm=re.search(r"馬場\s*[:：]\s*(不良|稍重|稍|良|重)",text)
    gv=gm.group(1) if gm else ""
    going="稍重" if gv=="稍" else gv
    if going not in ("良","稍重","重","不良"):
        going=""

    return {
        "race_id":race_id,
        "course":VENUE_CODE.get(str(race_id)[4:6],""),
        "race_no":int(str(race_id)[-2:]) if str(race_id)[-2:].isdigit() else None,
        "surface":surface,
        "going":going,
        "jra_weather":weather,
        "source_url":source_url,
    }

def _race_no_int_safe(v, default=None):
    m=re.search(r"\d+",str(v or ""))
    if not m:
        return default
    try:
        return int(m.group(0))
    except Exception:
        return default

def _race_id_for_course_no(target_date,course,race_no):
    wanted=_race_no_int_safe(race_no)
    if wanted is None:
        return ""
    for rid in all_known_race_ids(target_date):
        if (
            VENUE_CODE.get(str(rid)[4:6],"")==str(course)
            and int(str(rid)[-2:])==wanted
        ):
            return rid
    return ""

def fetch_public_live_contexts(
    target_date: date,
    courses,
    race_surface_pairs=None,
    session=None,
) -> dict:
    """
    Fallback used when Streamlit Cloud cannot reach/parse JRA directly.
    Fetch only one representative race per surface/course when possible.
    """
    s=session or _session()
    wanted=list(dict.fromkeys(str(x) for x in (courses or []) if str(x)))
    pairs=list(race_surface_pairs or [])

    chosen={course:{} for course in wanted}
    for item in pairs:
        try:
            course=str(item[0])
            race_no=_race_no_int_safe(item[1])
            surface=str(item[2])
        except Exception:
            continue
        if race_no is None:
            continue
        if course not in chosen or surface not in ("芝","ダート"):
            continue
        # Prefer later races because their page is usually the most current.
        prev=chosen[course].get(surface)
        if prev is None or race_no>prev:
            chosen[course][surface]=race_no

    # If the caller did not provide race/surface pairs, use a few candidate
    # race pages and stop as soon as both surfaces are found.
    known_by_course={course:[] for course in wanted}
    for rid in all_known_race_ids(target_date):
        course=VENUE_CODE.get(str(rid)[4:6],"")
        if course in known_by_course:
            known_by_course[course].append(rid)
    for course in wanted:
        known_by_course[course]=sorted(
            known_by_course[course],
            key=lambda x:int(str(x)[-2:]),
            reverse=True
        )

    out={}
    for course in wanted:
        ctx={
            "course":course,"jra_weather":"",
            "turf_going":"","dirt_going":"",
            "public_source_urls":[],
            "public_condition_ok":False,
            "public_condition_errors":[],
        }
        ids=[]
        for surface,rn in chosen.get(course,{}).items():
            rid=_race_id_for_course_no(target_date,course,rn)
            if rid:
                ids.append(rid)

        # Add candidates only as needed, with a modest request cap.
        for rid in known_by_course.get(course,[]):
            if rid not in ids:
                ids.append(rid)
            if len(ids)>=6:
                break

        for rid in ids:
            if ctx["turf_going"] and ctx["dirt_going"] and ctx["jra_weather"]:
                break
            url=f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}"
            try:
                r=s.get(url,timeout=15)
                r.raise_for_status()
                html=_decode_html(r.content)
                q=parse_public_race_condition_html(html,rid,url)
                if q.get("jra_weather") and not ctx["jra_weather"]:
                    ctx["jra_weather"]=q["jra_weather"]
                if q.get("surface")=="芝" and q.get("going"):
                    ctx["turf_going"]=q["going"]
                elif q.get("surface")=="ダート" and q.get("going"):
                    ctx["dirt_going"]=q["going"]
                if q.get("going") or q.get("jra_weather"):
                    ctx["public_source_urls"].append(url)
                    ctx["public_condition_ok"]=True
            except Exception as e:
                ctx["public_condition_errors"].append(
                    f"{rid}: {type(e).__name__}: {e}"
                )
        out[course]=ctx
    return out

def merge_entry_conditions(contexts: dict, entries: pd.DataFrame) -> dict:
    """Fallback: expose the going already present in acquired race-card rows."""
    out={k:dict(v) for k,v in (contexts or {}).items()}
    if entries is None or len(entries)==0:
        return out
    for course,g in entries.groupby(entries["course"].astype(str)):
        ctx=out.setdefault(str(course),{"course":str(course)})
        for surface,key in (("芝","turf_going"),("ダート","dirt_going")):
            x=g[g["surface"].astype(str)==surface]
            vals=x.get("going",pd.Series(dtype=object)).fillna("").astype(str)
            vals=vals[vals.isin(["良","稍重","重","不良"])]
            if len(vals):
                # Acquired race-card rows are same-day data. Keep an explicitly
                # parsed JRA live header if present; otherwise they override the
                # weekly baba-page value, which can still say Friday/noon.
                live_source=str(ctx.get(key+"_source", ""))
                if "JRA公式出馬表（当日）" not in live_source:
                    ctx[key]=vals.mode().iloc[0]
                    ctx[key+"_source"]="取得出走表（当日）"
        # If same-day acquired race rows contain a valid going, the app has
        # a current condition even when JRA direct access failed.
        if (
            ctx.get("turf_going") in ("良","稍重","重","不良") or
            ctx.get("dirt_going") in ("良","稍重","重","不良")
        ):
            ctx["current_condition_ok"]=True
        ctx.setdefault("is_target_date",True)
    return out

def fetch_jra_baba_contexts(target_date: date, courses=None, session=None) -> dict:
    s = session or _session()
    wanted = set(courses or [])
    out = {}
    for url in JRA_BABA_URLS:
        try:
            html,_=_get_jra_html(s,url,timeout=15)
            ctx=parse_jra_baba_html(html,url)
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

def fetch_day_contexts(
    target_date: date,
    courses,
    entry_url_map=None,
    race_surface_pairs=None,
) -> dict:
    courses=list(dict.fromkeys([str(x) for x in courses if str(x)]))
    s=_session()

    # 1) JRA official track-information page.
    jra=fetch_jra_baba_contexts(target_date,courses,s)

    # 2) JRA official same-day race-card header.
    live=fetch_jra_live_entry_contexts(
        target_date,courses,entry_url_map,s
    )

    # 3) Public current race-card fallback. This is especially important on
    # Streamlit Cloud when JRA direct requests are rejected upstream.
    public=fetch_public_live_contexts(
        target_date,courses,race_surface_pairs,s
    )

    out={}
    for course in courses:
        ctx=dict(jra.get(course,{"course":course}))
        ctx["course"]=course
        lv=live.get(course) or {}
        pv=public.get(course) or {}

        # JRA same-day source is always preferred.
        if lv.get("turf_going"):
            ctx["turf_going"]=lv["turf_going"]
            ctx["turf_going_source"]="JRA公式出馬表（当日）"
        elif pv.get("turf_going"):
            ctx["turf_going"]=pv["turf_going"]
            ctx["turf_going_source"]="公開出馬表（当日）"

        if lv.get("dirt_going"):
            ctx["dirt_going"]=lv["dirt_going"]
            ctx["dirt_going_source"]="JRA公式出馬表（当日）"
        elif pv.get("dirt_going"):
            ctx["dirt_going"]=pv["dirt_going"]
            ctx["dirt_going_source"]="公開出馬表（当日）"

        if lv.get("jra_weather"):
            ctx["jra_weather"]=lv["jra_weather"]
            ctx["weather_source"]="JRA公式出馬表（当日）"
        elif pv.get("jra_weather"):
            ctx["jra_weather"]=pv["jra_weather"]
            ctx["weather_source"]="公開出馬表（当日）"

        if lv.get("race_date"):
            ctx["live_race_date"]=lv["race_date"]
        if lv.get("source_urls"):
            ctx["live_source_urls"]=lv["source_urls"]
        if pv.get("public_source_urls"):
            ctx["public_source_urls"]=pv["public_source_urls"]

        ctx["live_condition_ok"]=bool(lv.get("live_condition_ok"))
        ctx["public_condition_ok"]=bool(pv.get("public_condition_ok"))
        ctx["current_condition_ok"]=bool(
            ctx["live_condition_ok"] or
            ctx["public_condition_ok"] or
            ctx.get("turf_going") in ("良","稍重","重","不良") or
            ctx.get("dirt_going") in ("良","稍重","重","不良")
        )
        ctx["condition_errors"]=(
            list(lv.get("condition_errors") or []) +
            list(pv.get("public_condition_errors") or [])
        )

        # Public fallback is tied to the requested target date/race IDs.
        ctx["is_target_date"]=bool(
            lv.get("is_target_date") or
            pv.get("public_condition_ok") or
            ctx.get("race_date")==target_date.isoformat()
        )

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
            out.at[i,"going_source"]=ctx.get(
                "turf_going_source" if surface=="芝" else "dirt_going_source",
                "JRA馬場情報"
            )
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
            html,_=_get_jra_html(s,seed,timeout=20)
            soup=BeautifulSoup(html,"lxml")
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
        html,_=_get_jra_html(session,entry_url,timeout=20)
        soup=BeautifulSoup(html,"lxml")

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
    c_jockey=_find_col(cols,["騎手名","騎手"])
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
        jockey=_clean(r.get(c_jockey,"")) if c_jockey else ""
        jockey=re.sub(r"^[▲△☆★]", "", jockey)
        jockey=re.sub(r"\s+", "", jockey)
        rows.append({
            "finish":finish,"horse_no":horse_no,
            "first_pos":passage[0] if passage else np.nan,
            "jockey":jockey,
        })
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
    current_no=_race_no_int_safe(current_race_no)
    if current_no is None:
        return {
            "sample_races":0,"sample_horses":0,
            "inside_top3_rate":np.nan,"outside_top3_rate":np.nan,
            "front_top3_rate":np.nan,"closer_top3_rate":np.nan,
            "jockey_day":{},
            "summary":"レース番号を解析できず、当日バイアスを計算できません",
            "source_urls":[],
        }
    prior=[]
    for rid,url in (entry_url_map or {}).items():
        if VENUE_CODE.get(rid[4:6],"") != course:
            continue
        rn=int(rid[-2:])
        if rn < current_no:
            prior.append((rn,rid,url))
    prior.sort()

    top3_norm_no=[]
    top3_norm_pos=[]
    used=0
    sources=[]
    jockey_acc={}
    for rn,rid,entry_url in prior:
        result_url=_result_url_from_entry(entry_url,s)
        if not result_url:
            continue
        try:
            rr=s.get(result_url,timeout=15)
            rr.raise_for_status()
            parsed=parse_jra_result_html(rr.text)
            df=parsed.get("rows")
            if df is None or len(df)<3:
                continue

            # Same-day jockey form uses all prior races at this course, not only
            # the current surface. It is a small transparent adjustment.
            for _,jr in df.iterrows():
                j=str(jr.get("jockey","")).strip()
                if not j:
                    continue
                a=jockey_acc.setdefault(j,{"rides":0,"wins":0,"top3":0})
                a["rides"]+=1
                fi=int(jr.get("finish",99))
                if fi==1: a["wins"]+=1
                if fi<=3: a["top3"]+=1

            if parsed.get("surface") != surface:
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

    jockey_stats={}
    for j,a in jockey_acc.items():
        rides=max(int(a.get("rides",0)),1)
        jockey_stats[j]={
            **a,
            "win_rate":float(a.get("wins",0))/rides,
            "top3_rate":float(a.get("top3",0))/rides,
        }
    return {
        "races_used":used,
        "inner_score":inner_score,
        "front_score":front_score,
        "summary":summary,
        "source_urls":sources,
        "jockey_stats":jockey_stats,
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
    d["騎手補正"]=0.0
    d["騎手評価理由"]=""

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
    jockey_adjs=[]
    jockey_reasons=[]
    day_jockey=(bias or {}).get("jockey_stats") or {}
    for _,r in d.iterrows():
        adj=0.0
        rs=[]
        jockey_adj=0.0
        jrs=[]

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

        # 4) Jockey condition fit. Overall jockey_top3_rate is already included
        # in the learned model, so only course/surface/distance/going/trainer
        # interaction deltas are added here to avoid double counting.
        overall=pd.to_numeric(pd.Series([r.get("jockey_top3_rate")]),errors="coerce").iloc[0]
        pieces=[]
        for label,rc,sc,w in [
            ("競馬場","jockey_course_top3_rate","jockey_course_starts",1.0),
            ("芝ダ","jockey_surface_top3_rate","jockey_surface_starts",0.8),
            ("距離","jockey_distance_top3_rate","jockey_distance_starts",0.9),
            ("馬場","jockey_going_top3_rate","jockey_going_starts",0.8),
            ("厩舎","jockey_trainer_top3_rate","jockey_trainer_starts",0.7),
        ]:
            rate=pd.to_numeric(pd.Series([r.get(rc)]),errors="coerce").iloc[0]
            n=pd.to_numeric(pd.Series([r.get(sc)]),errors="coerce").iloc[0]
            if pd.notna(overall) and pd.notna(rate) and pd.notna(n) and n>=2:
                shrunk=(float(n)*float(rate)+10.0*float(overall))/(float(n)+10.0)
                pieces.append((w,shrunk-float(overall),label,int(n)))
        if pieces:
            delta=sum(w*x for w,x,_,_ in pieces)/sum(w for w,_,_,_ in pieces)
            a=float(np.clip(delta*10.0,-1.25,1.25))
            if abs(a)>=0.08:
                jockey_adj+=a
                labels="・".join(f"{lab}{n}走" for _,_,lab,n in pieces[:3])
                jrs.append(f"条件別{a:+.1f}({labels})")

        # 5) Jockey's already-finished rides today at the same course.
        jockey=re.sub(r"\s+","",str(r.get("jockey","")).strip())
        js=day_jockey.get(jockey) or {}
        rides=int(js.get("rides",0) or 0)
        if rides>0 and pd.notna(overall):
            top3=int(js.get("top3",0) or 0)
            day_rate=(top3+2.0*float(overall))/(rides+2.0)
            a=float(np.clip((day_rate-float(overall))*2.2*min(1.0,rides/3.0),-0.75,0.75))
            if abs(a)>=0.08:
                jockey_adj+=a
                jrs.append(f"当日{top3}/{rides} {a:+.1f}")

        jockey_adj=float(np.clip(jockey_adj,-1.7,1.7))
        adj += jockey_adj
        if abs(jockey_adj)>=0.08:
            rs.append(f"騎手 {jockey_adj:+.1f}")

        adj=float(np.clip(adj,-5.0,5.0))
        adjs.append(adj)
        reasons.append("／".join(rs) if rs else "補正なし")
        jockey_adjs.append(jockey_adj)
        jockey_reasons.append("／".join(jrs) if jrs else "基礎AIに騎手総合成績を反映済み")

    d["当日補正"]=adjs
    d["補正理由"]=reasons
    d["騎手補正"]=jockey_adjs
    d["騎手評価理由"]=jockey_reasons

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

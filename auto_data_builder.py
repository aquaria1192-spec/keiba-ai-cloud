from __future__ import annotations

from pathlib import Path
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from io import StringIO
import gzip
import json
import re
import shutil
import time
import urllib.parse
import subprocess
import sys

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from feature_engineering import validate_raw_history, validate_entries

JST = ZoneInfo("Asia/Tokyo")
VENUE_CODE = {
    "01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京",
    "06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉",
}
ENTRY_COLS = [
    "date","course","race_no","race_name","surface","distance","going",
    "horse_no","gate","horse_name","jockey","trainer","age","sex","weight",
    "body_weight","body_weight_diff","odds","popularity","race_class","running_style"
]
KNOWN_RACE_DAYS = {
    # 2026-09-26: 4回中山8日 / 4回阪神8日
    "2026-09-26": (
        [f"2026060408{r:02d}" for r in range(1,13)] +
        [f"2026090408{r:02d}" for r in range(1,13)]
    ),
    # 2026-09-27: 4回中山9日 / 4回阪神9日
    "2026-09-27": (
        [f"2026060409{r:02d}" for r in range(1,13)] +
        [f"2026090409{r:02d}" for r in range(1,13)]
    )
}

JRA_OFFICIAL_SEED_URLS = {
    ("2026-09-27", "阪神"):
        "https://app.jra.jp/JRADB/accessD.html?CNAME=sw01dde0109202604090420260927%2F92",
}

# Flat-race AI only. The large training CSV excludes jump races.
UNSUPPORTED_RACES = {
    "2026-09-27": {
        "202609040901": "障害3歳以上未勝利（障害2970m）",
    }
}

class AutoDataError(RuntimeError):
    pass

def default_target_date() -> date:
    """Choose the next likely JRA race day in Japan."""
    now = datetime.now(JST)
    wd = now.weekday()  # Mon=0
    if wd == 5:  # Saturday
        return now.date() if now.hour < 16 else now.date() + timedelta(days=1)
    if wd == 6:  # Sunday
        return now.date() if now.hour < 16 else now.date() + timedelta(days=6)
    # Next Saturday.
    return now.date() + timedelta(days=(5-wd) % 7)

def _progress(cb, v, msg):
    if cb:
        cb(float(v), str(msg))

def ensure_training_csv(seed_dir: Path, generated_dir: Path, force=False, progress_callback=None) -> Path:
    """
    Create the large training CSV automatically from the bundled gzip seed.
    This makes the user-facing workflow independent of manual CSV selection.
    """
    seed_dir = Path(seed_dir)
    generated_dir = Path(generated_dir)
    generated_dir.mkdir(parents=True, exist_ok=True)

    out = generated_dir/"training_history_recommended_2016_2025.csv"
    src = seed_dir/"training_history_recommended_2016_2025.csv.gz"

    if out.exists() and out.stat().st_size > 1_000_000 and not force:
        _progress(progress_callback, .18, "学習CSVは作成済みです。")
        return out
    if not src.exists():
        raise AutoDataError("学習データの圧縮ファイルがありません。アプリ一式を再展開してください。")

    _progress(progress_callback, .08, "大規模学習CSVを自動作成しています…")
    tmp = out.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    with gzip.open(src, "rb") as f_in, tmp.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    tmp.replace(out)

    # Lightweight schema check only; avoid reading 90MB twice.
    head = pd.read_csv(out, nrows=20)
    missing = validate_raw_history(head)
    if missing:
        out.unlink(missing_ok=True)
        raise AutoDataError("自動作成した学習CSVに不足列があります: " + ", ".join(missing))
    _progress(progress_callback, .23, "学習CSVを作成しました。")
    return out

def _session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/153.0 Safari/537.36",
        "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
        "Referer": "https://race.netkeiba.com/",
    })
    return s


def _normalize_going_token(v):
    x=str(v or "").strip()
    if x in ("稍","稍重"):
        return "稍重"
    if x in ("良","重","不良"):
        return x
    # Known JRA mojibake for 稍重 can appear as "�c重" etc.
    if "�" in x and x.endswith("重"):
        return "稍重"
    return ""

def _decode_html(content: bytes) -> str:
    for enc in ("euc_jp", "cp932", "utf-8"):
        try:
            text = content.decode(enc)
            if "race_id" in text or "RaceName" in text or "出馬" in text:
                return text
        except Exception:
            pass
    return content.decode("utf-8", errors="replace")

def discover_race_ids(target_date: date, session=None, timeout=15) -> list[str]:
    """Discover all JRA race IDs on a date from the public race-list page."""
    s = session or _session()
    ymd = target_date.strftime("%Y%m%d")
    url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={ymd}"
    try:
        r = s.get(url, timeout=timeout)
        r.raise_for_status()
        html = _decode_html(r.content)
        ids = sorted(set(re.findall(r"race_id=(\d{12})", html)))
        # JRA only: venue code 01-10, target year, race no 01-12
        good = []
        for rid in ids:
            if rid[:4] != target_date.strftime("%Y"):
                continue
            if rid[4:6] not in VENUE_CODE:
                continue
            try:
                rn = int(rid[-2:])
            except Exception:
                continue
            if 1 <= rn <= 12:
                good.append(rid)
        if good:
            return good
    except Exception:
        pass
    return list(KNOWN_RACE_DAYS.get(target_date.isoformat(), []))

def _flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        cols = []
        for tup in df.columns:
            vals = [str(x).strip() for x in tup if str(x).strip() and not str(x).startswith("Unnamed")]
            cols.append(" ".join(dict.fromkeys(vals)))
        df = df.copy()
        df.columns = cols
    else:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
    return df

def _find_col(cols, keywords):
    for k in keywords:
        for c in cols:
            if k in str(c):
                return c
    return None

def _num(v):
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(v))
    return float(m.group()) if m else np.nan

def _sex_age(v):
    s = str(v)
    m = re.search(r"(牡|牝|セ|せん|セン|騙)\s*(\d+)", s)
    if not m:
        return "", np.nan
    sex = {"セ":"せん","セン":"せん","騙":"せん"}.get(m.group(1),m.group(1))
    return sex, int(m.group(2))

def _body(v):
    s = str(v)
    m = re.search(r"(\d{3,4})\s*(?:kg)?\s*\(([+-]?\d+)\)", s, re.I)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"(\d{3,4})\s*(?:kg)?", s, re.I)
    if m:
        return float(m.group(1)), np.nan
    return np.nan, np.nan

def _race_class(text):
    t = str(text)
    if re.search(r"\bG\s*[ⅠI1]\b|GI(?!I)", t, re.I): return "G1"
    if re.search(r"\bG\s*[ⅡI2]\b|GII(?!I)", t, re.I): return "G2"
    if re.search(r"\bG\s*[ⅢI3]\b|GIII", t, re.I): return "G3"
    if "リステッド" in t or re.search(r"\bL\b", t): return "L"
    if "3勝" in t: return "3勝クラス"
    if "2勝" in t: return "2勝クラス"
    if "1勝" in t: return "1勝クラス"
    if "オープン" in t: return "オープン"
    if "未勝利" in t: return "未勝利"
    if "新馬" in t: return "新馬"
    return ""

def _odds_for_race(race_id, session, timeout=12):
    url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=update"
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        odds = (((js or {}).get("data") or {}).get("odds") or {}).get("1") or {}
        result = {}
        for k, vals in odds.items():
            try:
                horse_no = int(k)
            except Exception:
                continue
            if not isinstance(vals, (list,tuple)) or not vals:
                continue
            ov = pd.to_numeric(pd.Series([vals[0]]), errors="coerce").iloc[0]
            pop = pd.to_numeric(pd.Series([vals[2] if len(vals) > 2 else np.nan]), errors="coerce").iloc[0]
            result[horse_no] = {
                "odds": float(ov) if pd.notna(ov) else np.nan,
                "popularity": int(pop) if pd.notna(pop) else np.nan,
            }
        return result
    except Exception:
        return {}

def parse_shutuba_html(html: str, race_id: str, target_date: date, odds_map=None) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    name_el = soup.select_one("h1.RaceName")
    race_name = name_el.get_text(" ", strip=True) if name_el else f"{int(race_id[-2:])}R"

    rd1 = soup.select_one("div.RaceData01")
    rd2 = soup.select_one("div.RaceData02")
    txt1 = rd1.get_text(" ", strip=True) if rd1 else ""
    txt2 = rd2.get_text(" ", strip=True) if rd2 else ""
    all_meta = " ".join([race_name, txt1, txt2, soup.title.get_text(" ",strip=True) if soup.title else ""])

    sm = re.search(r"(障(?:芝|ダ)?|芝|ダ)\s*(\d{3,4})m", all_meta)
    if not sm:
        raise AutoDataError(f"{race_id}: 芝/ダート・距離を取得できません。")
    raw_surface = sm.group(1)
    if raw_surface.startswith("障"):
        return pd.DataFrame(columns=ENTRY_COLS)  # skip jump races
    surface = "ダート" if raw_surface == "ダ" else "芝"
    distance = int(sm.group(2))

    gm = re.search(r"馬場\s*[:：]\s*(不良|稍重|稍|良|重)", all_meta)
    going = _normalize_going_token(gm.group(1) if gm else "")

    tables = pd.read_html(StringIO(html))
    table = None
    for t in tables:
        tt = _flatten_columns(t)
        joined = " ".join(map(str, tt.columns))
        if "馬名" in joined and "騎手" in joined and ("馬番" in joined or "馬 番" in joined):
            table = tt
            break
    if table is None:
        raise AutoDataError(f"{race_id}: 出走馬テーブルを取得できません。")

    cols = list(table.columns)
    c_gate = _find_col(cols, ["枠"])
    c_no = _find_col(cols, ["馬番","馬 番"])
    c_name = _find_col(cols, ["馬名"])
    c_sexage = _find_col(cols, ["性齢","性 齢"])
    c_weight = _find_col(cols, ["斤量","負担重量"])
    c_jockey = _find_col(cols, ["騎手"])
    c_trainer = _find_col(cols, ["厩舎","調教師"])
    c_body = _find_col(cols, ["馬体重"])

    if not all([c_no,c_name,c_jockey]):
        raise AutoDataError(f"{race_id}: 必須列（馬番・馬名・騎手）を特定できません。")

    venue = VENUE_CODE.get(race_id[4:6], "")
    race_no = int(race_id[-2:])
    odds_map = odds_map or {}
    rows = []

    for _, rr in table.iterrows():
        hno = _num(rr.get(c_no,""))
        if pd.isna(hno):
            continue
        hno = int(hno)
        gate = _num(rr.get(c_gate,"")) if c_gate else np.nan
        horse = str(rr.get(c_name,"")).strip()
        horse = re.sub(r"\s*(お気に入り馬|Image|編集).*$","",horse).strip()
        if not horse or horse.lower()=="nan":
            continue
        sex, age = _sex_age(rr.get(c_sexage,"")) if c_sexage else ("",np.nan)
        carried = _num(rr.get(c_weight,"")) if c_weight else np.nan
        jockey = str(rr.get(c_jockey,"")).strip()
        jockey = re.sub(r"\s+","",jockey)
        trainer = str(rr.get(c_trainer,"")).strip() if c_trainer else ""
        trainer = re.sub(r"^(美浦|栗東)","",trainer)
        trainer = re.sub(r"\s+","",trainer)
        bw, bwd = _body(rr.get(c_body,"")) if c_body else (np.nan,np.nan)
        od = odds_map.get(hno,{})
        rows.append({
            "date":target_date.isoformat(),
            "course":venue,
            "race_no":f"{race_no}R",
            "race_name":race_name,
            "surface":surface,
            "distance":distance,
            "going":going,
            "horse_no":hno,
            "gate":int(gate) if pd.notna(gate) else np.nan,
            "horse_name":horse,
            "jockey":jockey,
            "trainer":trainer,
            "age":age,
            "sex":sex,
            "weight":carried,
            "body_weight":bw,
            "body_weight_diff":bwd,
            "odds":od.get("odds",np.nan),
            "popularity":od.get("popularity",np.nan),
            "race_class":_race_class(all_meta),
            "running_style":"",
        })
    return pd.DataFrame(rows, columns=ENTRY_COLS)

def _daily_name_parts(v):
    """
    Extract horse/trainer/sex/age from a Daily-style cell.
    Works whether HTML table parsing preserves line breaks or converts them
    to spaces, e.g. "デンプシー 上原佑牡3鹿毛".
    """
    s = str(v).replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if not s or s.lower() == "nan":
        return "", "", np.nan, ""

    m = re.match(r"^(\S+)\s+(.+?)(牡|牝|騸|せん|セン)(\d+)", s)
    if m:
        horse = m.group(1).strip()
        trainer = m.group(2).strip()
        sex = {"騸":"せん","セン":"せん"}.get(m.group(3), m.group(3))
        age = int(m.group(4))
        return horse, trainer, age, sex

    # Fallback: horse name is the first token.
    return s.split()[0], "", np.nan, ""

def parse_daily_shutuba_html(html: str, race_id: str, target_date: date) -> pd.DataFrame:
    """
    Secondary provider. Missing values are allowed; import should continue
    rather than fail just because odds/body weight are unavailable.
    """
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)

    venue = VENUE_CODE.get(race_id[4:6], "")
    race_no = int(race_id[-2:])

    title = soup.find("h1")
    race_name = title.get_text(" ", strip=True) if title else f"{race_no}R"
    race_name = re.sub(r"^サラ\S*\s*", "", race_name).strip() or f"{race_no}R"

    meta_match = re.search(
        r"\d{4}/\d{1,2}/\d{1,2}.*?(芝|ダート).*?(\d{3,4})m",
        text
    )
    if not meta_match:
        raise AutoDataError(f"{race_id}: 予備取得元から芝/ダート・距離を取得できません。")
    surface = meta_match.group(1)
    distance = int(meta_match.group(2))

    gm = re.search(r"(?:馬場|馬場状態)\s*[:：]?\s*(不良|稍重|稍|良|重)", text)
    going = _normalize_going_token(gm.group(1) if gm else "")

    tables = pd.read_html(StringIO(html))
    table = None
    for t in tables:
        tt = _flatten_columns(t)
        joined = " ".join(map(str, tt.columns))
        if "馬番" in joined and "馬名" in joined:
            table = tt
            break
    if table is None:
        raise AutoDataError(f"{race_id}: 予備取得元の出走馬テーブルを特定できません。")

    cols = list(table.columns)
    c_gate = _find_col(cols, ["枠番","枠"])
    c_no = _find_col(cols, ["馬番"])
    c_name = _find_col(cols, ["馬名"])
    c_jockey = _find_col(cols, ["騎手"])
    c_weight = _find_col(cols, ["負担量","斤量"])
    c_body = _find_col(cols, ["馬体重"])

    if not c_no or not c_name:
        raise AutoDataError(f"{race_id}: 予備取得元で馬番・馬名を特定できません。")

    rows = []
    for _, rr in table.iterrows():
        hno = _num(rr.get(c_no, ""))
        if pd.isna(hno):
            continue
        hno = int(hno)
        if not (1 <= hno <= 18):
            continue

        gate = _num(rr.get(c_gate, "")) if c_gate else np.nan
        horse, trainer, age, sex = _daily_name_parts(rr.get(c_name, ""))
        if not horse or horse.lower() == "nan":
            continue

        carried = np.nan
        if c_weight:
            carried = _num(rr.get(c_weight, ""))

        jockey = ""
        if c_jockey:
            js = str(rr.get(c_jockey, "")).replace("\u3000", " ")
            js = re.sub(r"\s+", "", js)
            # In some layouts jockey cell can contain the assigned weight.
            wm = re.match(r"^[▲△☆◇]?(\d{2}\.\d)(.*)$", js)
            if wm:
                if pd.isna(carried):
                    carried = _num(wm.group(1))
                jockey = wm.group(2)
            else:
                jockey = js

        bw, bwd = (np.nan, np.nan)
        if c_body:
            bw, bwd = _body(rr.get(c_body, ""))

        rows.append({
            "date": target_date.isoformat(),
            "course": venue,
            "race_no": f"{race_no}R",
            "race_name": race_name,
            "surface": surface,
            "distance": distance,
            "going": going,
            "horse_no": hno,
            "gate": int(gate) if pd.notna(gate) else np.nan,
            "horse_name": horse,
            "jockey": jockey,
            "trainer": trainer,
            "age": age,
            "sex": sex,
            "weight": carried,
            "body_weight": bw,
            "body_weight_diff": bwd,
            "odds": np.nan,
            "popularity": np.nan,
            "race_class": _race_class(text),
            "running_style": "",
        })

    out = pd.DataFrame(rows, columns=ENTRY_COLS)
    if out.empty:
        raise AutoDataError(f"{race_id}: 予備取得元から出走馬を取得できません。")
    return out

def fetch_race_entries_daily(race_id: str, target_date: date, session=None, timeout=15):
    s = session or _session()
    old_ref = s.headers.get("Referer")
    s.headers["Referer"] = "https://www.daily.co.jp/umaya/"
    try:
        url = f"https://www.daily.co.jp/umaya/race/shutsuba.shtml?race_id={race_id}"
        r = s.get(url, timeout=timeout)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return parse_daily_shutuba_html(r.text, race_id, target_date)
    finally:
        if old_ref:
            s.headers["Referer"] = old_ref

def fetch_race_entries(race_id: str, target_date: date, session=None, timeout=15):
    """
    Two-step retrieval:
      1) primary public race-card source
      2) secondary public race-card source
    """
    s = session or _session()
    primary_error = None

    try:
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        r = s.get(url, timeout=timeout)
        r.raise_for_status()
        html = _decode_html(r.content)
        odds = _odds_for_race(race_id, s, timeout=min(timeout,12))
        out = parse_shutuba_html(html, race_id, target_date, odds)
        if out is not None and len(out):
            return out
    except Exception as e:
        primary_error = e

    try:
        return fetch_race_entries_daily(race_id, target_date, s, timeout=timeout)
    except Exception as secondary_error:
        raise AutoDataError(
            f"{race_id}: 通常取得失敗={primary_error} / 予備取得失敗={secondary_error}"
        )


def all_known_race_ids(target_date: date) -> list[str]:
    return list(KNOWN_RACE_DAYS.get(target_date.isoformat(), []))

def unsupported_race_ids(target_date: date) -> dict[str,str]:
    return dict(UNSUPPORTED_RACES.get(target_date.isoformat(), {}))

def expected_race_ids(target_date: date) -> list[str]:
    unsupported = set(unsupported_race_ids(target_date))
    return [rid for rid in all_known_race_ids(target_date) if rid not in unsupported]

def _race_key_from_id(rid: str) -> tuple[str,int]:
    return VENUE_CODE.get(rid[4:6], ""), int(rid[-2:])

def _have_race_ids(frames: list[pd.DataFrame], target_date: date) -> set[str]:
    """
    Convert successfully parsed frames back to known race IDs by course/race_no
    for the current known-day mapping.
    """
    known = expected_race_ids(target_date)
    lookup = {}
    for rid in known:
        lookup[_race_key_from_id(rid)] = rid

    found = set()
    for f in frames:
        if f is None or len(f) == 0:
            continue
        row = f.iloc[0]
        course = str(row.get("course",""))
        rn = str(row.get("race_no","")).upper().replace("R","")
        try:
            rn = int(rn)
        except Exception:
            continue
        rid = lookup.get((course,rn))
        if rid:
            found.add(rid)
    return found


def _clean_jra_text(v):
    s = str(v).replace("\u3000", " ")
    return re.sub(r"\s+", " ", s).strip()

def _jra_horse_detail(v):
    s = _clean_jra_text(v)
    horse = ""
    odds = np.nan
    pop = np.nan
    sex = ""
    age = np.nan
    bw = np.nan
    bwd = np.nan
    jockey = ""
    carried = np.nan
    trainer = ""

    stop_positions = []
    for pat in [r"\d+(?:\.\d+)?\s*\(\d+番人気\)", r"(?:牡|牝|せん|セン|騸)\d+"]:
        m = re.search(pat, s)
        if m:
            stop_positions.append(m.start())
    horse = s[:min(stop_positions)].strip() if stop_positions else (s.split(" ")[0] if s else "")
    horse = re.sub(r"^(?:Image:?|ブリンカー着用)\s*", "", horse).strip()

    om = re.search(r"(\d+(?:\.\d+)?)\s*\((\d+)番人気\)", s)
    if om:
        odds = float(om.group(1))
        pop = int(om.group(2))

    sm = re.search(r"(牡|牝|せん|セン|騸)\s*(\d+)", s)
    if sm:
        sex = {"セン":"せん","騸":"せん"}.get(sm.group(1), sm.group(1))
        age = int(sm.group(2))

    bm = re.search(r"(\d{3,4})kg\s*\(([+-]?\d+|0)\)", s)
    if bm:
        bw = float(bm.group(1))
        bwd = float(bm.group(2))
    else:
        bm = re.search(r"(\d{3,4})kg", s)
        if bm:
            bw = float(bm.group(1))

    tm = re.search(
        r"([^\s()]+(?:\s+[^\s()]+)?)\s*\((\d{2,3}(?:\.\d)?)\)"
        r"\s*([^\s()]+(?:\s+[^\s()]+)?)\s*\((?:栗東|美浦)\)",
        s
    )
    if tm:
        jockey = tm.group(1).strip()
        carried = float(tm.group(2))
        trainer = tm.group(3).strip()
    else:
        tr = re.search(r"([^\s()]+(?:\s+[^\s()]+)?)\s*\((栗東|美浦)\)", s)
        if tr:
            trainer = tr.group(1).strip()

    return horse, odds, pop, sex, age, bw, bwd, jockey, carried, trainer

def parse_jra_official_html(html: str, race_id: str, target_date: date) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    text = _clean_jra_text(soup.get_text(" ", strip=True))
    race_no = int(race_id[-2:])
    venue = VENUE_CODE.get(race_id[4:6], "")

    if "障害" in text[:1500] or "芝→ダート" in text[:1500]:
        return pd.DataFrame(columns=ENTRY_COLS)

    headings = [_clean_jra_text(x.get_text(" ", strip=True)) for x in soup.find_all(["h1","h2","h3","h4"])]
    headings = [x for x in headings if x]
    race_name = f"{race_no}R"
    for h in headings:
        if h in ("出馬表","開催・レース選択"):
            continue
        hh = re.sub(rf"^\s*{race_no}\s*R\s*", "", h, flags=re.I).strip()
        if hh and hh != h:
            race_name = hh
            break

    norm = text.replace(",", "")
    cm = re.search(r"コース\s*[:：]?\s*(\d{3,4})\s*(?:m|メートル)\s*[（(]?\s*(芝|ダート)", norm)
    if cm:
        distance = int(cm.group(1))
        surface = cm.group(2)
    else:
        cm2 = re.search(r"(芝|ダート).*?(\d{3,4})\s*m", norm)
        if not cm2:
            raise AutoDataError(f"{race_id}: JRA公式からコース・距離を取得できません。")
        surface = cm2.group(1)
        distance = int(cm2.group(2))

    gm = re.search(
        r"(?:芝|ダート)\s*[:：]?\s*(不良|稍重|稍|良|重|�.?重)", text
    )
    if not gm:
        gm = re.search(
            r"(?:馬場|馬場状態)\s*[:：]?\s*(不良|稍重|稍|良|重|�.?重)", text
        )
    going = _normalize_going_token(gm.group(1) if gm else "")

    tables = pd.read_html(StringIO(html))
    table = None
    for t in tables:
        tt = _flatten_columns(t)
        joined = " ".join(map(str, tt.columns))
        if "馬番" in joined and "馬名" in joined:
            table = tt
            break
    if table is None:
        raise AutoDataError(f"{race_id}: JRA公式の出走馬表を特定できません。")

    cols = list(table.columns)
    c_gate = _find_col(cols, ["枠"])
    c_no = _find_col(cols, ["馬番"])
    c_name = _find_col(cols, ["馬名"])
    c_sex = _find_col(cols, ["性齢"])
    c_weight = _find_col(cols, ["負担重量","斤量"])
    c_jockey = _find_col(cols, ["騎手"])
    c_trainer = _find_col(cols, ["調教師"])
    c_body = _find_col(cols, ["馬体重"])
    c_odds = _find_col(cols, ["単勝オッズ","オッズ"])
    c_pop = _find_col(cols, ["人気"])

    if not c_no or not c_name:
        raise AutoDataError(f"{race_id}: JRA公式で馬番・馬名列を特定できません。")

    # In mobile/app HTML, the horse column can be a combined cell containing
    # odds/popularity/body/jockey/trainer. Do not treat that same column as
    # separate odds/popularity/etc columns.
    separate = lambda c: c is not None and c != c_name

    rows = []
    for _, rr in table.iterrows():
        hno = _num(rr.get(c_no, ""))
        if pd.isna(hno):
            continue
        hno = int(hno)
        if not (1 <= hno <= 18):
            continue

        gate = _num(rr.get(c_gate, "")) if c_gate else np.nan
        detail = _clean_jra_text(rr.get(c_name, ""))
        horse, odds, pop, sex, age, bw, bwd, jockey, carried, trainer = _jra_horse_detail(detail)

        if separate(c_sex):
            sx, ag = _sex_age(rr.get(c_sex, ""))
            if sx: sex = sx
            if pd.notna(ag): age = ag
        if separate(c_weight):
            x = _num(rr.get(c_weight, ""))
            if pd.notna(x): carried = x
        if separate(c_jockey):
            j = _clean_jra_text(rr.get(c_jockey, ""))
            if j and j.lower() != "nan":
                jockey = re.sub(r"\s*\(.*$", "", j).strip()
        if separate(c_trainer):
            tr = _clean_jra_text(rr.get(c_trainer, ""))
            if tr and tr.lower() != "nan":
                trainer = re.sub(r"\s*\((?:栗東|美浦)\).*$", "", tr).strip()
        if separate(c_body):
            xbw, xbwd = _body(rr.get(c_body, ""))
            if pd.notna(xbw): bw = xbw
            if pd.notna(xbwd): bwd = xbwd
        if separate(c_odds):
            ox = _num(rr.get(c_odds, ""))
            if pd.notna(ox): odds = ox
        if separate(c_pop):
            px = _num(rr.get(c_pop, ""))
            if pd.notna(px): pop = int(px)

        horse = _clean_jra_text(horse)
        if not horse or horse.lower() == "nan":
            continue

        rows.append({
            "date":target_date.isoformat(),
            "course":venue,
            "race_no":f"{race_no}R",
            "race_name":race_name,
            "surface":surface,
            "distance":distance,
            "going":going,
            "horse_no":hno,
            "gate":int(gate) if pd.notna(gate) else np.nan,
            "horse_name":horse,
            "jockey":jockey,
            "trainer":trainer,
            "age":age,
            "sex":sex,
            "weight":carried,
            "body_weight":bw,
            "body_weight_diff":bwd,
            "odds":odds,
            "popularity":pop,
            "race_class":_race_class(text),
            "running_style":"",
        })

    out = pd.DataFrame(rows, columns=ENTRY_COLS)
    if out.empty:
        raise AutoDataError(f"{race_id}: JRA公式から出走馬を取得できません。")
    return out

def _jra_signature_for_race_id(rid: str, target_date: date) -> str:
    return (
        f"01{rid[4:6]}{rid[:4]}{rid[6:8]}{rid[8:10]}"
        f"{rid[10:12]}{target_date.strftime('%Y%m%d')}"
    )

def fetch_jra_official_missing_with_browser(
    race_ids: list[str],
    target_date: date,
    page,
    progress_callback=None,
) -> tuple[list[pd.DataFrame], list[str]]:
    if not race_ids:
        return [], []

    frames, errors = [], []
    groups = {}
    for rid in race_ids:
        groups.setdefault(VENUE_CODE.get(rid[4:6], ""), []).append(rid)

    done = 0
    total = len(race_ids)

    for venue, rids in groups.items():
        seed = JRA_OFFICIAL_SEED_URLS.get((target_date.isoformat(), venue))
        if not seed:
            errors.extend([f"{rid}: JRA公式シードURL未登録" for rid in rids])
            done += len(rids)
            continue

        try:
            page.goto(seed, wait_until="domcontentloaded")
            page.wait_for_timeout(700)
            hrefs = page.locator("a").evaluate_all("els => els.map(a => a.href).filter(Boolean)")
            hrefs = list(dict.fromkeys(hrefs))
            pairs = [(h, urllib.parse.unquote(h)) for h in hrefs]

            link_map = {}
            seed_uq = urllib.parse.unquote(seed)
            for rid in rids:
                sig = _jra_signature_for_race_id(rid, target_date)
                for raw, uq in pairs:
                    if sig in uq and "accessD.html" in uq:
                        link_map[rid] = raw
                        break
                if rid not in link_map and sig in seed_uq:
                    link_map[rid] = seed

            for rid in rids:
                done += 1
                _progress(
                    progress_callback,
                    .50 + .27 * done / max(total, 1),
                    f"JRA公式で阪神不足分を補完中… {done}/{total}"
                )
                url = link_map.get(rid)
                if not url:
                    errors.append(f"{rid}: JRA公式ページ内にレースリンクがありません")
                    continue
                try:
                    page.goto(url, wait_until="domcontentloaded")
                    page.wait_for_timeout(500)
                    out = parse_jra_official_html(page.content(), rid, target_date)
                    if out is not None and len(out):
                        frames.append(out)
                    else:
                        errors.append(f"{rid}: JRA公式ではAI対象外または空データ")
                except Exception as e:
                    errors.append(f"{rid}: JRA公式取得失敗={e}")
        except Exception as e:
            errors.extend([f"{rid}: JRA公式シードページ失敗={e}" for rid in rids])
            done += len(rids)

    return frames, errors
def fetch_missing_with_playwright(
    race_ids: list[str],
    target_date: date,
    progress_callback=None,
    timeout_ms=25000,
) -> tuple[list[pd.DataFrame], list[str]]:
    if not race_ids:
        return [], []

    unsupported = set(unsupported_race_ids(target_date))
    target_ids = [rid for rid in race_ids if rid not in unsupported]
    if not target_ids:
        return [], []

    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        return [], [f"ブラウザ取得を開始できません: playwright未導入 ({e})"]

    frames, errors = [], []
    try:
        with sync_playwright() as p:
            browser = None
            launch_errors = []
            for kwargs in ({"channel":"msedge","headless":True}, {"headless":True}):
                try:
                    browser = p.chromium.launch(**kwargs)
                    break
                except Exception as e:
                    launch_errors.append(str(e))
            if browser is None:
                return [], ["Edge/Chromiumを利用できません: " + " / ".join(launch_errors[-2:])]

            page = browser.new_page(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0 Safari/537.36"
                ),
                locale="ja-JP",
            )
            page.set_default_timeout(timeout_ms)

            still_missing = []
            total = len(target_ids)

            for i, rid in enumerate(target_ids, start=1):
                _progress(progress_callback, .42 + .08*(i-1)/max(total,1),
                          f"通常の予備取得を確認中… {i}/{total}")
                ok = False
                last1 = last2 = ""

                try:
                    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}"
                    page.goto(url, wait_until="domcontentloaded")
                    page.wait_for_timeout(700)
                    out = parse_shutuba_html(page.content(), rid, target_date, {})
                    if out is not None and len(out):
                        frames.append(out); ok = True
                except Exception as e:
                    last1 = str(e)

                if not ok:
                    try:
                        url = f"https://www.daily.co.jp/umaya/race/shutsuba.shtml?race_id={rid}"
                        page.goto(url, wait_until="domcontentloaded")
                        page.wait_for_timeout(600)
                        out = parse_daily_shutuba_html(page.content(), rid, target_date)
                        if out is not None and len(out):
                            frames.append(out); ok = True
                    except Exception as e:
                        last2 = str(e)

                if not ok:
                    still_missing.append(rid)
                    errors.append(
                        f"{rid}: 通常予備取得では未取得"
                        + (f" / netkeiba={last1}" if last1 else "")
                        + (f" / Daily={last2}" if last2 else "")
                    )

            if still_missing:
                official_frames, official_errors = fetch_jra_official_missing_with_browser(
                    still_missing, target_date, page, progress_callback
                )
                frames.extend(official_frames)
                errors.extend(official_errors)

            browser.close()
    except Exception as e:
        errors.append(f"ブラウザ取得全体エラー: {e}")

    return frames, errors

def ensure_complete_known_day(
    frames: list[pd.DataFrame],
    target_date: date,
    progress_callback=None,
) -> tuple[list[pd.DataFrame], list[str], dict]:
    """
    For known dates, do not silently accept one race as a complete day.
    Try browser fallback for missing races.
    """
    expected = expected_race_ids(target_date)
    if not expected:
        return frames, [], {
            "expected_races":0,
            "fetched_races":len(frames),
            "missing_race_ids":[],
        }

    found = _have_race_ids(frames, target_date)
    missing = [rid for rid in expected if rid not in found]
    browser_errors = []

    if missing:
        extra, browser_errors = fetch_missing_with_playwright(
            missing, target_date, progress_callback=progress_callback
        )
        if extra:
            frames.extend(extra)

    found2 = _have_race_ids(frames, target_date)
    missing2 = [rid for rid in expected if rid not in found2]

    return frames, browser_errors, {
        "expected_races": len(expected),
        "fetched_races": len(found2),
        "missing_race_ids": missing2,
        "excluded_races": unsupported_race_ids(target_date),
    }

def create_entries_csv(
    target_date: date,
    generated_dir: Path,
    seed_dir: Path|None=None,
    progress_callback=None,
    request_interval=0.7,
    allow_network=True,
) -> tuple[Path, dict]:
    """
    Automatically create one entry CSV containing all flat JRA races for target_date.
    Uses a public current-race page, and rate-limits user-triggered requests.
    """
    generated_dir = Path(generated_dir)
    generated_dir.mkdir(parents=True, exist_ok=True)
    out = generated_dir/f"entries_{target_date.strftime('%Y%m%d')}_auto.csv"

    if allow_network:
        _progress(progress_callback,.28,f"{target_date.isoformat()} の開催レースを確認しています…")
        s = _session()
        race_ids = discover_race_ids(target_date, s)
        if race_ids:
            frames=[]
            errors=[]
            total=len(race_ids)
            for i,rid in enumerate(race_ids, start=1):
                _progress(progress_callback,.30 + .46*(i-1)/max(total,1),
                          f"出走表を自動作成中… {i}/{total}レース")
                try:
                    f=fetch_race_entries(rid,target_date,s)
                    if len(f):
                        frames.append(f)
                except Exception as e:
                    errors.append(f"{rid}: {e}")
                if i < total:
                    time.sleep(max(float(request_interval),0.0))
            # Known race days (9/26, 9/27) must not treat one race as
            # a complete day. Browser fallback attempts all missing race IDs.
            frames, browser_errors, coverage = ensure_complete_known_day(
                frames, target_date, progress_callback=progress_callback
            )
            errors.extend(browser_errors)

            if frames:
                df=pd.concat(frames,ignore_index=True)
                # Prefer one row per date/course/race/horse.
                df=df.drop_duplicates(["date","course","race_no","horse_no","horse_name"])
                actual_races=int(df[["course","race_no"]].drop_duplicates().shape[0])

                # On known dates, never silently call a one-race file "complete".
                # Save partial CSV for diagnosis, but raise a clear message.
                if coverage.get("expected_races",0) and coverage.get("missing_race_ids"):
                    partial = generated_dir/f"entries_{target_date.strftime('%Y%m%d')}_PARTIAL.csv"
                    df.to_csv(partial,index=False,encoding="utf-8-sig")
                    missing_text=", ".join(coverage["missing_race_ids"])
                    raise AutoDataError(
                        f"{target_date.isoformat()} は "
                        f"{coverage['fetched_races']}/{coverage['expected_races']}レース取得です。"
                        f" AI対象の平地レースが揃っていないため停止しました。"
                        f" 未取得race_id: {missing_text}"
                        f" / 取得済みデータは {partial.name} に保存しました。"
                    )

                df.to_csv(out,index=False,encoding="utf-8-sig")
                missing=validate_entries(df)
                if missing:
                    raise AutoDataError("自動作成した出走表CSVに不足列があります: "+", ".join(missing))
                return out,{
                    "source":"公開出走表から自動取得（HTTP＋予備＋ブラウザ補完）",
                    "race_ids":race_ids,
                    "races":actual_races,
                    "rows":len(df),
                    "errors":errors,
                    "network":True,
                    "expected_races":coverage.get("expected_races",0),
                    "missing_race_ids":coverage.get("missing_race_ids",[]),
                    "excluded_races":coverage.get("excluded_races",{}),
                }

    # Do not silently fall back to the old one-race 9/27 CSV.
    # It is kept only as reference data in seed_data.
    if target_date.isoformat()=="2026-09-27":
        raise AutoDataError(
            "9月27日の全レース取得に失敗しました。"
            " Ver.1.1.6ではスプリンターズSだけを全レースとして表示しません。"
            " ネット接続後にもう一度「データを自動作成・取り込み」を押してください。"
        )

    known = KNOWN_RACE_DAYS.get(target_date.isoformat(), [])
    extra = f" 既知レースID={len(known)}件。" if known else ""
    raise AutoDataError(
        "出走表CSVを自動作成できませんでした。"
        "インターネット接続、対象日の出馬表公開状況を確認してください。"
        + extra
    )

def build_and_load_all(seed_dir, generated_dir, target_date, progress_callback=None, force_training=False, allow_network=True):
    generated_dir=Path(generated_dir)
    seed_dir=Path(seed_dir)
    hp=ensure_training_csv(seed_dir,generated_dir,force_training,progress_callback)
    ep,entry_info=create_entries_csv(
        target_date,generated_dir,seed_dir,
        progress_callback=progress_callback,
        allow_network=allow_network,
    )
    _progress(progress_callback,.82,"作成したCSVを読み込んでいます…")
    h=pd.read_csv(hp,low_memory=False)
    e=pd.read_csv(ep,low_memory=False)
    mh=validate_raw_history(h); me=validate_entries(e)
    if mh: raise AutoDataError("学習CSVに不足列があります: "+", ".join(mh))
    if me: raise AutoDataError("出走表CSVに不足列があります: "+", ".join(me))
    _progress(progress_callback,.90,"CSVの自動取り込みが完了しました。")
    info={
        "history_file":hp.name,
        "entries_file":ep.name,
        "history_rows":len(h),
        "entries_rows":len(e),
        "entry_date":target_date.isoformat(),
        "entry_source":entry_info.get("source",""),
        "race_count":entry_info.get("races",0),
        "network":entry_info.get("network",False),
        "entry_errors":entry_info.get("errors",[]),
    }
    manifest=generated_dir/"manifest.json"
    manifest.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
    return h,e,info
def _is_present(v):
    if pd.isna(v):
        return False
    s = str(v).strip()
    return s not in ("", "nan", "None", "<NA>")

def merge_entry_snapshots(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge a newly fetched snapshot into the existing entries.
    New non-empty values win. Missing new values never erase existing values.
    """
    if old_df is None or len(old_df) == 0:
        out = new_df.copy()
        return out[ENTRY_COLS]

    if new_df is None or len(new_df) == 0:
        out = old_df.copy()
        return out[ENTRY_COLS]

    keys = ["date","course","race_no","horse_no","horse_name"]
    old = old_df.copy()
    new = new_df.copy()

    for c in ENTRY_COLS:
        if c not in old.columns:
            old[c] = np.nan
        if c not in new.columns:
            new[c] = np.nan

    old_idx = old.set_index(keys, drop=False)
    new_idx = new.set_index(keys, drop=False)

    # Union keeps new horses/races if entries changed after an earlier snapshot.
    all_idx = old_idx.index.union(new_idx.index)
    result_rows = []
    for idx in all_idx:
        orow = old_idx.loc[idx] if idx in old_idx.index else None
        nrow = new_idx.loc[idx] if idx in new_idx.index else None

        # In unlikely duplicate-index cases, use the last row.
        if isinstance(orow, pd.DataFrame):
            orow = orow.iloc[-1]
        if isinstance(nrow, pd.DataFrame):
            nrow = nrow.iloc[-1]

        row = {}
        for c in ENTRY_COLS:
            nv = nrow[c] if nrow is not None else np.nan
            ov = orow[c] if orow is not None else np.nan
            row[c] = nv if _is_present(nv) else ov
        result_rows.append(row)

    out = pd.DataFrame(result_rows, columns=ENTRY_COLS)
    out["horse_no"] = pd.to_numeric(out["horse_no"], errors="coerce")
    out = out.dropna(subset=["horse_no","horse_name"])
    out = out.sort_values(["date","course","race_no","horse_no"]).reset_index(drop=True)
    return out

def entry_completeness(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return {
            "rows":0, "races":0, "odds":0, "popularity":0,
            "body_weight":0, "body_weight_diff":0, "going_races":0
        }

    d = df.copy()
    race_cols = ["date","course","race_no"]
    races = d[race_cols].drop_duplicates()
    going_by_race = (
        d.assign(_going=d["going"].astype(str).str.strip())
         .groupby(race_cols, dropna=False)["_going"]
         .apply(lambda s: bool((~s.isin(["","nan","None"])).any()))
    )
    return {
        "rows": int(len(d)),
        "races": int(len(races)),
        "odds": int(pd.to_numeric(d["odds"],errors="coerce").notna().sum()),
        "popularity": int(pd.to_numeric(d["popularity"],errors="coerce").notna().sum()),
        "body_weight": int(pd.to_numeric(d["body_weight"],errors="coerce").notna().sum()),
        "body_weight_diff": int(pd.to_numeric(d["body_weight_diff"],errors="coerce").notna().sum()),
        "going_races": int(going_by_race.sum()) if len(going_by_race) else 0,
    }

def refresh_entries_csv(
    target_date: date,
    generated_dir: Path,
    existing_entries: pd.DataFrame|None = None,
    progress_callback=None,
    request_interval=0.7,
    fetcher=None,
) -> tuple[Path, pd.DataFrame, dict]:
    """
    Refresh current-race data and keep a timestamped snapshot.
    This routine does NOT silently fall back to bundled stale data.
    If the network fetch cannot return any race, it raises an error so the UI
    can tell the user that the current information was not refreshed.
    """
    generated_dir = Path(generated_dir)
    generated_dir.mkdir(parents=True, exist_ok=True)

    _progress(progress_callback, .05, "最新の出走表・オッズ・馬体重を確認しています…")
    s = _session()
    race_ids = discover_race_ids(target_date, s)
    if not race_ids:
        raise AutoDataError(
            f"{target_date.isoformat()} の開催レースを確認できませんでした。"
            " 出馬表公開前、通信障害、取得元ページ変更の可能性があります。"
        )

    frames = []
    errors = []
    total = len(race_ids)
    fetch = fetcher or fetch_race_entries

    for i, rid in enumerate(race_ids, start=1):
        _progress(
            progress_callback,
            .08 + .62*(i-1)/max(total,1),
            f"最新データを取得中… {i}/{total}レース"
        )
        try:
            # custom fetcher for test can accept 2 args; production fetcher takes session too
            if fetcher is None:
                f = fetch(rid, target_date, s)
            else:
                f = fetch(rid, target_date)
            if f is not None and len(f):
                frames.append(f)
        except Exception as e:
            errors.append(f"{rid}: {e}")
        if fetcher is None and i < total:
            time.sleep(max(float(request_interval),0.0))

    if not frames:
        raise AutoDataError(
            "最新データを1レースも取得できませんでした。"
            + (" / " + errors[0] if errors else "")
        )

    live = pd.concat(frames, ignore_index=True)
    live = live.drop_duplicates(["date","course","race_no","horse_no","horse_name"])

    _progress(progress_callback, .74, "取得した最新値を現在の出走表へ反映しています…")
    merged = merge_entry_snapshots(existing_entries, live)

    missing = validate_entries(merged)
    if missing:
        raise AutoDataError("更新後の出走表CSVに不足列があります: " + ", ".join(missing))

    # Stable canonical file + timestamped history snapshot.
    canonical = generated_dir/f"entries_{target_date.strftime('%Y%m%d')}_auto.csv"
    stamp = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
    snapshot = generated_dir/f"entries_{target_date.strftime('%Y%m%d')}_{stamp}.csv"

    merged.to_csv(canonical, index=False, encoding="utf-8-sig")
    merged.to_csv(snapshot, index=False, encoding="utf-8-sig")

    before = entry_completeness(existing_entries)
    after = entry_completeness(merged)
    info = {
        "source": "公開出走表から最新データ更新",
        "race_ids": race_ids,
        "races": after["races"],
        "rows": after["rows"],
        "errors": errors,
        "network": True,
        "updated_at": datetime.now(JST).isoformat(timespec="seconds"),
        "snapshot_file": snapshot.name,
        "before": before,
        "after": after,
        "odds_gain": after["odds"] - before["odds"],
        "body_weight_gain": after["body_weight"] - before["body_weight"],
        "going_gain": after["going_races"] - before["going_races"],
    }

    log_path = generated_dir/"update_history.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(info, ensure_ascii=False) + "\n")

    _progress(progress_callback, .84, "最新データCSVの更新が完了しました。")
    return canonical, merged, info

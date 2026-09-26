from __future__ import annotations
from datetime import date
from urllib.parse import urljoin, unquote
import time
import pandas as pd
from bs4 import BeautifulSoup

from auto_data_builder import (
    _session, discover_race_ids, fetch_race_entries, parse_jra_official_html,
    expected_race_ids, unsupported_race_ids, JRA_OFFICIAL_SEED_URLS,
    _jra_signature_for_race_id, VENUE_CODE, ENTRY_COLS, AutoDataError,
)

def _official_http_missing(race_ids, target_date, session):
    frames, errors = [], []
    by_venue = {}
    for rid in race_ids:
        by_venue.setdefault(VENUE_CODE.get(rid[4:6], ""), []).append(rid)

    for venue, ids in by_venue.items():
        seed = JRA_OFFICIAL_SEED_URLS.get((target_date.isoformat(), venue))
        if not seed:
            errors.extend([f"{rid}: JRA公式補完URL未登録" for rid in ids])
            continue
        try:
            r = session.get(seed, timeout=20)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")
            links = []
            for a in soup.find_all("a", href=True):
                links.append(urljoin(seed, a["href"]))
            links = list(dict.fromkeys(links))
            link_pairs = [(x, unquote(x)) for x in links]
            seed_uq = unquote(seed)

            for rid in ids:
                sig = _jra_signature_for_race_id(rid, target_date)
                url = None
                for raw, uq in link_pairs:
                    if sig in uq and "accessD.html" in uq:
                        url = raw
                        break
                if url is None and sig in seed_uq:
                    url = seed
                if not url:
                    errors.append(f"{rid}: JRA公式レースリンクを発見できません")
                    continue
                try:
                    rr = session.get(url, timeout=20)
                    rr.raise_for_status()
                    out = parse_jra_official_html(rr.text, rid, target_date)
                    if out is not None and len(out):
                        frames.append(out)
                    else:
                        errors.append(f"{rid}: JRA公式は空データまたはAI対象外")
                except Exception as e:
                    errors.append(f"{rid}: JRA公式取得失敗={e}")
        except Exception as e:
            errors.extend([f"{rid}: JRA公式開催ページ失敗={e}" for rid in ids])
    return frames, errors

def fetch_entries_cloud(target_date: date, request_interval=0.35):
    s = _session()
    race_ids = discover_race_ids(target_date, s)
    if not race_ids:
        raise AutoDataError(f"{target_date.isoformat()} の開催レースを確認できませんでした。")

    unsupported = set(unsupported_race_ids(target_date))
    expected = expected_race_ids(target_date)
    frames, errors, got = [], [], set()

    for rid in race_ids:
        if rid in unsupported:
            continue
        try:
            f = fetch_race_entries(rid, target_date, s, timeout=15)
            if f is not None and len(f):
                frames.append(f)
                got.add(rid)
        except Exception as e:
            errors.append(f"{rid}: {e}")
        time.sleep(max(request_interval, 0))

    if expected:
        missing = [rid for rid in expected if rid not in got]
        if missing:
            extra, err2 = _official_http_missing(missing, target_date, s)
            errors.extend(err2)
            for f in extra:
                if f is not None and len(f):
                    frames.append(f)
                    row = f.iloc[0]
                    course = str(row["course"])
                    rn = int(str(row["race_no"]).replace("R",""))
                    for rid in missing:
                        if VENUE_CODE.get(rid[4:6]) == course and int(rid[-2:]) == rn:
                            got.add(rid)
                            break

        missing = [rid for rid in expected if rid not in got]
        if missing:
            raise AutoDataError(
                f"{target_date.isoformat()} はAI対象 {len(got)}/{len(expected)}レース取得です。"
                " 未取得race_id: " + ", ".join(missing)
            )

    if not frames:
        raise AutoDataError("出走表を取得できませんでした。")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["date","course","race_no","horse_no","horse_name"])
    for c in ENTRY_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    return df[ENTRY_COLS].reset_index(drop=True), {
        "date": target_date.isoformat(),
        "races": int(df[["course","race_no"]].drop_duplicates().shape[0]),
        "rows": int(len(df)),
        "errors": errors,
        "excluded": unsupported_race_ids(target_date),
    }

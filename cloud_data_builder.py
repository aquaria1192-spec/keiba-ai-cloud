from __future__ import annotations
from datetime import date
import time
import pandas as pd

from auto_data_builder import (
    _session, discover_race_ids, fetch_race_entries, parse_jra_official_html,
    expected_race_ids, unsupported_race_ids,
    VENUE_CODE, ENTRY_COLS, AutoDataError,
)
from race_day_context import collect_official_entry_urls, _decode_jra_response

def fetch_entries_cloud(target_date: date, request_interval=0.35):
    s = _session()
    race_ids = discover_race_ids(target_date, s)
    if not race_ids:
        raise AutoDataError(f"{target_date.isoformat()} の開催レースを確認できませんでした。")

    courses=sorted(set(VENUE_CODE.get(rid[4:6],"") for rid in race_ids if VENUE_CODE.get(rid[4:6],"")))
    official_urls=collect_official_entry_urls(target_date,courses,s)

    unsupported = set(unsupported_race_ids(target_date))
    expected = expected_race_ids(target_date)
    frames, errors, got = [], [], set()
    race_id_map={}

    for rid in race_ids:
        if rid in unsupported:
            continue
        try:
            f = fetch_race_entries(rid, target_date, s, timeout=15)
            if f is not None and len(f):
                frames.append(f)
                got.add(rid)
                row=f.iloc[0]
                race_id_map[f"{row['course']}|{str(row['race_no'])}"]=rid
        except Exception as e:
            errors.append(f"{rid}: {e}")
        time.sleep(max(request_interval, 0))

    # JRA official third source for missing flat races.
    if expected:
        missing = [rid for rid in expected if rid not in got]
        for rid in missing[:]:
            url=official_urls.get(rid)
            if not url:
                continue
            try:
                rr=s.get(url,timeout=20)
                rr.raise_for_status()
                f=parse_jra_official_html(_decode_jra_response(rr),rid,target_date)
                if f is not None and len(f):
                    frames.append(f)
                    got.add(rid)
                    row=f.iloc[0]
                    race_id_map[f"{row['course']}|{str(row['race_no'])}"]=rid
            except Exception as e:
                errors.append(f"{rid}: JRA公式補完失敗={e}")

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
        "race_id_map":race_id_map,
        "official_entry_urls":official_urls,
    }

from __future__ import annotations

import itertools
import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

BET_TYPES = ("単勝", "馬連", "ワイド", "三連複", "三連単")
TYPE_TO_GROUP = {
    1: "単勝",
    4: "馬連",
    5: "ワイド",
    7: "三連複",
    8: "三連単",
}
ORDERED_TYPES = {"三連単"}


def _to_float(value):
    if value is None:
        return None
    s = str(value).replace(",", "").strip()
    if s in ("", "---.-", "-", "None", "nan"):
        return None
    try:
        return float(s)
    except Exception:
        return None


def _to_int(value):
    if value is None:
        return None
    s = str(value).replace(",", "").replace("円", "").strip()
    m = re.search(r"-?\d+", s)
    return int(m.group(0)) if m else None


def _split_combo_key(raw_key: str):
    key = str(raw_key or "").strip()
    if not key or len(key) % 2:
        return None
    try:
        nums = [int(key[i:i+2]) for i in range(0, len(key), 2)]
    except Exception:
        return None
    if any(n <= 0 for n in nums):
        return None
    return nums


def normalize_combo(combo: str, bet_type: str):
    s = str(combo or "").strip()
    if not s:
        return ""
    nums = [int(x) for x in re.findall(r"\d+", s)]
    if not nums:
        return ""
    if bet_type in ORDERED_TYPES:
        return "→".join(str(x) for x in nums)
    if bet_type == "単勝":
        return str(nums[0])
    return "-".join(str(x) for x in sorted(nums))


def _format_api_combo(nums, bet_type):
    if not nums:
        return ""
    if bet_type in ORDERED_TYPES:
        return "→".join(str(x) for x in nums)
    if bet_type == "単勝":
        return str(nums[0])
    return "-".join(str(x) for x in sorted(nums))


def parse_market_odds_payload(payload: dict, requested_type: int):
    """Parse one netkeiba odds API payload into {bet_type: {combo: data}}."""
    if not isinstance(payload, dict):
        return {"status": "", "official_datetime": "", "odds": {}}
    status = str(payload.get("status", "") or "")
    if status not in ("middle", "result"):
        return {"status": status, "official_datetime": "", "odds": {}}
    data = payload.get("data") or {}
    raw_odds = data.get("odds") or {}
    out = {}
    for group_code, combos in raw_odds.items():
        try:
            group_num = int(str(group_code))
        except Exception:
            continue
        if group_num == 2:
            continue
        bet_type = TYPE_TO_GROUP.get(group_num)
        if not bet_type or not isinstance(combos, dict):
            continue
        parsed = {}
        for raw_key, values in combos.items():
            nums = _split_combo_key(str(raw_key))
            if not nums or not isinstance(values, (list, tuple)) or not values:
                continue
            low = _to_float(values[0])
            high = _to_float(values[1]) if len(values) > 1 else None
            if low is None:
                continue
            combo = _format_api_combo(nums, bet_type)
            if not combo:
                continue
            pop = _to_int(values[2]) if len(values) > 2 else None
            parsed[combo] = {
                "odds": float(low),
                "odds_low": float(low),
                "odds_high": float(high) if high not in (None, 0) else float(low),
                "popularity": pop,
            }
        if parsed:
            out[bet_type] = parsed
    return {
        "status": status,
        "official_datetime": str(data.get("official_datetime", "") or ""),
        "odds": out,
    }


def fetch_market_odds(race_id: str, session=None, timeout=10):
    """Fetch live/final odds for ticket types used by the recommendation UI."""
    race_id = str(race_id or "").strip()
    if not re.fullmatch(r"\d{12}", race_id):
        return {
            "race_id": race_id,
            "status": "invalid_race_id",
            "official_datetime": "",
            "odds": {},
            "errors": ["race_id不正"],
        }
    s = session or requests.Session()
    merged = {}
    statuses = []
    official_times = []
    errors = []
    for type_code in (1, 4, 5, 7, 8):
        url = (
            "https://race.netkeiba.com/api/api_get_jra_odds.html"
            f"?race_id={race_id}&type={type_code}&action=update"
        )
        try:
            r = s.get(url, timeout=timeout)
            r.raise_for_status()
            parsed = parse_market_odds_payload(r.json(), type_code)
            if parsed["status"]:
                statuses.append(parsed["status"])
            if parsed["official_datetime"]:
                official_times.append(parsed["official_datetime"])
            for bet_type, combos in parsed["odds"].items():
                merged.setdefault(bet_type, {}).update(combos)
        except Exception as exc:
            errors.append(f"type={type_code}: {exc}")
    status = "result" if "result" in statuses and "middle" not in statuses else (
        "middle" if "middle" in statuses else (statuses[0] if statuses else "")
    )
    return {
        "race_id": race_id,
        "status": status,
        "official_datetime": max(official_times) if official_times else "",
        "odds": merged,
        "errors": errors,
    }


def _ranked_horses(detail: pd.DataFrame):
    d = detail.copy()
    d["_rank_num"] = pd.to_numeric(d.get("順位"), errors="coerce")
    d["_win"] = pd.to_numeric(d.get("win_prob"), errors="coerce")
    d["_top3"] = pd.to_numeric(d.get("top3_prob"), errors="coerce")
    d["_odds"] = pd.to_numeric(d.get("odds"), errors="coerce")
    d["_horse_no"] = pd.to_numeric(d.get("horse_no"), errors="coerce")
    d = d.dropna(subset=["_horse_no"]).sort_values(["_rank_num", "_win"], ascending=[True, False])
    # Win probabilities are used as Plackett-Luce strengths. Normalise to a
    # proper race-level distribution so ordered-combination probabilities sum.
    strengths = np.clip(d["_win"].fillna(0).to_numpy(dtype=float), 1e-9, None)
    total = float(strengths.sum())
    if not math.isfinite(total) or total <= 0:
        strengths = np.ones(len(d), dtype=float) / max(len(d), 1)
    else:
        strengths = strengths / total
    d["_strength"] = strengths
    return d


def _ordered_prob(order, weights):
    remaining = 1.0
    p = 1.0
    for horse in order:
        w = float(weights.get(int(horse), 0.0))
        if remaining <= 1e-12 or w <= 0:
            return 0.0
        p *= w / remaining
        remaining -= w
    return max(0.0, min(1.0, float(p)))


def _quinella_prob(a, b, weights):
    return _ordered_prob((a, b), weights) + _ordered_prob((b, a), weights)


def _trio_prob(a, b, c, weights):
    return sum(_ordered_prob(order, weights) for order in itertools.permutations((a, b, c), 3))


def _wide_prob(a, b, weights):
    others = [x for x in weights if x not in (a, b)]
    p = 0.0
    for c in others:
        for order in itertools.permutations((a, b, c), 3):
            p += _ordered_prob(order, weights)
    return max(0.0, min(1.0, p))


def _ticket_candidates(detail: pd.DataFrame, market_odds: dict | None):
    d = _ranked_horses(detail)
    if d.empty:
        return []
    weights = {
        int(row["_horse_no"]): float(row["_strength"])
        for _, row in d.iterrows()
    }
    top5 = [int(x) for x in d.head(5)["_horse_no"].tolist()]
    top4 = top5[:4]
    market = (market_odds or {}).get("odds") or {}

    # If the live type=1 call failed, preserve the single odds already present
    # in the race card so single-value betting can still work.
    win_market = dict(market.get("単勝") or {})
    for _, row in d.iterrows():
        no = int(row["_horse_no"])
        ov = row["_odds"]
        if pd.notna(ov) and float(ov) > 0 and str(no) not in win_market:
            win_market[str(no)] = {
                "odds": float(ov), "odds_low": float(ov), "odds_high": float(ov),
                "popularity": None,
            }

    def odds_for(bet_type, combo):
        pool = win_market if bet_type == "単勝" else (market.get(bet_type) or {})
        x = pool.get(combo)
        if not isinstance(x, dict):
            return None, None
        low = _to_float(x.get("odds_low", x.get("odds")))
        high = _to_float(x.get("odds_high", x.get("odds")))
        # For wide, use the lower edge of the range for conservative EV.
        used = low if bet_type == "ワイド" else _to_float(x.get("odds"))
        if used is None:
            used = low
        return used, high

    candidates = []
    for no in top5:
        combo = str(no)
        odds, high = odds_for("単勝", combo)
        if odds:
            candidates.append(("単勝", combo, weights.get(no, 0.0), odds, high))

    for a, b in itertools.combinations(top5, 2):
        combo = normalize_combo(f"{a}-{b}", "馬連")
        odds, high = odds_for("馬連", combo)
        if odds:
            candidates.append(("馬連", combo, _quinella_prob(a, b, weights), odds, high))

        combo = normalize_combo(f"{a}-{b}", "ワイド")
        odds, high = odds_for("ワイド", combo)
        if odds:
            candidates.append(("ワイド", combo, _wide_prob(a, b, weights), odds, high))

    for a, b, c in itertools.combinations(top5, 3):
        combo = normalize_combo(f"{a}-{b}-{c}", "三連複")
        odds, high = odds_for("三連複", combo)
        if odds:
            candidates.append(("三連複", combo, _trio_prob(a, b, c, weights), odds, high))

    for order in itertools.permutations(top4, 3):
        combo = normalize_combo("→".join(str(x) for x in order), "三連単")
        odds, high = odds_for("三連単", combo)
        if odds:
            candidates.append(("三連単", combo, _ordered_prob(order, weights), odds, high))

    rows = []
    for bet_type, combo, prob, odds, odds_high in candidates:
        ev = float(prob) * float(odds)
        rows.append({
            "bet_type": bet_type,
            "combo": combo,
            "probability": float(prob),
            "odds": float(odds),
            "odds_high": float(odds_high) if odds_high else float(odds),
            "ev": ev,
        })
    return rows


def build_value_bet_plan(
    detail: pd.DataFrame,
    market_odds: dict | None = None,
    race_id: str = "",
    stake_per_ticket: int = 100,
    min_ev: float = 1.05,
    max_tickets: int = 8,
):
    """Select 100-yen tickets using AI probabilities × current market odds.

    Combination probabilities are derived from the race-level AI win
    probabilities with a Plackett-Luce ordering model. Wide EV uses the lower
    published odds edge so its expected value is conservative.
    """
    empty = {
        "version": "odds-value-v1",
        "race_id": str(race_id or ""),
        "status": "見送り",
        "stake_per_ticket": int(stake_per_ticket),
        "ticket_count": 0,
        "stake_total": 0,
        "min_ev": float(min_ev),
        "odds_status": str((market_odds or {}).get("status", "") or ""),
        "official_datetime": str((market_odds or {}).get("official_datetime", "") or ""),
        "tickets": [],
        "買い目": "見送り",
    }
    if detail is None or len(detail) == 0:
        return empty

    rows = _ticket_candidates(detail, market_odds)
    if not rows:
        empty["reason"] = "買い目用オッズを取得できませんでした。"
        return empty

    # Slightly stricter thresholds for high-variance exotic tickets.
    thresholds = {
        "単勝": max(float(min_ev), 1.03),
        "馬連": max(float(min_ev), 1.05),
        "ワイド": max(float(min_ev), 1.03),
        "三連複": max(float(min_ev), 1.08),
        "三連単": max(float(min_ev), 1.12),
    }
    type_caps = {"単勝": 1, "馬連": 2, "ワイド": 2, "三連複": 2, "三連単": 2}

    chosen = []
    for bet_type in BET_TYPES:
        q = [x for x in rows if x["bet_type"] == bet_type and x["ev"] >= thresholds[bet_type]]
        q.sort(key=lambda x: (x["ev"], x["probability"]), reverse=True)
        chosen.extend(q[:type_caps[bet_type]])

    chosen.sort(key=lambda x: (x["ev"], x["probability"]), reverse=True)
    chosen = chosen[: int(max_tickets)]

    if not chosen:
        best = max(rows, key=lambda x: x["ev"])
        empty["reason"] = (
            "AI確率と現在オッズから期待値1.0超の候補がないため見送り。"
            f" 最高候補は{best['bet_type']} {best['combo']} / 期待値{best['ev']:.2f}"
        )
        empty["best_candidate"] = best
        return empty

    tickets = []
    for x in chosen:
        t = dict(x)
        t["stake"] = int(stake_per_ticket)
        t["reason"] = (
            f"AI推定{t['probability']*100:.1f}% × "
            f"オッズ{t['odds']:.1f} = 期待値{t['ev']:.2f}"
        )
        tickets.append(t)

    by_type = []
    for bet_type in BET_TYPES:
        combos = [t["combo"] for t in tickets if t["bet_type"] == bet_type]
        if combos:
            by_type.append(f"{bet_type} " + " / ".join(combos))
    return {
        "version": "odds-value-v1",
        "race_id": str(race_id or ""),
        "status": "買い",
        "stake_per_ticket": int(stake_per_ticket),
        "ticket_count": int(len(tickets)),
        "stake_total": int(len(tickets) * int(stake_per_ticket)),
        "min_ev": float(min_ev),
        "odds_status": str((market_odds or {}).get("status", "") or ""),
        "official_datetime": str((market_odds or {}).get("official_datetime", "") or ""),
        "tickets": tickets,
        "買い目": "｜".join(by_type) if by_type else "見送り",
    }


def plan_to_json(plan: dict):
    return json.dumps(plan or {}, ensure_ascii=False, separators=(",", ":"))


def plan_from_json(raw: Any):
    if isinstance(raw, dict):
        return raw
    s = str(raw or "").strip()
    if not s or s.lower() == "nan":
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


_CLASS_TO_BET_TYPE = {
    "tan": "単勝",
    "fuku": "複勝",
    "waku": "枠連",
    "uren": "馬連",
    "wide": "ワイド",
    "utan": "馬単",
    "sanrenpuku": "三連複",
    "sanrentan": "三連単",
}


def _split_br(td):
    if td is None:
        return []
    return [x.strip() for x in td.get_text("\n").split("\n") if x.strip()]


def _bet_type_from_th(th):
    if th is None:
        return ""
    for cls in th.get("class") or []:
        if cls in _CLASS_TO_BET_TYPE:
            return _CLASS_TO_BET_TYPE[cls]
    label = th.get_text(strip=True)
    for bet_type in _CLASS_TO_BET_TYPE.values():
        if bet_type in label:
            return bet_type
    return ""


def parse_payouts_html(html: str):
    """Parse official 100-yen payouts from netkeiba-style result tables."""
    soup = BeautifulSoup(str(html or ""), "lxml")
    tables = soup.find_all("table", class_="pay_table_01")
    if not tables:
        tables = soup.find_all("table", class_="pay_block")
    out = {}
    for table in tables:
        for tr in table.find_all("tr"):
            th = tr.find("th")
            bet_type = _bet_type_from_th(th)
            if not bet_type:
                continue
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue
            combos = _split_br(tds[0])
            txt_r = [td for td in tds if "txt_r" in (td.get("class") or [])]
            amount_td = txt_r[0] if txt_r else (tds[1] if len(tds) > 1 else None)
            amounts = _split_br(amount_td)
            if not combos or not amounts:
                continue
            if bet_type in ("単勝", "複勝", "ワイド"):
                pairs = zip(combos, amounts)
            else:
                pairs = [(combos[0], amounts[0])]
            for combo_raw, amount_raw in pairs:
                amount = _to_int(amount_raw)
                combo = normalize_combo(combo_raw, bet_type)
                if amount is not None and combo:
                    out.setdefault(bet_type, {})[combo] = int(amount)
    return out


def settle_bet_plan(plan: dict | str, payouts: dict | None):
    plan = plan_from_json(plan)
    tickets = list(plan.get("tickets") or [])
    stake_per_ticket = int(plan.get("stake_per_ticket") or 100)
    total_stake = int(sum(int(t.get("stake") or stake_per_ticket) for t in tickets))
    if not tickets:
        return {
            "status": "買い目なし",
            "ticket_count": 0,
            "hit_count": 0,
            "stake": 0,
            "payout": 0,
            "profit": 0,
            "roi": np.nan,
            "results": [],
        }

    if not payouts:
        return {
            "status": "払戻未取得",
            "ticket_count": len(tickets),
            "hit_count": 0,
            "stake": total_stake,
            "payout": np.nan,
            "profit": np.nan,
            "roi": np.nan,
            "results": [],
        }

    results = []
    total_payout = 0
    hits = 0
    unknown_types = set()
    for ticket in tickets:
        bet_type = str(ticket.get("bet_type", ""))
        combo = normalize_combo(ticket.get("combo", ""), bet_type)
        stake = int(ticket.get("stake") or stake_per_ticket)
        pool = payouts.get(bet_type)
        if pool is None:
            unknown_types.add(bet_type)
            results.append({
                **ticket, "combo": combo, "hit": None,
                "payout_per_100": None, "payout": None,
            })
            continue
        payout_100 = int(pool.get(combo, 0) or 0)
        payout = int(round(payout_100 * (stake / 100.0)))
        hit = payout_100 > 0
        hits += int(hit)
        total_payout += payout
        results.append({
            **ticket,
            "combo": combo,
            "hit": hit,
            "payout_per_100": payout_100,
            "payout": payout,
        })

    if unknown_types:
        return {
            "status": "一部払戻未取得",
            "ticket_count": len(tickets),
            "hit_count": hits,
            "stake": total_stake,
            "payout": np.nan,
            "profit": np.nan,
            "roi": np.nan,
            "results": results,
            "unknown_types": sorted(unknown_types),
        }

    profit = int(total_payout - total_stake)
    return {
        "status": "精算済み",
        "ticket_count": len(tickets),
        "hit_count": hits,
        "stake": total_stake,
        "payout": int(total_payout),
        "profit": profit,
        "roi": (float(total_payout) / total_stake) if total_stake else np.nan,
        "results": results,
    }

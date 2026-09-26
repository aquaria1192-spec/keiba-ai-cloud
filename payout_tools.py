from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

BET_TYPE_MAP = {
    "単勝":"単勝","複勝":"複勝","枠連":"枠連","ワイド":"ワイド",
    "馬連":"馬連","馬単":"馬単",
    "3連複":"三連複","三連複":"三連複",
    "3連単":"三連単","三連単":"三連単",
}
UNORDERED_TYPES={"枠連","ワイド","馬連","三連複"}

def canonical_bet_type(kind):
    return BET_TYPE_MAP.get(str(kind).strip(),str(kind).strip())

def canonical_combo(kind,numbers):
    kind=canonical_bet_type(kind)
    nums=[int(x) for x in numbers]
    if kind in UNORDERED_TYPES:
        nums=sorted(nums)
    return "-".join(str(x) for x in nums)

def ticket_numbers(ticket_text):
    out=[]
    for part in str(ticket_text).split(" - "):
        m=re.match(r"\s*(\d+)",part)
        if m:
            out.append(int(m.group(1)))
    return out

def plan_to_json(plan):
    if plan is None or len(plan)==0:
        return "[]"
    rows=[]
    for _,r in plan.iterrows():
        q={}
        for c in plan.columns:
            v=r[c]
            if isinstance(v,np.integer):
                v=int(v)
            elif isinstance(v,np.floating):
                v=None if pd.isna(v) else float(v)
            elif pd.isna(v):
                v=None
            elif c=="金額":
                v=int(v)
            else:
                v=str(v)
            q[c]=v
        rows.append(q)
    return json.dumps(rows,ensure_ascii=False,separators=(",",":"))

def json_to_plan(plan_json):
    try:
        obj=json.loads(str(plan_json or "[]"))
        return obj if isinstance(obj,list) else []
    except Exception:
        return []

def parse_jra_payouts_html(html):
    soup=BeautifulSoup(html or "","lxml")
    text=re.sub(r"\s+"," ",soup.get_text(" ",strip=True))
    pos=text.find("払戻金")
    if pos<0:
        return {"payouts":{},"refund_horses":[],"available_types":[],"raw_count":0}

    section=text[pos+len("払戻金"):]
    ends=[
        section.find(x) for x in
        ("勝馬の紹介","競走中の出来事等","開催選択へ戻る","レース選択へ戻る")
        if section.find(x)>=0
    ]
    if ends:
        section=section[:min(ends)]

    label_pat=re.compile(
        r"(?<!\S)(単勝|複勝|枠連|ワイド|馬連|馬単|3連複|3連単|三連複|三連単|返還)(?!\S)"
    )
    matches=list(label_pat.finditer(section))
    payouts={}
    available=[]
    refund_horses=[]

    for i,m in enumerate(matches):
        label=m.group(1)
        body=section[m.end():matches[i+1].start() if i+1<len(matches) else len(section)]
        if label=="返還":
            mm=re.search(r"返還馬番\s*(.*?)(?:返還同枠|$)",body)
            if mm:
                refund_horses=sorted({
                    int(x) for x in re.findall(r"(\d+)\s*番",mm.group(1))
                })
            continue

        kind=canonical_bet_type(label)
        available.append(kind)
        for combo,amt in re.findall(
            r"(?<!\d)(\d+(?:\s*[-－]\s*\d+){0,2})\s*([0-9,]+)\s*円",
            body
        ):
            nums=[int(x) for x in re.findall(r"\d+",combo)]
            payouts[(kind,canonical_combo(kind,nums))]=int(amt.replace(",",""))

    return {
        "payouts":payouts,
        "refund_horses":refund_horses,
        "available_types":sorted(set(available)),
        "raw_count":len(payouts),
    }

def settle_plan(plan_json,payout_info):
    plan=json_to_plan(plan_json)
    if not plan:
        return {
            "status":"買い目なし","stake":0,"payout":0,"profit":0,"roi":np.nan,
            "hit_count":0,"ticket_count":0,"refund_count":0,"tickets":[],
        }

    payouts=(payout_info or {}).get("payouts") or {}
    refund_horses=set((payout_info or {}).get("refund_horses") or [])
    available=set((payout_info or {}).get("available_types") or [])

    stake_total=0
    payout_total=0
    hit_count=0
    refund_count=0
    incomplete=False
    rows=[]

    for t in plan:
        kind=canonical_bet_type(t.get("券種",""))
        amount=int(float(t.get("金額",0) or 0))
        nums=ticket_numbers(t.get("買い目",""))
        stake_total+=amount

        if not nums or amount<=0:
            rows.append({
                "券種":kind,"買い目":t.get("買い目",""),
                "購入額":amount,"払戻額":0,"結果":"形式不明"
            })
            incomplete=True
            continue

        # Current AI does not bet 枠連. For horse-number tickets, a returned
        # horse means the ticket stake is returned.
        if refund_horses.intersection(nums):
            payout_total+=amount
            refund_count+=1
            rows.append({
                "券種":kind,"買い目":t.get("買い目",""),
                "購入額":amount,"払戻額":amount,"結果":"返還"
            })
            continue

        if kind not in available:
            rows.append({
                "券種":kind,"買い目":t.get("買い目",""),
                "購入額":amount,"払戻額":0,"結果":"払戻未取得"
            })
            incomplete=True
            continue

        key=canonical_combo(kind,nums)
        payout100=int(payouts.get((kind,key),0) or 0)
        pay=int(round((amount/100.0)*payout100)) if payout100 else 0
        payout_total+=pay
        if pay>0:
            hit_count+=1
        rows.append({
            "券種":kind,"買い目":t.get("買い目",""),
            "購入額":amount,"払戻額":pay,
            "結果":"的中" if pay>0 else "不的中"
        })

    profit=payout_total-stake_total
    roi=(payout_total/stake_total*100.0) if stake_total>0 and not incomplete else np.nan
    return {
        "status":"払戻未取得" if incomplete else "確定",
        "stake":int(stake_total),"payout":int(payout_total),"profit":int(profit),
        "roi":roi,"hit_count":int(hit_count),"ticket_count":int(len(plan)),
        "refund_count":int(refund_count),"tickets":rows,
    }

def ticket_results_to_json(rows):
    return json.dumps(rows or [],ensure_ascii=False,separators=(",",":"))

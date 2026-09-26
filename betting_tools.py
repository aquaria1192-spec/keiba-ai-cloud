from itertools import combinations
import math
import pandas as pd
import numpy as np

STYLE_LABELS = {
    "堅実": "的中寄り",
    "標準": "バランス",
    "攻め": "高配当寄り",
}

def _num(v, default=np.nan):
    try:
        x=float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default

def _horse_text(row):
    no = row.get("horse_no","")
    try:
        no = int(float(no))
    except Exception:
        pass
    name = str(row.get("horse_name","")).strip()
    return f"{no} {name}".strip()

def _round100(x):
    return max(100, int(round(float(x)/100.0))*100)

def _allocate_budget(items, budget_yen):
    """
    items: [{weight, ...}]
    Allocate in 100-yen units. If budget is too small, lower-priority items
    are dropped from the end.
    """
    budget = max(100, int(budget_yen)//100*100)
    if not items:
        return []

    # Preserve priority order and ensure at least 100 yen per selected ticket.
    max_tickets = max(1, budget//100)
    items = items[:max_tickets]
    if not items:
        return []

    weights = np.array([max(float(x.get("weight",1)), 0.01) for x in items], dtype=float)
    weights = weights / weights.sum()

    units_total = budget // 100
    units = np.ones(len(items), dtype=int)
    remaining = units_total - len(items)
    if remaining > 0:
        raw = weights * remaining
        add = np.floor(raw).astype(int)
        units += add
        left = remaining - int(add.sum())
        if left > 0:
            frac = raw - np.floor(raw)
            order = np.argsort(-frac)
            for i in order[:left]:
                units[i] += 1

    out=[]
    for item,u in zip(items,units):
        q=dict(item)
        q["金額"]=int(u*100)
        q.pop("weight",None)
        out.append(q)
    return out

def _confidence(g):
    if g is None or len(g)==0:
        return "低"
    s=g.sort_values("順位")
    t=s.iloc[0]
    wp=_num(t.get("win_prob"),0)
    tp=_num(t.get("top3_prob"),0)
    ai=_num(t.get("ai_index"),0)
    gap=0
    if len(s)>=2:
        gap=ai-_num(s.iloc[1].get("ai_index"),0)
    score=0
    if wp>=0.20: score+=2
    elif wp>=0.14: score+=1
    if tp>=0.55: score+=2
    elif tp>=0.40: score+=1
    if gap>=8: score+=2
    elif gap>=4: score+=1
    return "高" if score>=5 else ("中" if score>=3 else "低")

def _ticket(kind, horses, reason, weight):
    return {
        "券種":kind,
        "買い目":" - ".join(horses),
        "狙い":reason,
        "weight":float(weight),
    }

def race_bet_plan(race_detail, style="標準", budget_yen=2000):
    """
    Create a concrete 100-yen-unit example ticket plan for one race.

    The plan is based on AI ranking/marks and, when available, win expected value.
    It does not attempt to estimate exact place/quinella/trio payouts because
    those pool odds are not available in the input data.
    """
    if race_detail is None or len(race_detail)==0:
        return pd.DataFrame(), {"信頼度":"低","コメント":"予想データがありません。"}

    g=race_detail.copy().sort_values(["順位","ai_index"], ascending=[True,False]).reset_index(drop=True)
    style = style if style in STYLE_LABELS else "標準"
    budget_yen=max(100,int(budget_yen)//100*100)

    top=[_horse_text(r) for _,r in g.head(8).iterrows()]
    conf=_confidence(g)
    t0=g.iloc[0]
    top_ev=_num(t0.get("expected_value"))
    top_odds=_num(t0.get("odds"))
    top_wp=_num(t0.get("win_prob"),0)
    odds_ready=pd.notna(top_odds)

    items=[]

    if style=="堅実":
        if len(top)>=1:
            items.append(_ticket("複勝",[top[0]],"◎の3着内を重視",4.0))
        if len(top)>=2:
            items.append(_ticket("ワイド",[top[0],top[1]],"◎－○の上位評価同士",3.0))
            items.append(_ticket("馬連",[top[0],top[1]],"◎－○の本線",1.5))
        if len(top)>=3:
            items.append(_ticket("ワイド",[top[0],top[2]],"◎－▲の押さえ",1.5))

    elif style=="攻め":
        if len(top)>=1:
            items.append(_ticket("単勝",[top[0]],"◎の勝ち切りを狙う",2.0))
        if len(top)>=3:
            items.append(_ticket("馬連",[top[0],top[1]],"◎－○本線",1.0))
            items.append(_ticket("馬連",[top[0],top[2]],"◎－▲",1.0))
        partners=top[1:min(len(top),6)]
        # ◎1頭軸の三連複。上位相手から最大6点。
        for a,b in list(combinations(partners,2))[:6]:
            items.append(_ticket("三連複",[top[0],a,b],"◎1頭軸・上位相手",1.0))

    else:  # 標準
        if len(top)>=1:
            # Expected value >= 1 makes the win ticket slightly heavier.
            w=3.0 if pd.notna(top_ev) and top_ev>=1.0 else 2.0
            items.append(_ticket("単勝",[top[0]],"◎の勝ち切り"+("・期待値重視" if w>=3 else ""),w))
        if len(top)>=2:
            items.append(_ticket("馬連",[top[0],top[1]],"◎－○の本線",2.0))
            items.append(_ticket("ワイド",[top[0],top[1]],"◎－○の安定側",1.5))
        if len(top)>=3:
            items.append(_ticket("馬連",[top[0],top[2]],"◎－▲の押さえ",1.2))
            items.append(_ticket("ワイド",[top[0],top[2]],"◎－▲",1.0))
            items.append(_ticket("三連複",[top[0],top[1],top[2]],"◎○▲の中心",1.5))
        if len(top)>=4:
            items.append(_ticket("三連複",[top[0],top[1],top[3]],"◎○－△",1.0))
        if len(top)>=5:
            items.append(_ticket("三連複",[top[0],top[2],top[4]],"◎▲－△",0.8))

    allocated=_allocate_budget(items,budget_yen)
    plan=pd.DataFrame(allocated)
    if len(plan):
        plan.insert(0,"スタイル",style)
        plan["予算内比率"]=plan["金額"]/plan["金額"].sum()
        plan["予算内比率"]=plan["予算内比率"].map(lambda x:f"{x*100:.0f}%")

    comments=[]
    if conf=="低":
        comments.append("上位差が小さいため、予算を抑えるか見送りも候補")
    elif conf=="高":
        comments.append("上位評価が比較的はっきり")
    else:
        comments.append("上位評価は中程度")
    if not odds_ready:
        comments.append("単勝オッズ未取得のため金額配分は暫定")
    elif pd.notna(top_ev):
        comments.append(f"◎単勝期待値 {top_ev:.2f}")
    if top_wp>0:
        comments.append(f"◎勝率 {top_wp*100:.1f}%")

    meta={
        "信頼度":conf,
        "コメント":"／".join(comments),
        "予算":int(plan["金額"].sum()) if len(plan) else 0,
        "スタイル":style,
    }
    return plan,meta

def all_race_bet_plans(detail, style="標準", budget_yen=2000):
    if detail is None or len(detail)==0:
        return pd.DataFrame()

    rows=[]
    keys=["開催日","競馬場","レース","レース表示"]
    for vals,g in detail.groupby(keys,sort=False,dropna=False):
        plan,meta=race_bet_plan(g,style=style,budget_yen=budget_yen)
        if plan.empty:
            continue
        for _,r in plan.iterrows():
            q={k:v for k,v in zip(keys,vals)}
            q.update({
                "信頼度":meta["信頼度"],
                "コメント":meta["コメント"],
                "券種":r["券種"],
                "買い目":r["買い目"],
                "金額":int(r["金額"]),
                "狙い":r["狙い"],
                "予算内比率":r["予算内比率"],
            })
            rows.append(q)
    return pd.DataFrame(rows)

def mark_legend():
    return "◎本命　○対抗　▲単穴　△連下　☆穴　注注意　×低評価"

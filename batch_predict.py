import pandas as pd
import numpy as np
from model_compare import ensemble_predict

def mark_for_rank(rank, field_size):
    """
    Every runner receives a mark.
      1: ◎ 本命
      2: ○ 対抗
      3: ▲ 単穴
      4-5: △ 連下
      6-7: ☆ 穴
      8-10: 注 注意
      11-: × 低評価
    Small fields naturally stop before the later marks.
    """
    r = int(rank)
    if r == 1:
        return "◎"
    if r == 2:
        return "○"
    if r == 3:
        return "▲"
    if r <= 5:
        return "△"
    if r <= 7:
        return "☆"
    if r <= 10:
        return "注"
    return "×"

def mark_label(mark):
    return {
        "◎":"本命",
        "○":"対抗",
        "▲":"単穴",
        "△":"連下",
        "☆":"穴",
        "注":"注意",
        "×":"低評価",
    }.get(str(mark), "")

def race_display(row):
    rno = str(row.get("race_no",""))
    name = str(row.get("race_name","")).strip()
    if name and name.lower() != "nan":
        return f"{rno} {name}"
    try:
        distance = int(float(row.get("distance",0)))
    except Exception:
        distance = 0
    return f'{rno} {row.get("surface","")}：{distance}m'

def batch_predict_day(entries_features, date_value, trained_models, weights):
    e = entries_features.copy()
    e["date"] = e["date"].astype(str)
    target = e[e["date"] == str(date_value)].copy()
    if target.empty:
        raise ValueError("指定日の出走データがありません。")
    if "race_no" not in target.columns:
        raise ValueError("出走表に race_no（レース番号）がありません。")

    all_rows = []
    summary_rows = []

    group_cols = ["course","race_no"]
    for (course,race_no), g in target.groupby(group_cols, sort=False):
        p = ensemble_predict(g, trained_models, weights).copy()
        p["順位"] = range(1, len(p)+1)
        p["印"] = [mark_for_rank(r, len(p)) for r in p["順位"]]
        p["評価"] = p["印"].map(mark_label)

        first = g.iloc[0]
        label = race_display(first)
        p["開催日"] = str(date_value)
        p["競馬場"] = str(course)
        p["レース"] = str(race_no)
        p["レース表示"] = label
        all_rows.append(p)

        top = p.iloc[0]
        second = p.iloc[1] if len(p) >= 2 else None
        ai_gap = (
            float(top.get("ai_index",0)) - float(second.get("ai_index",0))
            if second is not None else 0.0
        )
        value = p[
            (pd.to_numeric(p["expected_value"],errors="coerce") >= 1.0) &
            (pd.to_numeric(p["value_gap"],errors="coerce") > 0)
        ]
        summary_rows.append({
            "開催日": str(date_value),
            "競馬場": str(course),
            "レース": str(race_no),
            "レース表示": label,
            "頭数": len(p),
            "本命馬番": top.get("horse_no",""),
            "本命馬": top.get("horse_name",""),
            "本命勝率": top.get("win_prob",np.nan),
            "本命3着内率": top.get("top3_prob",np.nan),
            "本命AI指数": top.get("ai_index",np.nan),
            "本命単勝オッズ": top.get("odds",np.nan),
            "本命AI期待値": top.get("expected_value",np.nan),
            "AI指数差": ai_gap,
            "期待値1以上頭数": len(value),
        })

    detail = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return detail, summary

def high_value_candidates(detail, min_ev=1.0, min_win_prob=0.0):
    if detail is None or detail.empty:
        return pd.DataFrame()
    d = detail.copy()
    ev = pd.to_numeric(d["expected_value"], errors="coerce")
    wp = pd.to_numeric(d["win_prob"], errors="coerce")
    gap = pd.to_numeric(d["value_gap"], errors="coerce")
    out = d[(ev >= float(min_ev)) & (wp >= float(min_win_prob)) & (gap > 0)].copy()
    return out.sort_values(
        ["expected_value","win_prob"],
        ascending=[False,False]
    ).reset_index(drop=True)

def top_pick_table(detail):
    if detail is None or detail.empty:
        return pd.DataFrame()
    return detail[pd.to_numeric(detail["順位"],errors="coerce")==1].copy()

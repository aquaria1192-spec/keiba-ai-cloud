import math
import pandas as pd
import numpy as np
from ml_engine import train_models, predict, available_models, model_label

# Ver.1.12 probability calibration.
# Parameters were fitted on the 2024 proxy-year holdout only and then frozen.
# The untouched 2025 proxy-year holdout is used as the adoption check.
WIN_PROB_POWER = 2.0654155666646536
TOP3_LOGIT_SLOPE = 1.0117893511357146
TOP3_LOGIT_INTERCEPT = -1.2759957435674174
CALIBRATION_VERSION = "proxy2024-platt-v1"

def _calibrate_histgb_probabilities(win_prob, top3_prob):
    p1=pd.to_numeric(win_prob,errors="coerce").fillna(0).clip(1e-9,1.0)
    p1=np.power(p1,WIN_PROB_POWER)
    total=float(p1.sum())
    p1=p1/total if total>0 else pd.Series(
        np.full(len(p1),1/max(len(p1),1)),index=p1.index
    )

    p3=pd.to_numeric(top3_prob,errors="coerce").fillna(0.0).clip(1e-6,1-1e-6)
    logit=np.log(p3/(1-p3))
    z=np.clip(TOP3_LOGIT_SLOPE*logit+TOP3_LOGIT_INTERCEPT,-40,40)
    p3=1/(1+np.exp(-z))
    p3=np.maximum(p3,p1)
    return pd.Series(p1,index=win_prob.index),pd.Series(p3,index=top3_prob.index)

def compare_models(history, model_keys):
    rows = []
    trained = {}
    for key in model_keys:
        if not available_models().get(key, False):
            continue
        try:
            wm, tm, metrics = train_models(history, model_type=key)
            trained[key] = (wm, tm, metrics)
            rows.append({
                "model_type": key,
                "model_name": model_label(key),
                "win_auc": metrics.get("win_auc"),
                "top3_auc": metrics.get("top3_auc"),
                "win_logloss": metrics.get("win_logloss"),
                "top3_logloss": metrics.get("top3_logloss"),
                "win_brier": metrics.get("win_brier"),
                "top3_brier": metrics.get("top3_brier"),
                "rows": metrics.get("rows"),
                "valid_rows": metrics.get("valid_rows"),
            })
        except Exception as e:
            rows.append({
                "model_type": key,
                "model_name": model_label(key),
                "error": str(e),
            })
    return pd.DataFrame(rows), trained

def quality_weights(compare_df, selected_models=None):
    """
    Build conservative validation-based weights.
    Uses inverse win logloss and top3 logloss. If unavailable, equal weight.
    """
    if compare_df is None or len(compare_df) == 0:
        return {}
    df = compare_df.copy()
    if selected_models is not None:
        df = df[df["model_type"].isin(selected_models)]
    if len(df) == 0:
        return {}

    scores = {}
    for _, r in df.iterrows():
        key = r["model_type"]
        wl = pd.to_numeric(pd.Series([r.get("win_logloss")]), errors="coerce").iloc[0]
        tl = pd.to_numeric(pd.Series([r.get("top3_logloss")]), errors="coerce").iloc[0]
        if pd.notna(wl) and pd.notna(tl) and wl > 0 and tl > 0:
            scores[key] = 1.0 / (0.65 * wl + 0.35 * tl)
        else:
            scores[key] = 1.0

    s = sum(scores.values())
    if s <= 0:
        n = len(scores)
        return {k: 1/n for k in scores}
    return {k: v/s for k,v in scores.items()}

def equal_weights(selected_models):
    models = list(selected_models)
    if not models:
        return {}
    w = 1.0 / len(models)
    return {k:w for k in models}

def ensemble_predict(entries, trained_models, weights):
    if not weights:
        raise ValueError("アンサンブル対象モデルがありません。")

    preds = {}
    for key, weight in weights.items():
        if key not in trained_models:
            continue
        wm, tm, _ = trained_models[key]
        p = predict(entries, wm, tm).copy()
        # Re-align using horse identity, because predict() sorts by AI index.
        p["_merge_key"] = (
            p["horse_no"].astype(str) + "|" +
            p["horse_name"].astype(str)
        )
        preds[key] = p

    if not preds:
        raise ValueError("学習済みモデルがありません。")

    base_key = next(iter(preds))
    base = preds[base_key].copy()
    base = base.drop(columns=[
        c for c in ["win_prob","top3_prob","ai_index","expected_value","value_gap","win_prob_raw"]
        if c in base.columns
    ])

    merged = base.copy()
    merged["win_prob"] = 0.0
    merged["top3_prob"] = 0.0

    used_weight = 0.0
    used_models = []
    for key, p in preds.items():
        w = float(weights.get(key, 0))
        if w <= 0:
            continue
        q = p[["_merge_key","win_prob","top3_prob"]].copy()
        q = q.rename(columns={
            "win_prob": f"win_prob_{key}",
            "top3_prob": f"top3_prob_{key}"
        })
        merged = merged.merge(q, on="_merge_key", how="left")
        merged["win_prob"] += merged[f"win_prob_{key}"].fillna(0) * w
        merged["top3_prob"] += merged[f"top3_prob_{key}"].fillna(0) * w
        used_weight += w
        used_models.append(key)

    if used_weight <= 0:
        raise ValueError("有効な重みがありません。")
    merged["win_prob"] /= used_weight
    merged["top3_prob"] /= used_weight

    # Normalize win probability across race.
    total = merged["win_prob"].sum()
    if total > 0:
        merged["win_prob"] = merged["win_prob"] / total

    # The static calibration was validated for the production HistGB-only
    # package. Do not silently apply it to a future multi-model ensemble.
    if len(used_models)==1 and used_models[0]=="histgb":
        merged["win_prob"],merged["top3_prob"]=_calibrate_histgb_probabilities(
            merged["win_prob"],merged["top3_prob"]
        )
        merged["probability_calibration"]=CALIBRATION_VERSION
    else:
        merged["top3_prob"] = np.maximum(
            merged["top3_prob"], merged["win_prob"]
        )
        merged["probability_calibration"]="none"
    max_win = max(float(merged["win_prob"].max()), 1e-9)
    recent = pd.to_numeric(merged.get("recent_top3_rate"), errors="coerce").fillna(0).clip(0,1)
    merged["ai_index"] = np.clip(
        100 * (
            0.65 * (merged["win_prob"] / max_win) +
            0.25 * merged["top3_prob"] +
            0.10 * recent
        ),
        0, 100
    )
    # Validation showed that selecting ◎ primarily by race-normalized win
    # probability is more accurate than the legacy composite AI-index order.
    merged["prediction_score"] = merged["win_prob"]

    odds = pd.to_numeric(merged.get("odds"), errors="coerce")
    implied = pd.to_numeric(merged.get("implied_prob"), errors="coerce").fillna(0)
    merged["expected_value"] = merged["win_prob"] * odds
    merged["value_gap"] = merged["win_prob"] - implied

    return merged.sort_values(
        ["prediction_score","top3_prob","ai_index"], ascending=False
    ).reset_index(drop=True)

def ensemble_label(weights):
    parts = []
    for k,v in weights.items():
        parts.append(f"{model_label(k)} {v*100:.0f}%")
    return " + ".join(parts)

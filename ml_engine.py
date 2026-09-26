import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import importlib.util

NUM_COLS = [
    "distance","horse_no","gate","age","weight","body_weight","body_weight_diff",
    "recent_avg_finish","recent_top3_rate","recent_avg_margin","recent_avg_last3f_rank",
    "recent_avg_popularity","distance_top3_rate","course_top3_rate","surface_top3_rate",
    "jockey_top3_rate","trainer_top3_rate","gate_condition_top3_rate",
    "prev_finish","prev_margin","prev_last3f_rank","distance_change","days_since_last",
    "body_weight_change_from_prev","class_change","race_class_score","running_style_score",
    "odds","popularity","implied_prob","log_odds"
]
CAT_COLS = ["course","surface","going","jockey","trainer","sex","running_style","race_class"]
FEATURE_COLS = NUM_COLS + CAT_COLS

MODEL_LABELS = {
    "logistic":"ロジスティック回帰（軽量）",
    "histgb":"ヒストグラム勾配ブースティング（標準・推奨）",
    "lightgbm":"LightGBM（追加AI）",
    "xgboost":"XGBoost（追加AI）",
}

def available_models():
    return {
        "logistic": True,
        "histgb": True,
        "lightgbm": importlib.util.find_spec("lightgbm") is not None,
        "xgboost": importlib.util.find_spec("xgboost") is not None,
    }

def model_label(key):
    return MODEL_LABELS.get(key,key)

def _build_estimator(model_type):
    if model_type == "logistic":
        return LogisticRegression(
            max_iter=2500,class_weight="balanced",solver="liblinear",random_state=42
        ), True
    if model_type == "histgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.06,max_iter=140,max_leaf_nodes=31,
            l2_regularization=0.5,early_stopping=True,
            class_weight="balanced",random_state=42
        ), False
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except Exception as e:
            raise RuntimeError("LightGBMがインストールされていません。AI強化インストール.batを実行してください。") from e
        return LGBMClassifier(
            n_estimators=350,learning_rate=0.04,num_leaves=31,
            subsample=0.9,colsample_bytree=0.9,class_weight="balanced",
            random_state=42,n_jobs=-1,verbosity=-1
        ), True
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise RuntimeError("XGBoostがインストールされていません。AI強化インストール.batを実行してください。") from e
        return XGBClassifier(
            n_estimators=350,learning_rate=0.04,max_depth=4,
            min_child_weight=2,subsample=0.9,colsample_bytree=0.9,
            reg_lambda=1.0,objective="binary:logistic",
            eval_metric="logloss",random_state=42,n_jobs=-1
        ), True
    raise ValueError(f"未対応モデルです: {model_type}")

def _pipeline(model_type):
    estimator, sparse_ok = _build_estimator(model_type)
    num_steps=[("imputer",SimpleImputer(strategy="median"))]
    if model_type=="logistic":
        num_steps.append(("scale",StandardScaler()))
    num=Pipeline(num_steps)
    if model_type=="histgb":
        cat=Pipeline([
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("ordinal",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1,encoded_missing_value=-1))
        ])
        prep=ColumnTransformer([("num",num,NUM_COLS),("cat",cat,CAT_COLS)],sparse_threshold=0.0)
    else:
        cat=Pipeline([
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("onehot",OneHotEncoder(handle_unknown="ignore",min_frequency=2,sparse_output=sparse_ok))
        ])
        prep=ColumnTransformer([("num",num,NUM_COLS),("cat",cat,CAT_COLS)],sparse_threshold=0.3 if sparse_ok else 0.0)
    return Pipeline([("prep",prep),("model",estimator)])

def prepare_frame(df):
    out = df.copy()
    for c in NUM_COLS:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c],errors="coerce")
    for c in CAT_COLS:
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].fillna("").astype(str)
    return out

def train_models(history, model_type="histgb"):
    if model_type not in available_models():
        raise ValueError(f"未対応モデルです: {model_type}")
    if not available_models()[model_type]:
        raise RuntimeError(f"{model_label(model_type)} は未インストールです。")

    h = prepare_frame(history)
    if "finish" not in h.columns:
        raise ValueError("履歴データに finish（着順）が必要です。")
    h["finish"] = pd.to_numeric(h["finish"],errors="coerce")
    h = h[h["finish"].notna()].copy()
    if len(h)<100:
        raise ValueError("学習には100行以上の履歴データを推奨します。")

    if "date" in h.columns:
        h["_dt"] = pd.to_datetime(h["date"],errors="coerce")
        h = h.sort_values("_dt").drop(columns="_dt")

    # Ver.1.1 large-data path: train production models once on all rows.
    # This avoids holding validation and production transforms in memory at the same time.
    if len(h) >= 200000 and model_type == "histgb":
        y1 = (h["finish"]==1).astype(int)
        y3 = (h["finish"]<=3).astype(int)
        wm = _pipeline(model_type)
        tm = _pipeline(model_type)
        wm.fit(h[FEATURE_COLS], y1)
        tm.fit(h[FEATURE_COLS], y3)
        metrics = {
            "model_type":model_type,"rows":len(h),"train_rows":len(h),"valid_rows":0,
            "win_auc":None,"top3_auc":None,"win_logloss":None,"top3_logloss":None,
            "win_brier":None,"top3_brier":None,
        }
        return wm,tm,metrics

    cut = max(1,int(len(h)*0.8))
    train,valid = h.iloc[:cut].copy(),h.iloc[cut:].copy()
    if len(valid)<20:
        train,valid = h.iloc[:-20].copy(),h.iloc[-20:].copy()

    y1tr = (train["finish"]==1).astype(int)
    y3tr = (train["finish"]<=3).astype(int)
    if y1tr.nunique()<2 or y3tr.nunique()<2:
        raise ValueError("学習区間に1着/非1着、3着内/圏外の両方が必要です。")

    wm = _pipeline(model_type)
    tm = _pipeline(model_type)
    wm.fit(train[FEATURE_COLS],y1tr)
    tm.fit(train[FEATURE_COLS],y3tr)

    p1 = wm.predict_proba(valid[FEATURE_COLS])[:,1]
    p3 = tm.predict_proba(valid[FEATURE_COLS])[:,1]
    y1 = (valid["finish"]==1).astype(int)
    y3 = (valid["finish"]<=3).astype(int)

    metrics = {
        "model_type":model_type,
        "rows":len(h),"train_rows":len(train),"valid_rows":len(valid),
        "win_auc":float(roc_auc_score(y1,p1)) if y1.nunique()>1 else None,
        "top3_auc":float(roc_auc_score(y3,p3)) if y3.nunique()>1 else None,
        "win_logloss":float(log_loss(y1,p1,labels=[0,1])),
        "top3_logloss":float(log_loss(y3,p3,labels=[0,1])),
        "win_brier":float(brier_score_loss(y1,p1)),
        "top3_brier":float(brier_score_loss(y3,p3)),
    }

    # Production models use all history.
    wm.fit(h[FEATURE_COLS],(h["finish"]==1).astype(int))
    tm.fit(h[FEATURE_COLS],(h["finish"]<=3).astype(int))
    return wm,tm,metrics

def predict(entries,win_model,top3_model):
    e = prepare_frame(entries)
    e["win_prob_raw"] = win_model.predict_proba(e[FEATURE_COLS])[:,1]
    e["top3_prob"] = top3_model.predict_proba(e[FEATURE_COLS])[:,1]
    total = e["win_prob_raw"].sum()
    e["win_prob"] = e["win_prob_raw"]/total if total>0 else 1/max(len(e),1)
    max_win = max(float(e["win_prob"].max()),1e-9)
    e["ai_index"] = np.clip(
        100*(0.50*e["top3_prob"] + 0.35*(e["win_prob"]/max_win)
             + 0.15*np.clip(e["recent_top3_rate"],0,1)),0,100
    )
    e["expected_value"] = e["win_prob"]*pd.to_numeric(e["odds"],errors="coerce")
    e["value_gap"] = e["win_prob"]-pd.to_numeric(e["implied_prob"],errors="coerce").fillna(0)
    return e.sort_values(["ai_index","win_prob"],ascending=False).reset_index(drop=True)

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import importlib.util

BASE_NUM_COLS = [
    "distance","horse_no","gate","age","weight","body_weight","body_weight_diff",
    "recent_avg_finish","recent_top3_rate","recent_avg_margin","recent_avg_last3f_rank",
    "recent_avg_popularity","distance_top3_rate","course_top3_rate","surface_top3_rate",
    "jockey_top3_rate","trainer_top3_rate","gate_condition_top3_rate",
    "prev_finish","prev_margin","prev_last3f_rank","distance_change","days_since_last",
    "body_weight_change_from_prev","class_change","race_class_score","running_style_score",
    "odds","popularity","implied_prob","log_odds"
]
ENHANCED_NUM_COLS = [
    "horse_going_top3_rate",
    "jockey_course_top3_rate","jockey_surface_top3_rate","jockey_distance_top3_rate",
    "jockey_going_top3_rate","jockey_trainer_top3_rate",
    "trainer_course_top3_rate","trainer_surface_top3_rate","trainer_distance_top3_rate",
    "field_size","market_prob_norm","market_rank_pct","popularity_pct",
    "recent_top3_rate_rank_pct","jockey_top3_rate_rank_pct","trainer_top3_rate_rank_pct",
    "distance_top3_rate_rank_pct","course_top3_rate_rank_pct","surface_top3_rate_rank_pct",
]
NUM_COLS = BASE_NUM_COLS + ENHANCED_NUM_COLS
# Raw jockey/trainer names are intentionally excluded from the HistGB categorical
# input. Their performance is represented by leakage-safe numeric rates instead;
# this avoids imposing a meaningless ordinal relationship on names.
CAT_COLS = ["course","surface","going","sex","running_style","race_class"]
FEATURE_COLS = NUM_COLS + CAT_COLS

MODEL_LABELS = {
    "logistic":"ロジスティック回帰（軽量）",
    "histgb":"ヒストグラム勾配ブースティング（精度強化）",
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
def model_label(key): return MODEL_LABELS.get(key,key)

def _build_estimator(model_type):
    if model_type == "logistic":
        return LogisticRegression(max_iter=2500,class_weight="balanced",solver="liblinear",random_state=42), True
    if model_type == "histgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.06,max_iter=160,max_leaf_nodes=31,
            l2_regularization=0.5,early_stopping=True,
            class_weight="balanced",random_state=42
        ), False
    if model_type == "lightgbm":
        try: from lightgbm import LGBMClassifier
        except Exception as e: raise RuntimeError("LightGBMがインストールされていません。") from e
        return LGBMClassifier(n_estimators=350,learning_rate=0.04,num_leaves=31,subsample=0.9,colsample_bytree=0.9,class_weight="balanced",random_state=42,n_jobs=-1,verbosity=-1), True
    if model_type == "xgboost":
        try: from xgboost import XGBClassifier
        except Exception as e: raise RuntimeError("XGBoostがインストールされていません。") from e
        return XGBClassifier(n_estimators=350,learning_rate=0.04,max_depth=4,min_child_weight=2,subsample=0.9,colsample_bytree=0.9,reg_lambda=1.0,objective="binary:logistic",eval_metric="logloss",random_state=42,n_jobs=-1), True
    raise ValueError(f"未対応モデルです: {model_type}")

def _pipeline(model_type):
    estimator, _ = _build_estimator(model_type)
    num_steps=[("imputer",SimpleImputer(strategy="median"))]
    if model_type=="logistic": num_steps.append(("scale",StandardScaler()))
    num=Pipeline(num_steps)
    cat=Pipeline([
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("ordinal",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1,encoded_missing_value=-1))
    ])
    prep=ColumnTransformer([("num",num,NUM_COLS),("cat",cat,CAT_COLS)],sparse_threshold=0.0)
    return Pipeline([("prep",prep),("model",estimator)])

def prepare_frame(df):
    out=df.copy()
    for c in NUM_COLS:
        if c not in out.columns: out[c]=np.nan
        out[c]=pd.to_numeric(out[c],errors="coerce")
    for c in CAT_COLS:
        if c not in out.columns: out[c]=""
        out[c]=out[c].fillna("").astype(str)
    return out

def train_models(history, model_type="histgb"):
    h=prepare_frame(history)
    if "finish" not in h.columns: h["finish"]=pd.to_numeric(history.get("finish"),errors="coerce")
    else: h["finish"]=pd.to_numeric(h["finish"],errors="coerce")
    h=h[h["finish"].notna()].copy()
    if len(h)<100: raise ValueError("学習には100行以上の履歴データを推奨します。")
    if "date" in history.columns:
        h["_dt"]=pd.to_datetime(history.loc[h.index,"date"],errors="coerce")
        h=h.sort_values("_dt").drop(columns="_dt")
    if len(h)>=200000 and model_type=="histgb":
        wm=_pipeline(model_type); tm=_pipeline(model_type)
        wm.fit(h[FEATURE_COLS],(h["finish"]==1).astype(int))
        tm.fit(h[FEATURE_COLS],(h["finish"]<=3).astype(int))
        return wm,tm,{"model_type":model_type,"rows":len(h),"train_rows":len(h),"valid_rows":0}
    cut=max(1,int(len(h)*0.8)); train=h.iloc[:cut]; valid=h.iloc[cut:]
    wm=_pipeline(model_type); tm=_pipeline(model_type)
    wm.fit(train[FEATURE_COLS],(train["finish"]==1).astype(int))
    tm.fit(train[FEATURE_COLS],(train["finish"]<=3).astype(int))
    p1=wm.predict_proba(valid[FEATURE_COLS])[:,1];p3=tm.predict_proba(valid[FEATURE_COLS])[:,1]
    y1=(valid["finish"]==1).astype(int); y3=(valid["finish"]<=3).astype(int)
    metrics={"model_type":model_type,"rows":len(h),"train_rows":len(train),"valid_rows":len(valid),
      "win_auc":float(roc_auc_score(y1,p1)),"top3_auc":float(roc_auc_score(y3,p3)),
      "win_logloss":float(log_loss(y1,p1,labels=[0,1])),"top3_logloss":float(log_loss(y3,p3,labels=[0,1])),
      "win_brier":float(brier_score_loss(y1,p1)),"top3_brier":float(brier_score_loss(y3,p3))}
    wm.fit(h[FEATURE_COLS],(h["finish"]==1).astype(int));tm.fit(h[FEATURE_COLS],(h["finish"]<=3).astype(int))
    return wm,tm,metrics

def predict(entries,win_model,top3_model):
    e=prepare_frame(entries)
    e["win_prob_raw"]=win_model.predict_proba(e[FEATURE_COLS])[:,1]
    e["top3_prob"]=top3_model.predict_proba(e[FEATURE_COLS])[:,1]
    total=e["win_prob_raw"].sum();e["win_prob"]=e["win_prob_raw"]/total if total>0 else 1/max(len(e),1)
    e["top3_prob"]=np.maximum(e["top3_prob"],e["win_prob"])
    max_win=max(float(e["win_prob"].max()),1e-9)
    recent=pd.to_numeric(e.get("recent_top3_rate"),errors="coerce").fillna(0).clip(0,1)
    e["ai_index"]=np.clip(100*(0.65*(e["win_prob"]/max_win)+0.25*e["top3_prob"]+0.10*recent),0,100)
    e["prediction_score"]=e["win_prob"]
    odds=pd.to_numeric(e.get("odds"),errors="coerce")
    implied=pd.to_numeric(e.get("implied_prob"),errors="coerce").fillna(0)
    e["expected_value"]=e["win_prob"]*odds
    e["value_gap"]=e["win_prob"]-implied
    return e.sort_values(["prediction_score","top3_prob","ai_index"],ascending=False).reset_index(drop=True)

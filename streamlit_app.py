from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import joblib
import pandas as pd
import streamlit as st

from cloud_data_builder import fetch_entries_cloud
from cloud_features import load_feature_store, enrich_entries_cloud
from batch_predict import batch_predict_day
from betting_tools import race_bet_plan, mark_legend

BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE/"data"/"cloud_model.joblib"
JST = ZoneInfo("Asia/Tokyo")

st.set_page_config(
    page_title="競馬予想AI Cloud Ver.1.4",
    page_icon="🏇",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
#MainMenu{visibility:hidden} footer{visibility:hidden}
.block-container{padding-top:.8rem;padding-bottom:2.5rem;max-width:920px}
.hero{padding:1rem;border:1px solid #d7e6dd;border-radius:14px;background:#f7fbf8;margin-bottom:1rem}
.step{font-size:1.25rem;font-weight:750}
button,[role="button"]{min-height:46px}
div[data-baseweb="select"]>div{min-height:46px}
input{font-size:16px!important}
@media(max-width:700px){
 .block-container{padding:.55rem .55rem 2rem!important}
 h1{font-size:1.55rem!important;line-height:1.2!important}
 h2{font-size:1.25rem!important}
 h3{font-size:1.08rem!important}
 .hero{padding:.75rem!important}
 .step{font-size:1.08rem!important}
 div[data-testid="stHorizontalBlock"]{flex-direction:column!important;gap:.4rem!important}
 div[data-testid="stHorizontalBlock"]>div[data-testid="column"]{
   width:100%!important;flex:1 1 100%!important;min-width:0!important}
 div.stButton>button,div.stDownloadButton>button{
   width:100%!important;min-height:50px!important;font-size:1rem!important}
 div[data-testid="stDataFrame"]{max-width:100%!important;overflow-x:auto!important}
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_cloud_assets():
    model = joblib.load(MODEL_FILE)
    store = load_feature_store()
    return model, store

@st.cache_data(ttl=120, show_spinner=False)
def fetch_cached(date_iso: str):
    d = pd.Timestamp(date_iso).date()
    return fetch_entries_cloud(d)

def pct(v):
    return "-" if pd.isna(v) else f"{float(v)*100:.1f}%"

def race_num(v):
    try:
        return int(str(v).upper().replace("R","").strip())
    except Exception:
        return 999

def race_label(race_no, df):
    g = df[df["race_no"].astype(str)==str(race_no)]
    if g.empty:
        return str(race_no)
    r = g.iloc[0]
    name = str(r.get("race_name","")).strip()
    return f"{race_no}　{name}" if name and name.lower()!="nan" else str(race_no)

def predict_race(feature_df, date_iso, models, weights):
    return batch_predict_day(feature_df, date_iso, models, weights)

def show_prediction(detail):
    rg = detail.sort_values("順位").copy()
    label = str(rg.iloc[0].get("レース表示",""))
    st.subheader(label)
    st.markdown("#### 全出走馬の印付き予想")
    st.caption(mark_legend())

    cols = ["順位","印","評価","horse_no","horse_name",
            "win_prob","top3_prob","ai_index","odds","expected_value"]
    q = rg[[c for c in cols if c in rg.columns]].rename(columns={
        "horse_no":"馬番","horse_name":"馬名","win_prob":"勝率",
        "top3_prob":"3着内率","ai_index":"AI指数",
        "odds":"単勝オッズ","expected_value":"AI期待値",
    })
    for c in ["勝率","3着内率"]:
        if c in q:
            q[c] = pd.to_numeric(q[c],errors="coerce").map(pct)
    for c in ["AI指数","単勝オッズ","AI期待値"]:
        if c in q:
            q[c] = pd.to_numeric(q[c],errors="coerce").round(2)
    st.dataframe(q, use_container_width=True, hide_index=True)

    st.markdown("#### 馬券の買い方")
    c1,c2 = st.columns(2)
    with c1:
        style = st.selectbox("買い方",["堅実","標準","攻め"],index=1,key="cloud_bet_style")
    with c2:
        budget = st.number_input(
            "このレースの予算（円）",min_value=500,max_value=50000,
            value=2000,step=100,key="cloud_bet_budget"
        )
    plan, meta = race_bet_plan(rg,style=style,budget_yen=int(budget))
    st.write(f"**AI信頼度：{meta['信頼度']}**　{meta['コメント']}")
    if len(plan):
        st.dataframe(
            plan[["券種","買い目","金額","狙い","予算内比率"]],
            use_container_width=True,hide_index=True
        )
        st.caption(f"合計 {int(plan['金額'].sum()):,}円")
        st.download_button(
            "このレースの買い目CSV",
            plan.to_csv(index=False).encode("utf-8-sig"),
            "買い目.csv","text/csv",use_container_width=True
        )

def default_race_date():
    now = datetime.now(JST)
    if now.weekday() == 5:
        return now.date() if now.hour < 16 else now.date()+timedelta(days=1)
    if now.weekday() == 6:
        return now.date() if now.hour < 16 else now.date()+timedelta(days=6)
    return now.date()+timedelta(days=(5-now.weekday())%7)

st.title("🏇 競馬予想AI Cloud Ver.1.4")
st.caption("PC・スマホ共通／URLを開くだけで利用できます")

st.markdown("""
<div class="hero">
<b>クラウド版はPCの起動操作が不要です。</b><br>
① 出走データを取得<br>
② 開催地 → レースを選択<br>
③ 全頭印付き予想・買い目を確認
</div>
""", unsafe_allow_html=True)

try:
    model_pkg, feature_store = load_cloud_assets()
except Exception as e:
    st.error(f"クラウドAIの読み込みに失敗しました：{e}")
    st.stop()

with st.container(border=True):
    st.markdown('<div class="step">① 出走データを取得</div>',unsafe_allow_html=True)
    target_date = st.date_input("予想する開催日", value=default_race_date(), key="cloud_target_date")
    c1,c2 = st.columns(2)
    with c1:
        get_clicked = st.button("出走データを取得", type="primary", use_container_width=True)
    with c2:
        refresh_clicked = st.button("最新データに更新", use_container_width=True)

    if refresh_clicked:
        fetch_cached.clear()
        get_clicked = True

    if get_clicked:
        try:
            with st.spinner("出走表を取得しています…"):
                entries, info = fetch_cached(target_date.isoformat())
                features = enrich_entries_cloud(entries, feature_store)
            st.session_state["cloud_entries"] = entries
            st.session_state["cloud_features"] = features
            st.session_state["cloud_info"] = info
            st.session_state.pop("cloud_prediction", None)
            st.success(f"{info['races']}レース・{info['rows']}頭を取得しました。")
        except Exception as e:
            st.error(f"出走データを取得できませんでした：{e}")

    info = st.session_state.get("cloud_info")
    if info:
        st.caption(f"取得日：{info['date']}　／　{info['races']}レース　／　{info['rows']}頭")
        if info.get("excluded"):
            with st.expander("AI予想対象外のレース"):
                for rid, reason in info["excluded"].items():
                    st.write(f"・{rid}：{reason}")
        if info.get("errors"):
            with st.expander("取得時の補足"):
                for x in info["errors"][-20:]:
                    st.write("・"+str(x))

features = st.session_state.get("cloud_features")
if features is not None and len(features):
    features = features.copy()
    features["date"] = features["date"].astype(str)
    features["course"] = features["course"].astype(str)
    features["race_no"] = features["race_no"].astype(str)

    date_iso = str(st.session_state.get("cloud_info",{}).get("date",""))
    day = features[features["date"]==date_iso].copy()

    with st.container(border=True):
        st.markdown('<div class="step">② 開催地 → レースを選んで予想</div>',unsafe_allow_html=True)
        courses = sorted(day["course"].dropna().unique())
        course = st.selectbox("① 開催地を選択", courses, key="cloud_course")
        cdf = day[day["course"]==course].copy()
        races = sorted(cdf["race_no"].dropna().unique(), key=race_num)
        race = st.selectbox(
            "② レースを選択", races,
            format_func=lambda x: race_label(x,cdf),
            key="cloud_race"
        )
        rdf = cdf[cdf["race_no"]==race].copy()
        select_key = f"{date_iso}|{course}|{race}"

        if st.session_state.get("cloud_select_key") != select_key:
            try:
                detail, summary = predict_race(
                    rdf, date_iso,
                    model_pkg["trained_models"], model_pkg["weights"]
                )
                st.session_state["cloud_prediction"] = detail
                st.session_state["cloud_select_key"] = select_key
            except Exception as e:
                st.error(f"予想できませんでした：{e}")

        detail = st.session_state.get("cloud_prediction")
        if detail is not None and len(detail):
            show_prediction(detail)

with st.expander("クラウド版について"):
    st.write(
        "URLをスマホのホーム画面に追加すれば、次回からアイコンをタップして開けます。"
        "クラウド側でAIを実行するため、PCを起動しておく必要はありません。"
    )
    st.caption(
        "大規模学習データはクラウド向けに集約済みです。"
        "元学習データの日付はsequence-proxyのため、休養日数系の特徴は近似値です。"
    )

st.caption("AI予想・買い目は参考情報であり、的中や利益を保証するものではありません。")

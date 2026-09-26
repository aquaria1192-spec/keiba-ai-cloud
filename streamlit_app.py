from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from cloud_data_builder import fetch_entries_cloud
from cloud_features import load_feature_store, enrich_entries_cloud
from batch_predict import batch_predict_day
from betting_tools import race_bet_plan, mark_legend
from race_day_context import (
    fetch_day_contexts, apply_official_going,
    fetch_same_day_bias, apply_day_adjustments
)
from evaluation_store import (
    HistoryBackend, save_prediction_if_new, settle_day_snapshots,
    evaluation_metrics, mark_summary, condition_summary,
    calibration_summary, training_candidate_csv
)

BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE/"data"/"cloud_model.joblib"
JST = ZoneInfo("Asia/Tokyo")

st.set_page_config(
    page_title="競馬予想AI Cloud Ver.1.6.2",
    page_icon="🏇",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
#MainMenu{visibility:hidden} footer{visibility:hidden}
.block-container{padding-top:.8rem;padding-bottom:2.5rem;max-width:960px}
.hero{padding:1rem;border:1px solid #d7e6dd;border-radius:14px;background:#f7fbf8;margin-bottom:1rem}
.step{font-size:1.25rem;font-weight:750}
.condition{padding:.7rem .8rem;border:1px solid #e2e8e4;border-radius:12px;margin:.35rem 0}
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
def fetch_entries_cached(date_iso: str):
    return fetch_entries_cloud(pd.Timestamp(date_iso).date())

@st.cache_data(ttl=120, show_spinner=False)
def fetch_context_cached(date_iso: str, courses_tuple):
    return fetch_day_contexts(pd.Timestamp(date_iso).date(), list(courses_tuple))

@st.cache_data(ttl=90, show_spinner=False)
def fetch_bias_cached(date_iso, course, race_no, surface, official_items):
    official=dict(official_items)
    return fetch_same_day_bias(
        pd.Timestamp(date_iso).date(),course,int(race_no),surface,official
    )

def pct(v):
    return "-" if pd.isna(v) else f"{float(v)*100:.1f}%"

def race_num(v):
    try:
        return int(str(v).upper().replace("R","").strip())
    except Exception:
        return 999

def race_label(race_no, df):
    g=df[df["race_no"].astype(str)==str(race_no)]
    if g.empty:
        return str(race_no)
    r=g.iloc[0]
    name=str(r.get("race_name","")).strip()
    return f"{race_no}　{name}" if name and name.lower()!="nan" else str(race_no)

def nfmt(v, suffix="", digits=1):
    x=pd.to_numeric(pd.Series([v]),errors="coerce").iloc[0]
    return "-" if pd.isna(x) else f"{x:.{digits}f}{suffix}"

def show_day_context(contexts):
    if not contexts:
        return
    st.markdown("#### 当日コンディション")
    for course,ctx in contexts.items():
        weather=ctx.get("weather") or {}
        valid=ctx.get("is_target_date",False)
        date_note="" if valid else f"（JRA表示日 {ctx.get('race_date') or '-'}）"
        with st.expander(f"{course}競馬場　当日馬場・天気",expanded=False):
            c1,c2,c3=st.columns(3)
            c1.metric("芝",ctx.get("turf_going") or "-")
            c2.metric("ダート",ctx.get("dirt_going") or "-")
            c3.metric("天気",weather.get("weather_text") or ctx.get("jra_weather") or "-")
            st.write(
                f"**JRA馬場情報** {date_note}　"
                f"含水率 芝：{nfmt(ctx.get('turf_goal'),'%')} / {nfmt(ctx.get('turf_corner'),'%')}　"
                f"ダート：{nfmt(ctx.get('dirt_goal'),'%')} / {nfmt(ctx.get('dirt_corner'),'%')}"
            )
            cushion=ctx.get("cushion")
            if pd.notna(pd.to_numeric(pd.Series([cushion]),errors="coerce").iloc[0]):
                st.write(f"芝クッション値：**{float(cushion):.1f}**")
            st.write(
                f"**14時頃の天気予報**　"
                f"{nfmt(weather.get('temperature_c'),'℃')}　"
                f"降水 {nfmt(weather.get('precipitation_mm'),'mm')}　"
                f"降水確率 {nfmt(weather.get('precip_probability'),'%',0)}　"
                f"風 {nfmt(weather.get('wind_kmh'),'km/h')}"
            )
            if ctx.get("used_course"):
                st.caption("使用コース："+str(ctx["used_course"]))
            if ctx.get("turf_state"):
                st.caption("芝の状態："+str(ctx["turf_state"]))
            if not valid:
                st.warning("JRA馬場情報が予想日と一致していないため、この値はAIの馬場状態には上書きしていません。")

def show_prediction(detail, context, bias):
    rg=detail.sort_values("順位").copy()
    label=str(rg.iloc[0].get("レース表示",""))
    st.subheader(label)

    st.markdown("#### 当日馬場・バイアス")
    c1,c2,c3=st.columns(3)
    c1.metric("馬場",str(rg.iloc[0].get("going","")) or "-")
    c2.metric("当日傾向",(bias or {}).get("summary","データなし"))
    weather=(context or {}).get("weather") or {}
    c3.metric("天気",weather.get("weather_text") or (context or {}).get("jra_weather") or "-")
    st.caption(
        "JRA公式の馬場状態は学習済みAIの going 特徴量へ直接反映。"
        "含水率・クッション値・当日バイアスは別の「当日補正」として表示しています。"
    )

    st.markdown("#### 全出走馬の印付き予想")
    st.caption(mark_legend())

    cols=["順位","印","評価","horse_no","horse_name",
          "win_prob","top3_prob","基礎AI指数","当日補正","ai_index",
          "odds","expected_value","補正理由"]
    q=rg[[c for c in cols if c in rg.columns]].rename(columns={
        "horse_no":"馬番","horse_name":"馬名","win_prob":"勝率",
        "top3_prob":"3着内率","ai_index":"AI指数",
        "odds":"単勝オッズ","expected_value":"AI期待値",
    })
    for c in ["勝率","3着内率"]:
        if c in q:
            q[c]=pd.to_numeric(q[c],errors="coerce").map(pct)
    for c in ["基礎AI指数","当日補正","AI指数","単勝オッズ","AI期待値"]:
        if c in q:
            q[c]=pd.to_numeric(q[c],errors="coerce").round(2)
    st.dataframe(q,use_container_width=True,hide_index=True)

    st.markdown("#### 馬券の買い方")
    c1,c2=st.columns(2)
    with c1:
        style=st.selectbox("買い方",["堅実","標準","攻め"],index=1,key="cloud_bet_style")
    with c2:
        budget=st.number_input(
            "このレースの予算（円）",min_value=500,max_value=50000,
            value=2000,step=100,key="cloud_bet_budget"
        )
    plan,meta=race_bet_plan(rg,style=style,budget_yen=int(budget))
    st.write(f"**AI上位評価の差：{meta['信頼度']}**　{meta['コメント']}")
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
    now=datetime.now(JST)
    if now.weekday()==5:
        return now.date() if now.hour<16 else now.date()+timedelta(days=1)
    if now.weekday()==6:
        return now.date() if now.hour<16 else now.date()+timedelta(days=6)
    return now.date()+timedelta(days=(5-now.weekday())%7)

st.title("🏇 競馬予想AI Cloud Ver.1.6.2")
st.caption("当日補正＋レース結果自動照合＋AI自己評価・学習データ蓄積")

st.markdown("""
<div class="hero">
<b>予想 → 結果照合 → 自己評価を自動でつなげます。</b><br>
① 当日の出走表・馬場・天気を更新<br>
② 開催地 → レースを選んで全頭予想<br>
③ 予想スナップショットを自動記録<br>
④ レース終了後に結果と照合<br>
⑤ 精度・得意条件をダッシュボードへ蓄積
</div>
""",unsafe_allow_html=True)

try:
    model_pkg,feature_store=load_cloud_assets()
except Exception as e:
    st.error(f"クラウドAIの読み込みに失敗しました：{e}")
    st.stop()

history_backend=HistoryBackend(st.secrets)
with st.expander("📊 予想履歴の保存先",expanded=False):
    st.write(f"現在：**{history_backend.label}**")
    if history_backend.persistent:
        st.success("履歴専用GitHubリポジトリへ永続保存します。")
    else:
        st.warning(
            "現在は一時保存です。Streamlit Cloud再起動で履歴が消える場合があります。"
            " 下のダッシュボードから履歴CSVを定期的にダウンロードしてください。"
        )

with st.container(border=True):
    st.markdown('<div class="step">① 当日データを取得・更新</div>',unsafe_allow_html=True)
    target_date=st.date_input("予想する開催日",value=default_race_date(),key="cloud_target_date")
    c1,c2=st.columns(2)
    with c1:
        get_clicked=st.button("出走データを取得",type="primary",use_container_width=True)
    with c2:
        refresh_clicked=st.button("最新データに更新",use_container_width=True)

    if refresh_clicked:
        fetch_entries_cached.clear()
        fetch_context_cached.clear()
        fetch_bias_cached.clear()
        get_clicked=True

    if get_clicked:
        try:
            with st.spinner("出走表・馬場・天気を取得しています…"):
                entries,info=fetch_entries_cached(target_date.isoformat())
                courses=tuple(sorted(entries["course"].dropna().astype(str).unique()))
                contexts=fetch_context_cached(target_date.isoformat(),courses)
                entries,going_changed=apply_official_going(entries,contexts,target_date)
                features=enrich_entries_cloud(entries,feature_store)

            info["going_changed_rows"]=going_changed
            info["body_weight_count"]=int(pd.to_numeric(entries["body_weight"],errors="coerce").notna().sum())
            info["odds_count"]=int(pd.to_numeric(entries["odds"],errors="coerce").notna().sum())
            st.session_state["cloud_entries"]=entries
            st.session_state["cloud_features"]=features
            st.session_state["cloud_info"]=info
            st.session_state["cloud_contexts"]=contexts
            st.success(f"{info['races']}レース・{info['rows']}頭を取得しました。")
        except Exception as e:
            st.error(f"当日データを取得できませんでした：{e}")

    info=st.session_state.get("cloud_info")
    if info:
        st.caption(
            f"取得日：{info['date']}　／　{info['races']}レース　／　{info['rows']}頭　"
            f"／ 馬体重 {info.get('body_weight_count',0)}頭　"
            f"／ オッズ {info.get('odds_count',0)}頭"
        )
        if info.get("excluded"):
            with st.expander("AI予想対象外のレース"):
                for rid,reason in info["excluded"].items():
                    st.write(f"・{rid}：{reason}")
        if info.get("errors"):
            with st.expander("取得時の補足"):
                for x in info["errors"][-20:]:
                    st.write("・"+str(x))

    show_day_context(st.session_state.get("cloud_contexts") or {})

features=st.session_state.get("cloud_features")
if features is not None and len(features):
    features=features.copy()
    features["date"]=features["date"].astype(str)
    features["course"]=features["course"].astype(str)
    features["race_no"]=features["race_no"].astype(str)
    date_iso=str(st.session_state.get("cloud_info",{}).get("date",""))
    day=features[features["date"]==date_iso].copy()

    with st.container(border=True):
        st.markdown('<div class="step">② 開催地 → レースを選んで予想</div>',unsafe_allow_html=True)
        courses=sorted(day["course"].dropna().unique())
        course=st.selectbox("① 開催地を選択",courses,key="cloud_course")
        cdf=day[day["course"]==course].copy()
        races=sorted(cdf["race_no"].dropna().unique(),key=race_num)
        race=st.selectbox(
            "② レースを選択",races,
            format_func=lambda x: race_label(x,cdf),
            key="cloud_race"
        )
        rdf=cdf[cdf["race_no"]==race].copy()
        surface=str(rdf.iloc[0].get("surface","")) if len(rdf) else ""
        context=(st.session_state.get("cloud_contexts") or {}).get(course,{})
        info=st.session_state.get("cloud_info") or {}
        official=info.get("official_entry_urls") or {}

        use_day_adjustment=st.toggle(
            "当日馬場・バイアス補正を反映",
            value=True,
            help="JRA公式馬場状態はAI本体へ反映し、含水率・馬場適性・当日内外/脚質傾向を小幅補正します。"
        )

        try:
            with st.spinner("選択レースを予想しています…"):
                detail,_=batch_predict_day(
                    rdf,date_iso,
                    model_pkg["trained_models"],model_pkg["weights"]
                )
                bias=fetch_bias_cached(
                    date_iso,course,race_num(race),surface,
                    tuple(sorted(official.items()))
                )
                detail=apply_day_adjustments(
                    detail,context,bias,enabled=use_day_adjustment
                )
            show_prediction(detail,context,bias)

            fingerprint=(
                date_iso,str(course),str(race),
                tuple(
                    (str(r.get("horse_no","")),round(float(r.get("win_prob",0)),8),
                     str(r.get("印","")),str(r.get("odds","")))
                    for _,r in detail.sort_values("horse_no").iterrows()
                )
            )
            if st.session_state.get("_saved_fp_16") != fingerprint:
                try:
                    si=save_prediction_if_new(history_backend,detail,"1.6.2")
                    st.session_state["_saved_fp_16"]=fingerprint
                    if si.get("saved"):
                        st.caption("📝 この予想を評価履歴へ記録しました。")
                except Exception as ex:
                    st.warning(f"予想履歴を保存できませんでした：{ex}")

        except Exception as e:
            st.error(f"予想できませんでした：{e}")


with st.container(border=True):
    st.markdown('<div class="step">③ 1日まとめて答え合わせ・AI自己評価</div>',unsafe_allow_html=True)

    # Use the date currently loaded in the app. All races for that date are
    # included in the report; races without a saved pre-result prediction are
    # explicitly shown as "予想履歴なし".
    settle_date=str(st.session_state.get("cloud_info",{}).get("date",""))
    entries_for_day=st.session_state.get("cloud_entries")
    expected_races=[]
    if entries_for_day is not None and len(entries_for_day):
        eq=entries_for_day.copy()
        eq=eq[eq["date"].astype(str)==settle_date] if settle_date else eq
        expected_races=list(
            eq[["course","race_no"]]
            .drop_duplicates()
            .itertuples(index=False,name=None)
        )

    history=history_backend.load()
    saved_day_races=0
    if settle_date and len(history):
        hday=history[history["date"].astype(str)==settle_date]
        saved_day_races=int(
            hday[["course","race_no"]].drop_duplicates().shape[0]
        )

    st.write(
        f"対象日：**{settle_date or '-'}**　／　"
        f"当日レース：**{len(expected_races)}R**　／　"
        f"保存済み予想：**{saved_day_races}R**"
    )
    st.caption(
        "1回押すだけで、その日の全レースを順番に確認します。"
        "結果未公開のレースは飛ばし、予想履歴がないレースも一覧で確認できます。"
    )

    bulk_clicked=st.button(
        "この日の全レースをまとめて答え合わせ",
        type="primary",
        use_container_width=True,
        disabled=not bool(settle_date),
        key="bulk_settle_day_162"
    )

    if bulk_clicked:
        info_now=st.session_state.get("cloud_info") or {}
        progress=st.progress(0,text="答え合わせを開始します…")
        def _bulk_progress(i,total,msg):
            progress.progress(
                min(1.0, i/max(total,1)),
                text=msg
            )
        try:
            result=settle_day_snapshots(
                history_backend,
                settle_date,
                official_entry_urls=info_now.get("official_entry_urls") or {},
                race_id_map=info_now.get("race_id_map") or {},
                expected_races=expected_races,
                progress_callback=_bulk_progress,
            )
            progress.progress(1.0,text="1日分の答え合わせが完了しました。")
            st.session_state["bulk_settle_result_162"]=result
            st.success(
                f"{result['total']}R確認："
                f"新規照合 {result['completed']}R／"
                f"照合済み {result['already']}R／"
                f"未公開・取得失敗 {result['failed']}R／"
                f"予想履歴なし {result['no_prediction']}R"
            )
        except Exception as ex:
            progress.empty()
            st.error(f"1日まとめて答え合わせできませんでした：{ex}")

    bulk_result=st.session_state.get("bulk_settle_result_162")
    if bulk_result and bulk_result.get("date")==settle_date:
        with st.expander("全レースの答え合わせ結果",expanded=True):
            rep=bulk_result.get("report")
            if rep is not None and len(rep):
                st.dataframe(rep,use_container_width=True,hide_index=True)

    try:
        history=history_backend.load()
        metrics,settled_df=evaluation_metrics(history)
        if metrics["races"]==0:
            st.info(
                "まだ答え合わせ済みレースがありません。"
                "結果公開後に「この日の全レースをまとめて答え合わせ」を押してください。"
            )
        else:
            a,b,c,d=st.columns(4)
            a.metric("評価済み",f"{metrics['races']}R")
            b.metric("◎勝率","-" if pd.isna(metrics["main_win_rate"]) else f"{metrics['main_win_rate']*100:.1f}%")
            c.metric("◎3着内率","-" if pd.isna(metrics["main_top3_rate"]) else f"{metrics['main_top3_rate']*100:.1f}%")
            d.metric("勝率Brier","-" if pd.isna(metrics["brier_win"]) else f"{metrics['brier_win']:.4f}")

            a,b,c=st.columns(3)
            a.metric("勝者Log Loss","-" if pd.isna(metrics["log_loss"]) else f"{metrics['log_loss']:.3f}")
            b.metric("3着内Brier","-" if pd.isna(metrics["brier_top3"]) else f"{metrics['brier_top3']:.4f}")
            c.metric("◎単勝参考回収率","-" if pd.isna(metrics["approx_main_roi"]) else f"{metrics['approx_main_roi']:.1f}%")
            st.caption(
                "Brier / Log Lossは小さいほど良好。参考回収率は予想保存時オッズ換算で、"
                "確定払戻額そのものではありません。"
            )

            t1,t2,t3,t4=st.tabs(["印別成績","競馬場・距離別","確率校正","再学習候補"])
            with t1:
                q=mark_summary(history)
                if len(q):
                    for col in ["勝率","3着内率","平均予測勝率"]:
                        q[col]=pd.to_numeric(q[col],errors="coerce").map(
                            lambda x:"-" if pd.isna(x) else f"{x*100:.1f}%"
                        )
                    st.dataframe(q,use_container_width=True,hide_index=True)
            with t2:
                q=condition_summary(history)
                if len(q):
                    for col in ["◎勝率","◎3着内率","◎平均予測勝率"]:
                        q[col]=pd.to_numeric(q[col],errors="coerce").map(
                            lambda x:"-" if pd.isna(x) else f"{x*100:.1f}%"
                        )
                    st.dataframe(q,use_container_width=True,hide_index=True)
            with t3:
                q=calibration_summary(history)
                if len(q):
                    for col in ["平均予測勝率","実勝率","差"]:
                        q[col]=pd.to_numeric(q[col],errors="coerce").map(
                            lambda x:"-" if pd.isna(x) else f"{x*100:.1f}%"
                        )
                    st.dataframe(q,use_container_width=True,hide_index=True)
                    st.caption("予測勝率と実勝率が近いほど、確率予測が適切に校正されています。")
            with t4:
                cand=training_candidate_csv(history)
                st.write(f"再学習候補：**{metrics['races']}レース / {len(cand)}頭**")
                if len(cand):
                    st.download_button(
                        "再学習候補CSVをダウンロード",
                        cand.to_csv(index=False).encode("utf-8-sig"),
                        "keiba_ai_training_candidates.csv","text/csv",
                        use_container_width=True
                    )

        st.markdown("#### 履歴バックアップ")
        raw=history_backend.load()
        st.download_button(
            "予想・結果履歴CSVをダウンロード",
            raw.to_csv(index=False).encode("utf-8-sig"),
            "keiba_ai_prediction_history.csv","text/csv",
            use_container_width=True
        )
        uploaded=st.file_uploader("履歴CSVを復元・統合",type=["csv"],key="restore16")
        if uploaded is not None and st.button("履歴CSVを統合する",use_container_width=True,key="merge16"):
            try:
                merged=history_backend.merge_import(pd.read_csv(uploaded,low_memory=False))
                st.success(f"{len(merged)}行の履歴になりました。")
            except Exception as ex:
                st.error(f"履歴を統合できませんでした：{ex}")
    except Exception as ex:
        st.error(f"自己評価を読み込めませんでした：{ex}")

with st.expander("Ver.1.6.2の自己評価について"):
    st.write(
        "レース1つごとにAIモデルを自動更新することはしません。"
        "少数データへの過学習を避けるため、まず予想確率と実結果を蓄積します。"
    )
    st.write(
        "十分なレース数が集まったら、再学習候補CSVを使って次版のモデルを"
        "時系列検証付きで再学習します。"
    )
    st.caption(
        "確実な永続保存を使う場合は、アプリ本体とは別の履歴専用GitHubリポジトリを設定します。"
    )

st.caption(
    "AI予想・当日補正・自己評価・買い目は参考情報です。"
    "過去成績は将来の的中や利益を保証するものではありません。"
)

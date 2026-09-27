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
from payout_tools import plan_to_json
from race_day_context import (
    fetch_day_contexts, apply_official_going, merge_entry_conditions,
    fetch_same_day_bias, apply_day_adjustments
)
from evaluation_store import (
    HistoryBackend, save_prediction_if_new, settle_day_snapshots,
    evaluation_metrics, mark_summary, condition_summary,
    calibration_summary, training_candidate_csv,
    race_roi_summary, daily_roi_summary, bet_type_roi_summary,
    load_public_auto_history, merge_histories
)

BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE/"data"/"cloud_model.joblib"
JST = ZoneInfo("Asia/Tokyo")

st.set_page_config(
    page_title="競馬予想AI Cloud Ver.1.9.2",
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
def fetch_context_cached(
    date_iso: str, courses_tuple, official_items=(), race_surface_pairs=()
):
    return fetch_day_contexts(
        pd.Timestamp(date_iso).date(),
        list(courses_tuple),
        dict(official_items),
        list(race_surface_pairs),
    )

@st.cache_data(ttl=90, show_spinner=False)
def fetch_bias_cached(date_iso, course, race_no, surface, official_items):
    official=dict(official_items)
    rn=race_num(race_no)
    if rn==999:
        return {
            "summary":"レース番号を解析できないため当日バイアスなし",
            "sample_races":0,
            "sample_horses":0,
            "gate_adjustments":{},
            "style_adjustments":{},
            "jockey_adjustments":{},
        }
    return fetch_same_day_bias(
        pd.Timestamp(date_iso).date(),course,rn,surface,official
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
            c3.metric("JRA天候",ctx.get("jra_weather") or weather.get("weather_text") or "-")
            srcs=[]
            if ctx.get("turf_going_source"): srcs.append("芝="+str(ctx.get("turf_going_source")))
            if ctx.get("dirt_going_source"): srcs.append("ダート="+str(ctx.get("dirt_going_source")))
            if ctx.get("weather_source"): srcs.append("天候="+str(ctx.get("weather_source")))
            if srcs:
                st.caption("取得元："+" / ".join(srcs))
            if ctx.get("live_condition_ok"):
                st.success("JRA公式当日出馬表から現在の馬場・天候を取得できています。")
            elif ctx.get("current_condition_ok"):
                st.success(
                    "JRA公式への直接取得はできませんでしたが、"
                    "同日の公開出馬表から現在の馬場・天候を取得できています。"
                )
            else:
                st.warning(
                    "現在の馬場状態を取得できていません。"
                    "下の「馬場取得診断」を確認してください。"
                )
            if ctx.get("live_source_urls"):
                st.caption("JRA当日公式ページ："+str(ctx.get("live_source_urls")[0]))
            elif ctx.get("public_source_urls"):
                st.caption("当日公開出馬表："+str(ctx.get("public_source_urls")[0]))
            if ctx.get("condition_errors"):
                with st.expander("馬場取得診断"):
                    for err in ctx.get("condition_errors")[-8:]:
                        st.write("・"+str(err))
            st.write(
                f"**JRA馬場情報ページ** {date_note}　"
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
    c3.metric("天気",(context or {}).get("jra_weather") or weather.get("weather_text") or "-")
    st.caption(
        "JRA公式出馬表の当日馬場を最優先で going 特徴量へ反映。"
        "騎手の過去総合成績は学習済みAI本体、条件別騎手成績と当日騎乗成績は小幅な騎手補正として反映します。"
    )

    st.markdown("#### 全出走馬の印付き予想")
    st.caption(mark_legend())

    cols=["順位","印","評価","horse_no","horse_name","jockey",
          "win_prob","top3_prob","jockey_top3_rate","騎手補正",
          "基礎AI指数","当日補正","ai_index",
          "odds","expected_value","騎手評価理由","補正理由"]
    q=rg[[c for c in cols if c in rg.columns]].rename(columns={
        "horse_no":"馬番","horse_name":"馬名","jockey":"騎手",
        "win_prob":"勝率","top3_prob":"3着内率",
        "jockey_top3_rate":"騎手過去3着内率","ai_index":"AI指数",
        "odds":"単勝オッズ","expected_value":"AI期待値",
    })
    for c in ["勝率","3着内率","騎手過去3着内率"]:
        if c in q:
            q[c]=pd.to_numeric(q[c],errors="coerce").map(pct)
    for c in ["騎手補正","基礎AI指数","当日補正","AI指数","単勝オッズ","AI期待値"]:
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
    return plan,meta

def default_race_date():
    now=datetime.now(JST)
    if now.weekday()==5:
        return now.date() if now.hour<16 else now.date()+timedelta(days=1)
    if now.weekday()==6:
        return now.date() if now.hour<16 else now.date()+timedelta(days=6)
    return now.date()+timedelta(days=(5-now.weekday())%7)

st.title("🏇 競馬予想AI Cloud Ver.1.9.2")
st.caption("開催地ごと全レース一括予想＋当日馬場・騎手データ・回収率集計")

st.markdown("""
<div class="hero">
<b>予想 → 結果照合 → 自己評価を自動でつなげます。</b><br>
① 当日の出走表・馬場・天気を更新<br>
② 開催地を選ぶと、その開催地の全レースを一括予想<br>
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
                official_items=tuple(sorted((info.get("official_entry_urls") or {}).items()))
                rsp=(
                    entries[["course","race_no","surface"]]
                    .drop_duplicates()
                    .copy()
                )
                race_surface_pairs=tuple(
                    (
                        str(r["course"]),
                        race_num(r["race_no"]),
                        str(r["surface"])
                    )
                    for _,r in rsp.iterrows()
                    if race_num(r["race_no"]) != 999
                )
                contexts=fetch_context_cached(
                    target_date.isoformat(),courses,official_items,race_surface_pairs
                )
                contexts=merge_entry_conditions(contexts,entries)
                entries,going_changed=apply_official_going(entries,contexts,target_date)
                features=enrich_entries_cloud(entries,feature_store)

            info["going_changed_rows"]=going_changed
            info["body_weight_count"]=int(pd.to_numeric(entries["body_weight"],errors="coerce").notna().sum())
            info["odds_count"]=int(pd.to_numeric(entries["odds"],errors="coerce").notna().sum())
            st.session_state["cloud_entries"]=entries
            st.session_state["cloud_features"]=features
            st.session_state["cloud_info"]=info
            st.session_state["cloud_contexts"]=contexts

            # 当日データを取り直したら、開催地一括予想も必ず作り直す。
            st.session_state["cloud_course_prediction_cache"]={}
            st.session_state.pop("_saved_fp_192",None)

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
date_iso=str(st.session_state.get("cloud_info",{}).get("date",""))
day=pd.DataFrame()

if features is not None and len(features):
    features=features.copy()
    features["date"]=features["date"].astype(str)
    features["course"]=features["course"].astype(str)
    features["race_no"]=features["race_no"].astype(str)
    day=features[features["date"]==date_iso].copy()


with st.container(border=True):
    st.markdown(
        '<div class="step">② 開催地ごとに全レースを一括予想</div>',
        unsafe_allow_html=True
    )

    if day.empty:
        st.info(
            "予想する当日データがまだ読み込まれていません。"
            "上の「最新データに更新」を押してください。"
        )
    else:
        courses=sorted(day["course"].dropna().unique())
        if not courses:
            st.warning("開催地データがありません。最新データを更新してください。")
            st.stop()

        course=st.selectbox(
            "① 開催地を選択",
            courses,
            key="cloud_course"
        )
        cdf=day[day["course"]==course].copy()
        races=sorted(
            cdf["race_no"].dropna().unique(),
            key=race_num
        )
        if not races:
            st.warning(f"{course}の予想対象レースがありません。")
            st.stop()

        context=(st.session_state.get("cloud_contexts") or {}).get(course,{})
        info=st.session_state.get("cloud_info") or {}
        official=info.get("official_entry_urls") or {}

        use_day_adjustment=st.toggle(
            "当日馬場・バイアス補正を反映",
            value=True,
            help=(
                "JRA公式・同日公開出馬表の当日馬場、"
                "馬場適性、内外/脚質傾向、騎手の条件別成績と"
                "当日騎乗成績を小幅補正します。"
            ),
            key="cloud_use_day_adjustment"
        )

        cache=st.session_state.setdefault(
            "cloud_course_prediction_cache",{}
        )
        cache_key=(
            str(date_iso),
            str(course),
            bool(use_day_adjustment),
        )

        if cache_key not in cache:
            try:
                with st.spinner(
                    f"{course}競馬場の全{len(races)}レースを一括予想しています…"
                ):
                    # batch_predict_day は course/race_no 単位で内部処理するため、
                    # 開催地全体を1回渡せば全レースの基礎予想をまとめて計算できる。
                    base_detail,_=batch_predict_day(
                        cdf,
                        date_iso,
                        model_pkg["trained_models"],
                        model_pkg["weights"]
                    )

                    adjusted_parts=[]
                    bias_map={}
                    summary_rows=[]

                    for rno in races:
                        race_key=str(rno)
                        rdf=cdf[cdf["race_no"].astype(str)==race_key].copy()
                        rd=base_detail[
                            base_detail["レース"].astype(str)==race_key
                        ].copy()
                        if rd.empty:
                            continue

                        surface=str(rdf.iloc[0].get("surface","")) if len(rdf) else ""
                        bias=fetch_bias_cached(
                            date_iso,
                            course,
                            race_num(rno),
                            surface,
                            tuple(sorted(official.items()))
                        )
                        rd=apply_day_adjustments(
                            rd,
                            context,
                            bias,
                            enabled=use_day_adjustment
                        )
                        adjusted_parts.append(rd)
                        bias_map[race_key]=bias

                        rg=rd.sort_values("順位").copy()
                        top=rg.iloc[0]
                        second=rg.iloc[1] if len(rg)>=2 else None
                        gap=(
                            float(top.get("ai_index",0))
                            - float(second.get("ai_index",0))
                            if second is not None else 0.0
                        )
                        race_name=str(rdf.iloc[0].get("race_name","")).strip() if len(rdf) else ""
                        going=str(rg.iloc[0].get("going",""))
                        summary_rows.append({
                            "レース":race_key,
                            "レース名":(
                                race_name
                                if race_name and race_name.lower()!="nan"
                                else f"{surface} {int(float(rdf.iloc[0].get('distance',0) or 0))}m"
                                if len(rdf) else ""
                            ),
                            "馬場":going,
                            "◎馬番":top.get("horse_no",""),
                            "◎本命馬":top.get("horse_name",""),
                            "騎手":top.get("jockey",""),
                            "勝率":top.get("win_prob",np.nan),
                            "3着内率":top.get("top3_prob",np.nan),
                            "AI指数":top.get("ai_index",np.nan),
                            "単勝オッズ":top.get("odds",np.nan),
                            "AI上位差":gap,
                            "当日傾向":(bias or {}).get("summary","データなし"),
                        })

                    all_detail=(
                        pd.concat(adjusted_parts,ignore_index=True)
                        if adjusted_parts else pd.DataFrame()
                    )
                    course_summary=pd.DataFrame(summary_rows)

                    cache[cache_key]={
                        "detail":all_detail,
                        "summary":course_summary,
                        "bias_map":bias_map,
                    }
                    st.session_state[
                        "cloud_course_prediction_cache"
                    ]=cache

            except Exception as e:
                st.error(
                    f"{course}競馬場の一括予想に失敗しました：{e}"
                )

        result=cache.get(cache_key)

        if result and len(result.get("detail",pd.DataFrame())):
            all_detail=result["detail"]
            course_summary=result.get("summary",pd.DataFrame())
            bias_map=result.get("bias_map",{})

            st.success(
                f"✅ {course}競馬場の{len(races)}レースを一括予想しました。"
                " レースを切り替えても再計算しません。"
            )

            st.markdown("#### 全レース一括予想")
            if len(course_summary):
                sq=course_summary.copy()
                for c in ["勝率","3着内率"]:
                    if c in sq:
                        sq[c]=pd.to_numeric(
                            sq[c],errors="coerce"
                        ).map(pct)
                for c in ["AI指数","単勝オッズ","AI上位差"]:
                    if c in sq:
                        sq[c]=pd.to_numeric(
                            sq[c],errors="coerce"
                        ).round(2)

                st.dataframe(
                    sq,
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown("#### レースを選択")
            st.caption(
                "ここでレースを選ぶと、下の詳細表示がそのレースへ即座に切り替わります。"
                " AI予想そのものは開催地選択時に全レース計算済みです。"
            )

            race=st.selectbox(
                "② 詳細を見るレース",
                races,
                format_func=lambda x: race_label(x,cdf),
                key="cloud_race"
            )

            # 選択レースの詳細だけを下に表示する。
            race_key=str(race)
            detail=all_detail[
                all_detail["レース"].astype(str)==race_key
            ].copy()
            bias=bias_map.get(race_key,{})
            rdf=cdf[
                cdf["race_no"].astype(str)==race_key
            ].copy()

            st.markdown(
                '<div id="selected-race-detail"></div>',
                unsafe_allow_html=True
            )

            try:
                shown_plan,shown_meta=show_prediction(
                    detail,
                    context,
                    bias
                )
                detail["bet_style"]=shown_meta.get("スタイル","")
                detail["bet_budget"]=shown_meta.get("予算",0)
                detail["bet_plan_json"]=plan_to_json(shown_plan)

                fingerprint=(
                    date_iso,
                    str(course),
                    str(race),
                    tuple(
                        (
                            str(r.get("horse_no","")),
                            round(float(r.get("win_prob",0)),8),
                            str(r.get("印","")),
                            str(r.get("odds","")),
                            str(r.get("bet_style","")),
                            str(r.get("bet_budget","")),
                            str(r.get("bet_plan_json","")),
                        )
                        for _,r in detail.sort_values(
                            "horse_no"
                        ).iterrows()
                    )
                )

                if st.session_state.get("_saved_fp_192") != fingerprint:
                    try:
                        si=save_prediction_if_new(
                            history_backend,
                            detail,
                            "1.9.2"
                        )
                        st.session_state[
                            "_saved_fp_192"
                        ]=fingerprint
                        if si.get("saved"):
                            st.caption(
                                "📝 このレースの予想を評価履歴へ記録しました。"
                            )
                    except Exception as ex:
                        st.warning(
                            f"予想履歴を保存できませんでした：{ex}"
                        )

            except Exception as e:
                st.error(
                    f"選択レースの詳細を表示できませんでした：{e}"
                )


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

    manual_history=history_backend.load()
    auto_history=load_public_auto_history()
    history=merge_histories(manual_history,auto_history)
    saved_day_races=0
    auto_day_races=0
    if settle_date and len(auto_history):
        ah=auto_history[auto_history["date"].astype(str)==settle_date]
        auto_day_races=int(
            ah[["course","race_no"]].drop_duplicates().shape[0]
        )
    if settle_date and len(history):
        hday=history[history["date"].astype(str)==settle_date]
        saved_day_races=int(
            hday[["course","race_no"]].drop_duplicates().shape[0]
        )

    st.write(
        f"対象日：**{settle_date or '-'}**　／　"
        f"当日レース：**{len(expected_races)}R**　／　"
        f"保存済み予想：**{saved_day_races}R**　／　自動レース前予想：**{auto_day_races}R**"
    )
    st.caption(
        "1回押すだけで、その日の全レースを順番に確認します。"
        " 自動レース前予想はGitHub Actionsでも17:30頃に自動答え合わせされます。"
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

with st.expander("⏰ 自動レース前予想の保存状況",expanded=False):
    auto_hist=load_public_auto_history()
    if auto_hist.empty:
        st.info(
            "自動保存履歴はまだありません。GitHub Actionsのワークフローを有効にすると、"
            "開催日の朝と発走前に自動予想を保存します。"
        )
    else:
        ah=auto_hist.copy()
        ah["recorded_dt"]=pd.to_datetime(ah["recorded_at"],errors="coerce")
        latest_date=ah["date"].astype(str).max()
        q=ah[ah["date"].astype(str)==latest_date]
        races=int(q[["course","race_no"]].drop_duplicates().shape[0])
        near=int(
            q[q["snapshot_type"].astype(str)=="pre_race"]
            [["course","race_no"]].drop_duplicates().shape[0]
        )
        morning=int(
            q[q["snapshot_type"].astype(str)=="morning"]
            [["course","race_no"]].drop_duplicates().shape[0]
        )
        settled=int(
            q[pd.to_numeric(q["actual_finish"],errors="coerce").notna()]
            [["course","race_no"]].drop_duplicates().shape[0]
        )
        day_one=q.sort_values("recorded_dt").drop_duplicates("snapshot_id",keep="last")
        exact=day_one[day_one["bet_status"].astype(str)=="確定"].copy()
        exact_stake=int(pd.to_numeric(exact["bet_stake"],errors="coerce").fillna(0).sum()) if len(exact) else 0
        exact_payout=int(pd.to_numeric(exact["bet_payout"],errors="coerce").fillna(0).sum()) if len(exact) else 0
        exact_roi=(exact_payout/exact_stake*100) if exact_stake else np.nan
        st.write(
            f"最新日：**{latest_date}**　／　対象 {races}R　／　"
            f"朝保存 {morning}R　／　発走前保存 {near}R　／　答え合わせ済み {settled}R"
        )
        if exact_stake:
            st.write(
                f"**確定馬券成績**　購入 {exact_stake:,}円　／　"
                f"払戻 {exact_payout:,}円　／　"
                f"収支 {exact_payout-exact_stake:+,}円　／　"
                f"回収率 {exact_roi:.1f}%"
            )
        show_cols=[
            "course","race_no","snapshot_type","post_time",
            "minutes_before_post","recorded_at",
            "bet_stake","bet_payout","bet_profit","bet_roi","bet_status"
        ]
        last=(
            q.sort_values("recorded_dt")
            .groupby(["course","race_no","snapshot_type"],as_index=False)
            .tail(1)
        )
        last=last[[c for c in show_cols if c in last.columns]].rename(columns={
            "course":"競馬場","race_no":"レース","snapshot_type":"保存種別",
            "post_time":"発走","minutes_before_post":"発走何分前",
            "recorded_at":"保存時刻","actual_finish":"実着順",
        })
        st.dataframe(last,use_container_width=True,hide_index=True)

    try:
        manual_history=history_backend.load()
        auto_history=load_public_auto_history()
        history=merge_histories(manual_history,auto_history)
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
            c.metric("確定回収率","-" if pd.isna(metrics["bet_roi"]) else f"{metrics['bet_roi']:.1f}%")

            if metrics["bet_races"]>0:
                a,b,c=st.columns(3)
                a.metric("購入総額",f"{metrics['bet_stake']:,}円")
                b.metric("払戻総額",f"{metrics['bet_payout']:,}円")
                c.metric("収支",f"{metrics['bet_profit']:+,}円")
            st.caption(
                "確定回収率はレース前に固定保存した買い目とJRA公式の確定払戻金で計算します。 旧履歴で買い目が未保存の場合も、保存済み予想だけから自動補完して再精算します。"
                "返還馬を含む買い目は購入額を返還として計上します。"
            )

            t1,t2,t3,t4,t5=st.tabs(
                ["回収率","印別成績","競馬場・距離別","確率校正","再学習候補"]
            )
            with t1:
                rr=race_roi_summary(history)
                if len(rr):
                    show=rr.copy()
                    show["回収率"]=pd.to_numeric(show["回収率"],errors="coerce").map(
                        lambda x:"-" if pd.isna(x) else f"{x:.1f}%"
                    )
                    st.markdown("##### レース別")
                    st.dataframe(show,use_container_width=True,hide_index=True)

                dd=daily_roi_summary(history)
                if len(dd):
                    show=dd.copy()
                    show["回収率"]=pd.to_numeric(show["回収率"],errors="coerce").map(
                        lambda x:"-" if pd.isna(x) else f"{x:.1f}%"
                    )
                    st.markdown("##### 日別")
                    st.dataframe(show,use_container_width=True,hide_index=True)

                bt=bet_type_roi_summary(history)
                if len(bt):
                    show=bt.copy()
                    show["回収率"]=pd.to_numeric(show["回収率"],errors="coerce").map(
                        lambda x:"-" if pd.isna(x) else f"{x:.1f}%"
                    )
                    st.markdown("##### 券種別")
                    st.dataframe(show,use_container_width=True,hide_index=True)
            with t2:
                q=mark_summary(history)
                if len(q):
                    for col in ["勝率","3着内率","平均予測勝率"]:
                        q[col]=pd.to_numeric(q[col],errors="coerce").map(
                            lambda x:"-" if pd.isna(x) else f"{x*100:.1f}%"
                        )
                    st.dataframe(q,use_container_width=True,hide_index=True)
            with t3:
                q=condition_summary(history)
                if len(q):
                    for col in ["◎勝率","◎3着内率","◎平均予測勝率"]:
                        q[col]=pd.to_numeric(q[col],errors="coerce").map(
                            lambda x:"-" if pd.isna(x) else f"{x*100:.1f}%"
                        )
                    st.dataframe(q,use_container_width=True,hide_index=True)
            with t4:
                q=calibration_summary(history)
                if len(q):
                    for col in ["平均予測勝率","実勝率","差"]:
                        q[col]=pd.to_numeric(q[col],errors="coerce").map(
                            lambda x:"-" if pd.isna(x) else f"{x*100:.1f}%"
                        )
                    st.dataframe(q,use_container_width=True,hide_index=True)
                    st.caption("予測勝率と実勝率が近いほど、確率予測が適切に校正されています。")
            with t5:
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
        raw=merge_histories(
            history_backend.load(),
            load_public_auto_history()
        )
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

with st.expander("Ver.1.9.2の自己評価について"):
    st.write(
        "レース1つごとにAIモデルを自動更新することはしません。"
        "少数データへの過学習を避けるため、まず予想確率と実結果を蓄積します。"
    )
    st.write(
        "十分なレース数が集まったら、再学習候補CSVを使って次版のモデルを"
        "時系列検証付きで再学習します。"
    )
    st.caption(
        "Ver.1.9.2の自動レース前予想は、同じリポジトリの prediction-history ブランチへ"
        "GitHub Actionsが保存します。mainブランチを更新しないため、予想保存のたびに"
        "Streamlitアプリが再デプロイされることはありません。"
    )

st.caption(
    "AI予想・当日補正・自己評価・買い目は参考情報です。"
    "過去成績は将来の的中や利益を保証するものではありません。"
)

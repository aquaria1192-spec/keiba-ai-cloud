# Ver.1.12 自動学習設計

## 目的

既存の HistGradientBoosting を基準モデルとして維持しながら、実際の予想時点で固定した特徴量と確定結果を蓄積し、最近の実データに対する確率補正モデルを champion / challenger 方式で安全に更新する。

## 変えないもの

- 開催地ごとの全レース一括予想
- `course_batch` の固定保存
- 一括予想時点で固定した予測値による答え合わせ
- JRA確定着順による単勝的中率・3着内率・確率精度の評価
- 基準 `data/cloud_model.joblib` / `data/cloud_feature_store.joblib`

基準モデルのバイナリは自動学習処理では上書きしない。

## Ver.1.12 の静的確率校正

2016〜2023で再学習したHistGradientBoostingに対し、2024 proxy-yearだけを使って確率校正値を固定した。

- 勝率：レース内の勝率を power=2.0654155666646536 で再配分
- 3着内率：logit確率へ slope=1.0117893511357146 / intercept=-1.2759957435674174 のPlatt校正
- 2025 proxy-yearは校正値の決定に使わず最終ホールドアウトに保持
- 本命順位は単調な勝率変換のため変更しない
- この校正はHistGB単独パッケージにだけ適用し、将来の別アンサンブルへ自動流用しない

再検証値は `VER112_REVALIDATION.json` に固定している。

## 学習データ

GitHub の `prediction-history` ブランチに次を保存する。

- `automation_data/learning/pre_race_feature_rows.csv`
- `automation_data/learning/status.json`
- `automation_data/learning/adapter_metrics.json`
- 採用時のみ `automation_data/learning/online_adapter.joblib`

レース前に以下を固定する。

- 真の日付
- 競馬場・レース番号・芝/ダート・距離・馬場
- 馬・騎手・調教師
- 予想時点のオッズ・人気・近走特徴・条件別実績
- 補正前の `raw_win_prob` / `raw_top3_prob`
- その時点の champion が実際に出した `base_win_prob` / `base_top3_prob`

確定後に `actual_finish` / `actual_win` / `actual_top3` を付与する。
結果を見てレース前特徴量を作り直さない。

## 二重補正防止

`raw_*` は静的確率校正済みHistGradientBoosting + 当日補正の出力として、オンライン補正をかける前に固定する。
次世代 challenger は常にこの `raw_*` から学習する。

現在の champion の出力へさらに次世代補正を重ねる方式にはしない。
昇格時は旧オンライン補正モデルを新しい補正モデルへ置き換える。

## 再学習タイミング

新しく確定したレースが 200 レース以上増えた場合だけ challenger を評価する。
GitHub Actions は毎週火曜 03:30 JST に確認する。

## 時系列検証

確定済みレースを日付順に並べ、最新側を検証期間にする。
未来側のレースを学習側へ混ぜない。

## 採用指標

主指標:

- 勝者 Log Loss
- Win Brier
- Top3 Log Loss
- Top3 Brier
- Calibration (ECE)

補助指標:

- 本命1着率
- 本命3着内率

馬券の買い目・回収率は評価対象にしない。モデル昇格は予測精度だけで判定する。

## 昇格条件

challenger は少なくとも次を満たした場合だけ昇格する。

- Win Log Loss が悪化しない
- Win Brier が悪化しない
- Top3 Log Loss が悪化しない
- Top3 Brier が悪化しない
- 上記4指標の平均比で 0.5% 以上改善
- Calibration が大幅悪化しない
- 本命1着率が2ポイント超悪化しない

条件を満たさなければ現 champion を維持する。

## 回帰チェック

GitHub Actions 実行前に次を確認する。

- `automation_runner.py` / `online_learning.py` / `streamlit_app.py` の構文
- ImportError
- `1R` / `01R` / 数値1のレース番号変換
- 自動学習の最低レース数が200以上であること

## 既存2016-2025データについて

既存の大規模学習データは sequence_proxy 日付を含むため、自動オンライン補正の true-date データとは区別する。
将来、基準 HistGradientBoosting 自体を再学習する場合は、既存データ＋新規 true-date データを別途統合し、時系列検証してからモデル本体と特徴量ストアを同期更新する。

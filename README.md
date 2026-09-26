# 競馬予想AI Cloud Ver.1.7

## レース前予想 自動保存版

Ver.1.7では、PCやスマホでアプリを開いていなくても
GitHub Actionsが開催日に自動実行されます。

### 自動処理

1. 開催日朝 8:30頃（JST）
   - 当日のレース一覧を取得
   - 全レースの朝予想を作成
   - `morning` として保存

2. 9:00～16:40頃
   - 20分間隔でGitHub Actionsが確認
   - 発走15～65分前に入ったレースを自動検出
   - 最新オッズ・馬体重・馬場状態を取得
   - `pre_race` として1レース1回保存

3. 17:30頃
   - 保存済み予想を結果と自動照合
   - `pre_race` があればそれを優先
   - なければ `morning` を評価

### 保存場所

アプリ本体は `main` ブランチです。

自動予想履歴は同じGitHubリポジトリ内の
`prediction-history` ブランチへ保存します。

ファイル:
- `automation_data/pre_race_predictions.csv`
- `automation_data/schedules/race_schedule_YYYYMMDD.json`

mainブランチを書き換えないため、
自動予想保存ごとにStreamlitが再デプロイされません。

## GitHub Actionsの設定

Ver.1.7のファイルをGitHubへアップロードすると、

`.github/workflows/keiba_auto_predictions.yml`

が追加されます。

### 1. Actionsを有効にする

GitHubリポジトリ
→ Actions
→ ワークフローを有効化

### 2. 書き込み権限

GitHub
→ Settings
→ Actions
→ General
→ Workflow permissions

`Read and write permissions`

を選択して Save します。

これによりGitHub Actionsが
`prediction-history` ブランチを自動作成・更新できます。

## 手動テスト

GitHub
→ Actions
→ `Keiba AI automatic pre-race predictions`
→ Run workflow

Mode:
- `morning` 朝予想を全レース作成
- `pre_race` 発走15～65分前のレースだけ保存
- `settle` その日の結果を自動照合
- `auto` 現在時刻から自動判定

Target dateに `2026-09-27` のように入力してテストできます。

## 精度評価

レース終了後の評価では、
`pre_race` の予想がある場合はそれを優先し、
なければ朝の `morning` 予想を利用します。

「結果を知った後に作った予想」は自動評価には使用しません。

## 注意

GitHub Actionsのscheduleは厳密なリアルタイム保証ではなく、
混雑時には数分～それ以上遅れることがあります。
そのため発走前保存の対象時間を15～65分前と広めにしています。

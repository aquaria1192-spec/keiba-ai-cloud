# 競馬予想AI Cloud Ver.1.6

Ver.1.6は「自己評価・学習データ蓄積版」です。

## 新機能
- 予想時の全頭確率・印・AI指数・オッズを自動記録
- 同じ予想は重複保存しない
- オッズ・馬体重などが変われば新しいスナップショットとして記録
- レース終了後に結果を取得して答え合わせ
- ◎勝率 / ◎3着内率
- 勝率Brier Score / 3着内Brier Score
- 勝者Log Loss
- 印別成績
- 競馬場・馬場・距離別成績
- 予測勝率と実勝率の校正
- 再学習候補CSV
- 履歴CSVバックアップ / 復元

## 自動再学習をまだ行わない理由
レース結果が少ないうちに毎回モデルを更新すると、偶然の結果へ過学習します。
Ver.1.6では評価データを蓄積し、十分な件数が集まってから次版で再学習します。

## 履歴の永続保存（任意・推奨）
Streamlit Cloudのローカル保存は再起動で消える可能性があります。
確実に保存したい場合は、アプリ本体とは別に履歴専用GitHubリポジトリ
（例 `keiba-ai-history`）を作成し、Streamlit Secretsへ設定します。

```toml
GITHUB_TOKEN = "github_pat_xxxxxxxxx"
GITHUB_HISTORY_REPO = "ユーザー名/keiba-ai-history"
GITHUB_HISTORY_BRANCH = "main"
GITHUB_HISTORY_PATH = "prediction_history.csv"
```

Fine-grained tokenは履歴専用リポジトリの Contents を Read and write にします。

**本物のトークンをGitHubのPython/READMEへ書かないでください。**
Streamlit CloudのSecretsだけに保存してください。

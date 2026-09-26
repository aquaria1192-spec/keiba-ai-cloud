# Cloud Ver.1.7.1 更新手順

差し替え用ZIPを展開し、`keiba-ai-cloud` の main ブランチへアップロードします。

更新:
- streamlit_app.py
- evaluation_store.py
- automation_runner.py
- README.md
- CLOUD_DEPLOY手順.md
- .github/workflows/keiba_auto_predictions.yml

新規:
- payout_tools.py

Commit changes 後、既存のGitHub Actions設定と
`prediction-history` ブランチはそのまま利用できます。

## 自動買い目
初期値は「標準・2,000円/レース」です。

## 9月27日の既存朝予想
Ver.1.7ですでに朝予想CSVを作っていても削除不要です。
Ver.1.7.1の次回Actions実行時に、買い目がない既存朝予想へ
保存済み予想だけを使って買い目を補完します。

## 答え合わせ後
アプリに次を表示します。
- 確定回収率
- 購入総額
- 払戻総額
- 収支
- レース別回収率
- 日別回収率
- 券種別回収率

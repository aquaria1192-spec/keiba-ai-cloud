# Cloud Ver.1.8 更新手順

GitHubの `keiba-ai-cloud` main ブランチへ差し替えてください。

更新ファイル:
- streamlit_app.py
- race_day_context.py
- cloud_features.py
- automation_runner.py
- README.md
- CLOUD_DEPLOY手順.md
- data/cloud_feature_store.joblib

`cloud_feature_store.joblib` は騎手の条件別成績を追加しているため、必ず更新してください。

既存の以下はそのまま利用できます。
- prediction-history ブランチ
- GitHub Actions設定
- payout_tools.py
- evaluation_store.py
- cloud_model.joblib

更新後、Streamlitで「最新データに更新」を押してください。
当日コンディション欄にJRA天候と芝/ダート馬場が表示され、
予想表には騎手過去3着内率・騎手補正が表示されます。

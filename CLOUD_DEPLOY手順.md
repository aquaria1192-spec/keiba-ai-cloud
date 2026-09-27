# Cloud Ver.1.11 更新手順

GitHub `keiba-ai-cloud` の main ブランチへ、差し替え用ZIPの内容を上書きしてください。

必須更新:
- streamlit_app.py
- feature_engineering.py
- cloud_features.py
- ml_engine.py
- model_compare.py
- race_day_context.py
- automation_runner.py
- evaluation_store.py
- README.md
- MODEL_VALIDATION.json
- data/cloud_model.joblib
- data/cloud_feature_store.joblib

特に `data/cloud_model.joblib` と `data/cloud_feature_store.joblib` は必ず両方更新してください。片方だけ古いと特徴量が一致しません。

更新後:
1. Commit changes
2. 必要なら Streamlit の `Manage app → Reboot app`
3. 「最新データに更新」
4. 開催地を選択して全レース一括予想
5. 一括予想の固定保存を確認

Ver.1.11では本命順位を勝率モデル優先で決定します。

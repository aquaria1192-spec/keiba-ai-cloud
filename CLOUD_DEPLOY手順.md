# Cloud Ver.1.8.1 更新手順

GitHub `keiba-ai-cloud` の main ブランチへ、差し替え用ZIPの内容を上書きしてください。

更新:
- race_day_context.py
- cloud_data_builder.py
- streamlit_app.py
- automation_runner.py
- README.md
- CLOUD_DEPLOY手順.md

`data/cloud_feature_store.joblib` はVer.1.8で更新済みなら再アップロード不要です。

Commit changes 後、
1. Streamlitアプリを開く
2. 「最新データに更新」を押す
3. 「当日コンディション」を開く
4. 「JRA公式当日出馬表から現在の馬場・天候を取得できています。」を確認

取得に失敗した場合は「馬場取得診断」に具体的な理由が表示されます。

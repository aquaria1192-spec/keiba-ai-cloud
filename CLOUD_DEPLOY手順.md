# Cloud Ver.1.8.2 更新手順

GitHub `keiba-ai-cloud` の main ブランチへ以下を上書きしてください。

- auto_data_builder.py
- race_day_context.py
- streamlit_app.py
- automation_runner.py
- README.md
- CLOUD_DEPLOY手順.md

Commit changes 後、
1. Streamlitアプリを開く
2. 必要なら Manage app → Reboot app
3. 「最新データに更新」を押す
4. 「当日コンディション」を開く

JRA直接取得ができなくても、
「同日の公開出馬表から現在の馬場・天候を取得できています。」
と表示され、芝・ダートに現在値が出れば正常です。

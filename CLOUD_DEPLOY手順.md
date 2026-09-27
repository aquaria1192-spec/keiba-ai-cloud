# Cloud Ver.1.10 更新手順

GitHub `keiba-ai-cloud` の main ブランチへ以下を上書きしてください。

- streamlit_app.py
- evaluation_store.py
- README.md
- CLOUD_DEPLOY手順.md

Commit changes 後、必要なら `Manage app → Reboot app` を実行してください。

## 使用手順

1. 「最新データに更新」
2. 開催地を選択
3. 全レース一括予想が完了するまで待つ
4. `答え合わせ用として一括予想 ○R を固定保存しました` を確認
5. レース詳細は必要なレースを選んで閲覧
6. 結果公開後「この日の全レースをまとめて答え合わせ」

答え合わせでは `course_batch` が最優先されます。

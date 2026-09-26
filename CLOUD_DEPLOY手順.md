# Cloud Ver.1.7 GitHub更新手順

## GitHubへアップロードするファイル

差し替え用ZIPを展開し、`keiba-ai-cloud` リポジトリへ
フォルダ構成を保ったままアップロードします。

更新:
- streamlit_app.py
- evaluation_store.py
- README.md
- CLOUD_DEPLOY手順.md

新規:
- automation_runner.py
- github_branch_store.py
- .github/workflows/keiba_auto_predictions.yml

`.github` は先頭にドットが付くフォルダです。
GitHubのUpload filesでフォルダごとドラッグしてください。

## アップロード後

1. GitHubの `Actions` タブを開く
2. ワークフローを有効化
3. Settings → Actions → General
4. Workflow permissions を `Read and write permissions`
5. Save
6. Actionsへ戻る
7. `Keiba AI automatic pre-race predictions`
8. `Run workflow`
9. mode=`morning`
10. target_date=開催日
11. 実行

成功すると `prediction-history` ブランチが自動作成されます。

## Streamlit側

Streamlitアプリは `prediction-history` ブランチの履歴を
読み取り専用で自動表示します。
追加のGitHubトークンやStreamlit Secretsは不要です。

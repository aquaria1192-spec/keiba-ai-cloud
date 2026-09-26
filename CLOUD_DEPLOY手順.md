# Streamlit Community Cloud 公開手順

このフォルダはそのままGitHubへアップロードするクラウド専用版です。

## 1. GitHub
1. GitHubで新しいリポジトリを作成します。
2. このフォルダの中身をすべてアップロードします。
3. `streamlit_app.py`、`requirements.txt`、`data` フォルダも必ず含めます。

## 2. Streamlit Community Cloud
1. https://share.streamlit.io/ にGitHubでログインします。
2. `Create app` を押します。
3. 作成したGitHubリポジトリを選択します。
4. Branch は `main`。
5. Main file path は `streamlit_app.py`。
6. `Advanced settings` を開きます。
7. **Python version は 3.13** を選択します。
8. Deploy を押します。

## 3. スマホ
公開後は `https://xxxxx.streamlit.app/` のURLが発行されます。
そのURLをiPhone/Androidで開けば、PCなしで利用できます。

ホーム画面に追加しておくと便利です。

## 重要
学習済みAIはPython 3.13 / scikit-learn 1.8.0で作成しています。
Streamlit CloudのPythonは必ず3.13を選んでください。

クラウドのローカルファイルは永続保存を前提にしていません。
本版は学習済みAIと集約済み特徴量をリポジトリに同梱しているため、
サーバー再起動後に大規模再学習する必要はありません。


## Ver.1.5へ更新する場合

すでに `keiba-ai-cloud` を公開済みの場合は、
Ver.1.5のファイル一式を同じGitHubリポジトリへ上書きしてください。

特に次のファイルは必ず更新してください。

- streamlit_app.py
- cloud_data_builder.py
- cloud_features.py
- race_day_context.py（新規）
- data/cloud_feature_store.joblib
- README.md

Streamlit Community CloudはGitHubの更新を検知すると通常は自動で再起動します。


## Cloud Ver.1.6 更新

既存 `keiba-ai-cloud` へ差し替え用ファイルをアップロードして Commit changes します。

追加:
- evaluation_store.py
- STREAMLIT_SECRETS設定例.txt

更新:
- streamlit_app.py
- README.md
- CLOUD_DEPLOY手順.md

### 履歴永続保存（任意）
1. GitHubで別リポジトリ `keiba-ai-history` を作成し、READMEを追加して初期化。
2. Fine-grained personal access tokenを作成。
3. 対象を `keiba-ai-history` のみに限定。
4. Repository permissions → Contents → Read and write。
5. Streamlitの Manage app → Settings → Secrets に設定例を貼り付け。
6. 本物のtokenはGitHubへアップロードしない。

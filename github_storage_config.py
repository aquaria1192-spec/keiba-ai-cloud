from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_HISTORY_REPO = "aquaria1192-spec/keiba-ai-cloud"
DEFAULT_HISTORY_BRANCH = "prediction-history"

_ALIAS = {
    "GITHUB_TOKEN": ("token", "github_token"),
    "GITHUB_HISTORY_REPO": ("history_repo", "repo", "repository"),
    "GITHUB_HISTORY_BRANCH": ("history_branch", "branch"),
    "GITHUB_HISTORY_PATH": ("history_path",),
    "GITHUB_ENTRY_PATH_PREFIX": ("entry_path_prefix",),
}


def _get(mapping: Any, key: str, default=""):
    if mapping is None:
        return default
    try:
        return mapping[key]
    except Exception:
        try:
            return mapping.get(key, default)
        except Exception:
            return default


def resolve_secret(secrets: Any, key: str, default=""):
    """Resolve Streamlit/GitHub settings from common secret layouts.

    Supported examples:
      GITHUB_TOKEN = "..."
      [github]
      token = "..."

    Environment variables are also accepted for deployments that inject them
    outside Streamlit's secrets.toml.
    """
    direct = _get(secrets, key, "")
    if direct not in (None, ""):
        return direct, f"secrets.{key}"

    aliases = _ALIAS.get(key, ())
    for group_name in ("github", "GITHUB"):
        group = _get(secrets, group_name, None)
        if group is None:
            continue
        for nested_key in (key, *aliases):
            value = _get(group, nested_key, "")
            if value not in (None, ""):
                return value, f"secrets.{group_name}.{nested_key}"

    env_keys = [key]
    if key == "GITHUB_TOKEN":
        env_keys.append("GH_TOKEN")
    for env_key in env_keys:
        value = os.environ.get(env_key, "")
        if value:
            return value, f"env.{env_key}"

    return default, "default"


def resolve_token(secrets: Any):
    value, source = resolve_secret(secrets, "GITHUB_TOKEN", "")
    return str(value or "").strip(), source


def resolve_history_repo(secrets: Any):
    value, source = resolve_secret(
        secrets, "GITHUB_HISTORY_REPO", DEFAULT_HISTORY_REPO
    )
    return str(value or DEFAULT_HISTORY_REPO).strip(), source


def resolve_history_branch(secrets: Any):
    value, source = resolve_secret(
        secrets, "GITHUB_HISTORY_BRANCH", DEFAULT_HISTORY_BRANCH
    )
    return str(value or DEFAULT_HISTORY_BRANCH).strip(), source


def github_headers(token: str):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _token_format_issue(token: str):
    token = str(token or "").strip()
    if not token:
        return "missing"
    try:
        token.encode("ascii")
    except UnicodeEncodeError:
        return "non_ascii"
    if any(ch.isspace() for ch in token):
        return "whitespace"
    if "ここに" in token or "token" == token.lower():
        return "placeholder"
    # GitHub user tokens commonly begin with github_pat_ (fine-grained)
    # or ghp_/gho_/ghu_/ghs_/ghr_ (classic/app-related token types).
    allowed_prefixes = ("github_pat_", "ghp_", "gho_", "ghu_", "ghs_", "ghr_")
    if not token.startswith(allowed_prefixes):
        return "unexpected_prefix"
    return ""


def diagnose_github_storage(token: str, repo: str, branch: str, timeout=12):
    """Read-only GitHub diagnostics. Never returns or logs the token."""
    result = {
        "token_detected": bool(token),
        "auth_ok": False,
        "repo_ok": False,
        "branch_ok": False,
        "write_permission": "unknown",
        "login": "",
        "status": "token_missing" if not token else "checking",
        "detail": "",
    }
    issue = _token_format_issue(token)
    if issue == "missing":
        result["detail"] = "GITHUB_TOKEN をアプリから認識できていません。"
        return result
    if issue == "non_ascii":
        result["status"] = "invalid_token_format"
        result["detail"] = (
            "GITHUB_TOKEN に日本語などASCII以外の文字が含まれています。"
            " Secretsの例文ではなく、GitHubで実際に発行された"
            " github_pat_... を設定してください。"
        )
        return result
    if issue == "whitespace":
        result["status"] = "invalid_token_format"
        result["detail"] = (
            "GITHUB_TOKEN に空白または改行が含まれています。"
            " トークン本体だけを貼り付けてください。"
        )
        return result
    if issue == "placeholder":
        result["status"] = "invalid_token_format"
        result["detail"] = (
            "GITHUB_TOKEN がサンプル文字列のままです。"
            " GitHubで実際に発行された github_pat_... に置き換えてください。"
        )
        return result
    if issue == "unexpected_prefix":
        result["status"] = "invalid_token_format"
        result["detail"] = (
            "GITHUB_TOKEN の形式を確認できません。"
            " Fine-grained personal access tokenなら通常 github_pat_... で始まります。"
        )
        return result

    headers = github_headers(token)
    try:
        user = requests.get(
            "https://api.github.com/user", headers=headers, timeout=timeout
        )
        if user.status_code != 200:
            result["status"] = "auth_failed"
            result["detail"] = f"GitHub認証に失敗しました（HTTP {user.status_code}）。"
            return result
        result["auth_ok"] = True
        try:
            result["login"] = str(user.json().get("login", "") or "")
        except Exception:
            pass

        rr = requests.get(
            f"https://api.github.com/repos/{repo}",
            headers=headers,
            timeout=timeout,
        )
        if rr.status_code != 200:
            result["status"] = "repo_failed"
            result["detail"] = (
                f"{repo} へアクセスできません（HTTP {rr.status_code}）。"
            )
            return result
        result["repo_ok"] = True
        try:
            permissions = rr.json().get("permissions") or {}
            if permissions.get("push") is True or permissions.get("admin") is True:
                result["write_permission"] = "ok"
            elif permissions and permissions.get("push") is False:
                result["write_permission"] = "denied"
        except Exception:
            pass

        br = requests.get(
            f"https://api.github.com/repos/{repo}/branches/{branch}",
            headers=headers,
            timeout=timeout,
        )
        if br.status_code != 200:
            result["status"] = "branch_failed"
            result["detail"] = (
                f"{repo} の {branch} ブランチへアクセスできません"
                f"（HTTP {br.status_code}）。"
            )
            return result
        result["branch_ok"] = True

        if result["write_permission"] == "denied":
            result["status"] = "write_denied"
            result["detail"] = (
                "リポジトリは読めますが、GitHubが書込権限なしと判定しています。"
            )
            return result

        result["status"] = "ok"
        result["detail"] = (
            "GitHub認証・リポジトリ・prediction-history ブランチへの"
            "読み取り確認に成功しました。"
        )
        return result
    except Exception as exc:
        result["status"] = "network_error"
        result["detail"] = f"GitHub診断中に通信エラーが発生しました：{exc}"
        return result

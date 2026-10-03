from __future__ import annotations

from datetime import datetime
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo
import base64
import json

import pandas as pd
import requests

BASE = Path(__file__).resolve().parent
LOCAL_DIR = BASE / "data" / "saved_entries"
JST = ZoneInfo("Asia/Tokyo")
DEFAULT_HISTORY_REPO = "aquaria1192-spec/keiba-ai-cloud"
DEFAULT_HISTORY_BRANCH = "prediction-history"


def _secret_get(secrets, key, default=""):
    try:
        return secrets[key]
    except Exception:
        try:
            return secrets.get(key, default)
        except Exception:
            return default


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class EntryDataBackend:
    """
    Persist fetched race-entry data outside Streamlit session state.

    When the same GitHub credentials used for evaluation history are available,
    CSV + metadata are stored in the history repository. This survives app
    shutdowns/reboots without touching the application repository or triggering
    a Streamlit redeploy. Local files are also kept as a development fallback.
    """

    def __init__(self, secrets=None):
        self.token = str(_secret_get(secrets, "GITHUB_TOKEN", "") or "").strip()
        self.repo = str(
            _secret_get(secrets, "GITHUB_HISTORY_REPO", DEFAULT_HISTORY_REPO)
            or DEFAULT_HISTORY_REPO
        ).strip()
        self.branch = str(
            _secret_get(secrets, "GITHUB_HISTORY_BRANCH", DEFAULT_HISTORY_BRANCH)
            or DEFAULT_HISTORY_BRANCH
        ).strip()
        self.prefix = str(
            _secret_get(secrets, "GITHUB_ENTRY_PATH_PREFIX", "saved_entries")
            or "saved_entries"
        ).strip().strip("/")
        self.mode = "github" if self.token and self.repo else "local"

    @property
    def persistent(self):
        return self.mode == "github"

    @property
    def label(self):
        if self.persistent:
            return f"{self.repo}@{self.branch}/{self.prefix}"
        return "一時ローカル保存"

    def _paths(self, date_iso):
        ymd = str(date_iso).replace("-", "")
        base = f"{self.prefix}/entries_{ymd}" if self.prefix else f"entries_{ymd}"
        return base + ".csv", base + ".json"

    def _local_paths(self, date_iso):
        ymd = str(date_iso).replace("-", "")
        return (
            LOCAL_DIR / f"entries_{ymd}.csv",
            LOCAL_DIR / f"entries_{ymd}.json",
        )

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _gh_read(self, path):
        url = f"https://api.github.com/repos/{self.repo}/contents/{path}"
        r = requests.get(
            url,
            headers=self._headers(),
            params={"ref": self.branch},
            timeout=20,
        )
        if r.status_code == 404:
            return b"", None
        r.raise_for_status()
        js = r.json()
        return base64.b64decode(js.get("content", "")), js.get("sha")

    def _gh_write(self, path, raw, message):
        _, sha = self._gh_read(path)
        payload = {
            "message": message,
            "content": base64.b64encode(raw).decode("ascii"),
            "branch": self.branch,
        }
        if sha:
            payload["sha"] = sha
        url = f"https://api.github.com/repos/{self.repo}/contents/{path}"
        r = requests.put(
            url,
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        r.raise_for_status()

    def load(self, date_iso):
        csv_local, json_local = self._local_paths(date_iso)

        if self.persistent:
            try:
                csv_path, meta_path = self._paths(date_iso)
                csv_raw, _ = self._gh_read(csv_path)
                if csv_raw:
                    df = pd.read_csv(
                        StringIO(csv_raw.decode("utf-8-sig")),
                        low_memory=False,
                    )
                    meta_raw, _ = self._gh_read(meta_path)
                    meta = (
                        json.loads(meta_raw.decode("utf-8"))
                        if meta_raw
                        else {}
                    )
                    return df, meta, {"mode": "github", "path": csv_path}
            except Exception:
                # A local copy may still be available in the current container.
                pass

        if csv_local.exists():
            try:
                df = pd.read_csv(csv_local, low_memory=False)
                meta = (
                    json.loads(json_local.read_text(encoding="utf-8"))
                    if json_local.exists()
                    else {}
                )
                return df, meta, {"mode": "local", "path": str(csv_local)}
            except Exception:
                pass

        return pd.DataFrame(), {}, {"mode": self.mode, "path": ""}

    def save(self, date_iso, entries, info=None):
        if entries is None or len(entries) == 0:
            raise ValueError("保存する出走データがありません。")

        now = datetime.now(JST).isoformat(timespec="seconds")
        meta = dict(info or {})
        meta.update({
            "date": str(date_iso),
            "saved_at": now,
            "saved_rows": int(len(entries)),
            "saved_races": int(
                entries[["course", "race_no"]].drop_duplicates().shape[0]
            )
            if {"course", "race_no"}.issubset(entries.columns)
            else 0,
        })
        meta = _json_safe(meta)

        LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        csv_local, json_local = self._local_paths(date_iso)
        entries.to_csv(csv_local, index=False, encoding="utf-8-sig")
        json_local.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        result = {
            "mode": "local",
            "path": str(csv_local),
            "saved_at": now,
        }

        if self.persistent:
            csv_path, meta_path = self._paths(date_iso)
            csv_raw = entries.to_csv(index=False).encode("utf-8-sig")
            meta_raw = json.dumps(
                meta, ensure_ascii=False, indent=2
            ).encode("utf-8")
            self._gh_write(
                csv_path,
                csv_raw,
                f"Save race entries for {date_iso}",
            )
            self._gh_write(
                meta_path,
                meta_raw,
                f"Save race-entry metadata for {date_iso}",
            )
            result = {
                "mode": "github",
                "path": csv_path,
                "saved_at": now,
            }

        return result

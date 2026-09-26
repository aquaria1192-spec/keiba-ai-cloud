from __future__ import annotations

import base64
import os
from io import StringIO

import pandas as pd
import requests

API = "https://api.github.com"

class GitHubBranchStore:
    def __init__(
        self,
        token=None,
        repo=None,
        branch="prediction-history",
        base_branch="main",
    ):
        self.token=(token or os.environ.get("GITHUB_TOKEN","")).strip()
        self.repo=(repo or os.environ.get("GITHUB_REPOSITORY","")).strip()
        self.branch=branch
        self.base_branch=base_branch
        if not self.token or not self.repo:
            raise RuntimeError("GITHUB_TOKEN / GITHUB_REPOSITORY がありません。")

    def headers(self):
        return {
            "Authorization":f"Bearer {self.token}",
            "Accept":"application/vnd.github+json",
            "X-GitHub-Api-Version":"2022-11-28",
        }

    def ensure_branch(self):
        ref=f"{API}/repos/{self.repo}/git/ref/heads/{self.branch}"
        r=requests.get(ref,headers=self.headers(),timeout=20)
        if r.status_code==200:
            return
        if r.status_code!=404:
            r.raise_for_status()

        base=f"{API}/repos/{self.repo}/git/ref/heads/{self.base_branch}"
        b=requests.get(base,headers=self.headers(),timeout=20)
        b.raise_for_status()
        sha=b.json()["object"]["sha"]

        c=requests.post(
            f"{API}/repos/{self.repo}/git/refs",
            headers=self.headers(),
            json={"ref":f"refs/heads/{self.branch}","sha":sha},
            timeout=20,
        )
        if c.status_code not in (201,422):
            c.raise_for_status()

    def read_bytes(self,path):
        self.ensure_branch()
        url=f"{API}/repos/{self.repo}/contents/{path}"
        r=requests.get(
            url,headers=self.headers(),
            params={"ref":self.branch},timeout=20
        )
        if r.status_code==404:
            return b"",None
        r.raise_for_status()
        js=r.json()
        return base64.b64decode(js.get("content","")),js.get("sha")

    def write_bytes(self,path,data,message):
        self.ensure_branch()
        _,sha=self.read_bytes(path)
        payload={
            "message":message,
            "content":base64.b64encode(data).decode("ascii"),
            "branch":self.branch,
        }
        if sha:
            payload["sha"]=sha
        url=f"{API}/repos/{self.repo}/contents/{path}"
        r=requests.put(
            url,headers=self.headers(),json=payload,timeout=30
        )
        r.raise_for_status()
        return r.json()

    def read_csv(self,path):
        raw,_=self.read_bytes(path)
        if not raw:
            return pd.DataFrame()
        return pd.read_csv(StringIO(raw.decode("utf-8-sig")),low_memory=False)

    def write_csv(self,path,df,message):
        raw=df.to_csv(index=False).encode("utf-8-sig")
        return self.write_bytes(path,raw,message)

    def read_json(self,path):
        import json
        raw,_=self.read_bytes(path)
        return json.loads(raw.decode("utf-8")) if raw else None

    def write_json(self,path,obj,message):
        import json
        raw=json.dumps(obj,ensure_ascii=False,indent=2).encode("utf-8")
        return self.write_bytes(path,raw,message)

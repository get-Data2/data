"""InfluxDB 2.x 읽기 전용 Flux 조회 클라이언트. 쓰기·삭제 API는 부르지 않는다(FR-026)."""
from __future__ import annotations

import csv
import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv


class InfluxReader:
    def __init__(self, env_path: str | Path = ".env", bucket: str = "iot-bucket", timeout: float = 300):
        load_dotenv(env_path, override=False)
        self.url = os.environ["INFLUX_URL"].rstrip("/")
        self._token = os.environ["INFLUX_TOKEN"]
        self.bucket = os.environ.get("INFLUX_BUCKET") or bucket
        self.timeout = timeout
        self.org_id = self._find_org_id()

    def _get(self, path: str):
        req = urllib.request.Request(self.url + path, headers={"Authorization": f"Token {self._token}"})
        return json.loads(urllib.request.urlopen(req, timeout=30).read())

    def _find_org_id(self) -> str:
        """INFLUX_ORG가 비어 있어도 되게 버킷 이름으로 조직 ID를 찾는다."""
        org = os.environ.get("INFLUX_ORG")
        for b in self._get("/api/v2/buckets?limit=100").get("buckets", []):
            if b["name"] == self.bucket:
                return b["orgID"]
        if org:
            return org
        raise RuntimeError(f"버킷 '{self.bucket}'을 찾을 수 없음(토큰 권한 확인)")

    def query(self, flux: str) -> list[dict]:
        """Flux 결과(annotated CSV)를 dict 목록으로. 표가 여러 개여도 머리글을 따라 읽는다."""
        req = urllib.request.Request(
            f"{self.url}/api/v2/query?orgID={self.org_id}",
            data=json.dumps({"query": flux, "type": "flux"}).encode(),
            headers={"Authorization": f"Token {self._token}", "Content-Type": "application/json",
                     "Accept": "application/csv"})
        try:
            text = urllib.request.urlopen(req, timeout=self.timeout).read().decode()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"InfluxDB 조회 실패 HTTP {e.code}: {e.read().decode()[:300]}") from None
        return parse_csv(text)


def parse_csv(text: str) -> list[dict]:
    out, head = [], None
    for r in csv.reader(io.StringIO(text)):
        if not r or r[0].startswith("#"):
            if not r:
                head = None      # 빈 줄 = 표 경계
            continue
        if head is None:
            head = r
            continue
        out.append(dict(zip(head, r)))
    return out

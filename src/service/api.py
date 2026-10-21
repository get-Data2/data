"""조회 API: 표준 라이브러리 HTTP 서버, GET만, 메모리 저장소 조회 (contracts/api.md)."""
from __future__ import annotations

import errno
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .infer import HORIZONS


class PortInUseError(RuntimeError):
    pass


OPENAPI = {
    "openapi": "3.0.3",
    "info": {"title": "team2 상시 추론 서비스", "version": "v1",
             "description": "교육장 공간별 60분 예측·이상·상태·경보(읽기 전용). 필드는 MQTT team2/inference/v1과 같다."},
    "paths": {
        "/v1/spaces": {"get": {"summary": "공간 목록과 판단 가용 상태"}},
        "/v1/spaces/{space}": {"get": {"summary": "최신 결과 묶음(예측·이상·규칙·상태·활성 경보)",
                                       "parameters": [{"name": "space", "in": "path", "required": True,
                                                       "schema": {"type": "string", "enum": ["lab", "office", "meeting"]}}]}},
        "/v1/spaces/{space}/forecast": {"get": {"summary": "t0 기준 minutes분 뒤 CO₂(ppm)·온도(°C)",
                                                "parameters": [{"name": "minutes", "in": "query",
                                                                "schema": {"type": "integer", "minimum": 5, "maximum": 60}}]}},
        "/v1/spaces/{space}/anomaly": {"get": {"summary": "이상 판단"}},
        "/v1/spaces/{space}/state": {"get": {"summary": "사용 여부·환기 선제 경보"}},
        "/v1/alerts": {"get": {"summary": "활성 경보(전체 공간)"}},
        "/v1/health": {"get": {"summary": "서비스 상태"}},
    },
}


class Api:
    def __init__(self, store, cfg):
        self.store = store
        self.cfg = cfg
        self.server = None

    def route(self, path: str, query: dict) -> tuple[int, dict]:
        parts = [p for p in path.split("/") if p]
        if parts[:1] != ["v1"]:
            return 404, {"error": f"없는 경로: {path}"}
        parts = parts[1:]
        if parts == ["openapi.json"]:
            return 200, OPENAPI
        if parts == ["health"]:
            return 200, self.store.health()
        if parts == ["alerts"]:
            st = (query.get("status") or ["active"])[0]
            if st != "active":
                return 400, {"error": "status는 active만 지원"}
            return 200, {"schema": "team2.inference.v1", "alerts": self.store.alerts()}
        if not parts or parts[0] != "spaces":
            return 404, {"error": f"없는 경로: {path}"}
        if len(parts) == 1:
            sp = self.store.spaces()
            return 200, {"schema": "team2.inference.v1", "spaces": [
                {"space_id": s, "space_name": r["space_name"], "t0": r["t0"], "freshness_min": r["freshness_min"],
                 "availability": r["availability"]} for s, r in sp.items()]}
        space = parts[1]
        if space not in self.cfg.spaces:
            return 404, {"error": f"없는 공간: {space}"}
        r = self.store.space(space)
        if r is None:
            return 503, {"error": "unavailable", "reason": "first_cycle_pending"}
        if len(parts) == 2:
            return 200, dict(r, alerts=[a for a in self.store.alerts() if a["space_id"] == space])
        kind = parts[2]
        if kind == "forecast":
            raw = (query.get("minutes") or ["60"])[0]
            try:
                m = int(raw)
            except ValueError:
                return 400, {"error": "minutes는 5~60"}
            if not 5 <= m <= int(self.cfg["api"]["max_minutes"]):
                return 400, {"error": "minutes는 5~60"}
            m = int(round(m / 5) * 5)
            avail = r["availability"].get("forecast", "")
            if avail != "ok" or not r.get("forecast"):
                return 503, {"error": "unavailable", "reason": avail.split(":", 1)[-1] or "forecast"}
            i = HORIZONS.index(m)
            co2, tmp = r["forecast"]["co2"][i], r["forecast"]["temperature"][i]
            tt = (pd.Timestamp(r["t0"]) + pd.Timedelta(minutes=m)).isoformat().replace("+00:00", "Z")
            return 200, {"schema": r["schema"], "space_id": space, "t0": r["t0"], "minutes": m, "target_time": tt,
                         "freshness_min": r["freshness_min"], "stale": r["stale"],
                         "co2": {"value": co2, "unit": "ppm"}, "temperature": {"value": tmp, "unit": "°C"},
                         "co2_over": r["co2_over"],
                         "summary_ko": f"{r['space_name']} {m}분 뒤 CO₂ {co2:,.0f}ppm, 온도 {tmp:.1f}°C 예상",
                         "model": r["models"].get("forecast")}
        if kind in ("anomaly", "state"):
            avail = r["availability"].get(kind, "")
            if avail != "ok":
                return 503, {"error": "unavailable", "reason": avail.split(":", 1)[-1] or kind}
            return 200, {"schema": r["schema"], "space_id": space, "t0": r["t0"], "freshness_min": r["freshness_min"],
                         **r[kind], "model": r["models"].get(kind)}
        return 404, {"error": f"없는 경로: {path}"}

    def start(self, bind: str | None = None, port: int | None = None):
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                u = urlparse(self.path)
                try:
                    code, body = api.route(u.path, parse_qs(u.query))
                except Exception as e:      # 처리 실패도 JSON으로
                    code, body = 500, {"error": f"{type(e).__name__}"}
                data = json.dumps(body, ensure_ascii=False, default=str).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        bind = bind or self.cfg["api"]["bind"]
        port = self.cfg["api"]["port"] if port is None else port
        try:
            self.server = ThreadingHTTPServer((bind, int(port)), Handler)
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                raise PortInUseError(f"조회 API 포트 {port}이(가) 이미 사용 중 — config/service.yaml api.port를 바꾸거나 점유 프로세스 확인") from e
            raise
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True, name="api").start()
        return self.server.server_address[1]

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()

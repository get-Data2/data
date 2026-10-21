"""경보 상태 기계: raised → renotified(재알림 간격) → cleared(연속 정상 주기) (research R7, data-model "경보").

같은 경보를 매분 보내지 않는다. 활성 경보와 정상 주기 수를 state.json에 저장해 재시작 뒤에도 다시 raised되지 않게 한다.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pandas as pd

ACTION = {
    "data_gap": ("센서 전원·게이트웨이 확인", "sensor_fault"),
    "stuck": ("센서 재시작·교체 검토(값이 바뀌지 않음)", "sensor_fault"),
    "out_of_range": ("센서 점검(오류 코드·측정 범위 밖 값)", "sensor_fault"),
    "anomaly": ("원인 센서·설비 확인", "equipment_fault"),
    "co2_over_forecast": ("환기 준비(창문·환기 장치)", "air_quality"),
    "ventilation_soon": ("30분 안에 환기 필요", "air_quality"),
    "model_drift": ("예측 오차 증가 — 모델 재학습 검토", "equipment_fault"),
    "service_degraded": ("서비스 판단 일부 불가 — 운영자 확인", "equipment_fault"),
}


def alert_id(kind: str, space: str, device: str | None) -> str:
    return hashlib.sha256(f"{kind}|{space}|{device or ''}".encode()).hexdigest()[:12]


def _iso(t: pd.Timestamp) -> str:
    return pd.Timestamp(t).isoformat().replace("+00:00", "Z")


class AlertManager:
    def __init__(self, cfg, state_path: Path | None):
        a = cfg["alerts"]
        self.renotify = pd.Timedelta(minutes=a["renotify_min"])
        self.clear_after = int(a["clear_after"])
        self.roles = a["roles"]
        self.path = state_path
        self.active: dict[str, dict] = {}
        self.ok_count: dict[str, int] = {}
        self.lock = threading.Lock()
        if state_path and state_path.exists():
            try:
                s = json.loads(state_path.read_text())
                self.active, self.ok_count = s.get("active", {}), s.get("ok_count", {})
            except Exception:
                pass                     # 깨진 상태 파일은 버리고 새로 시작한다

    def update(self, now: pd.Timestamp, findings: list[dict]) -> list[dict]:
        """findings: [{kind, space_id, device_id?, evidence}] → 이번 주기 이벤트 목록."""
        events = []
        now_s = _iso(now)
        seen = set()
        with self.lock:
            for f in findings:
                aid = alert_id(f["kind"], f["space_id"], f.get("device_id"))
                seen.add(aid)
                self.ok_count[aid] = 0
                cur = self.active.get(aid)
                action, role = ACTION.get(f["kind"], ("확인 필요", "equipment_fault"))
                if cur is None:
                    cur = {"alert_id": aid, "kind": f["kind"], "space_id": f["space_id"], "device_id": f.get("device_id"),
                           "status": "active", "started_at": now_s, "last_notified_at": now_s, "cleared_at": None,
                           "evidence": f.get("evidence", {}), "action_ko": action, "notify_role": self.roles.get(role, role)}
                    self.active[aid] = cur
                    events.append(dict(cur, event="raised"))
                else:
                    cur["evidence"] = f.get("evidence", {})
                    if now - pd.Timestamp(cur["last_notified_at"]) >= self.renotify:
                        cur["last_notified_at"] = now_s
                        events.append(dict(cur, event="renotified"))
            for aid in list(self.active):
                if aid in seen:
                    continue
                self.ok_count[aid] = self.ok_count.get(aid, 0) + 1
                if self.ok_count[aid] >= self.clear_after:
                    cur = self.active.pop(aid)
                    cur.update(status="cleared", cleared_at=now_s)
                    events.append(dict(cur, event="cleared"))
                    self.ok_count.pop(aid, None)
            self._save()
        return events

    def active_list(self, space: str | None = None) -> list[dict]:
        with self.lock:
            return [dict(a) for a in self.active.values() if space is None or a["space_id"] == space]

    def _save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"active": self.active, "ok_count": self.ok_count}, ensure_ascii=False))
            tmp.replace(self.path)

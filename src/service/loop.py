"""상시 실행: 매분 :offset_s에 주기 한 번(증분 읽기 → 전처리 → 판단 → 경보 → 저장소·발행 → 기록) (research R1).

- 주기가 다음 예정 시각을 넘기면 다음 주기는 건너뛴다(skipped). 겹쳐 돌지 않는다.
- 주기 하나가 실패해도(partial·failed 기록) 다음 주기는 처음부터 다시 한다.
- 감시 스레드: watchdog_cycles 주기 동안 진행이 없으면 프로세스를 끝내 systemd가 다시 띄우게 한다.
"""
from __future__ import annotations

import logging
import os
import resource
import threading
import time
from pathlib import Path

import pandas as pd

from .alerts import AlertManager
from .bundles import load_all
from .infer import compute, iso
from .prep import prepare
from .records import Recorder
from .selftest import SelfMonitor
from .store import Store
from .window import Window

log = logging.getLogger("service.loop")


def _rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


class Service:
    def __init__(self, cfg, source, device, publisher=None, recorder: Recorder | None = None, clock=None):
        self.cfg = cfg
        self.device = device
        self.window = Window(source, cfg["cycle"]["window_h"], cfg["cycle"]["late_min"])
        self.store = Store()
        self.pub = publisher
        self.rec = recorder or Recorder(cfg)
        self.alerts = AlertManager(cfg, Path(cfg["records"]["dir"]) / "state.json")
        self.monitor = SelfMonitor(cfg, self.rec)
        self.rule_state: dict = {}
        self.clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
        self.bundles = load_all(cfg, device)
        self.last_progress = time.monotonic()
        self.cycles = {"ok": 0, "partial": 0, "failed": 0, "skipped": 0}
        self._stop = threading.Event()
        self._reload = threading.Event()
        self.started_at = self.clock()

    # ---------- 한 주기 ----------
    def step(self, now: pd.Timestamp | None = None, scheduled: pd.Timestamp | None = None) -> dict:
        now = now or self.clock()
        scheduled = scheduled or now
        if self._reload.is_set():
            self._reload.clear()
            self.bundles = load_all(self.cfg, self.device)
            self._event("reload", "묶음 다시 적재")
        rec = {"cycle_id": int(scheduled.timestamp() // 60), "scheduled_at": iso(scheduled), "started_at": iso(now),
               "status": "ok", "error": None}
        t = time.perf_counter()
        try:
            rec["rows_read"] = self.window.update(now)
        except Exception as e:                       # DB 장애: 남은 창으로 계속(데이터 끊김으로 드러남)
            rec.update(status="partial", error=f"read:{type(e).__name__}", rows_read=0)
        rec["read_ms"] = int((time.perf_counter() - t) * 1000)
        try:
            t = time.perf_counter()
            prep = prepare(self.window.raw, self.cfg.acfg, now, self.cfg.spaces)
            rec["prep_ms"] = int((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            results, findings = compute(prep, self.bundles, self.cfg, self.rule_state)
            rec["infer_ms"] = int((time.perf_counter() - t) * 1000)
            # 자기 성능 감시
            fb = self.bundles["forecast"]
            for space, r in results.items():
                if r.get("forecast"):
                    self.monitor.log(prep.t0_bin, space, r["forecast"])
            self.monitor.score(prep.space, now)
            if fb.ok:
                findings += self.monitor.drift(getattr(fb.obj, "test_mae", {}) or {})
            for space, r in results.items():
                bad = {k: v for k, v in r["availability"].items() if v.startswith("error")}
                if bad:
                    rec["status"] = "partial"
                    findings.append({"kind": "service_degraded", "space_id": space, "device_id": None, "evidence": bad})
            computed = iso(self.clock())
            for r in results.values():
                r["computed_at"] = computed
            events = self.alerts.update(now, findings)
            active = self.alerts.active_list()
            health = self.health(rec, now)
            self.store.set_cycle(results, active, health)
            t = time.perf_counter()
            if self.pub is not None:
                from .publish import publish_cycle
                try:
                    publish_cycle(self.pub, results, active, events, health)
                except Exception as e:
                    rec.update(status="partial", error=f"publish:{type(e).__name__}")
            rec["publish_ms"] = int((time.perf_counter() - t) * 1000)
            for e in events:
                self.rec.put("alert", dict(e, at=iso(now)))
        except Exception as e:
            rec.update(status="failed", error=f"{type(e).__name__}: {e}"[:200])
            log.exception("주기 실패")
        rec["finished_at"] = iso(self.clock())
        rec["total_ms"] = sum(rec.get(k) or 0 for k in ("read_ms", "prep_ms", "infer_ms", "publish_ms"))
        rec["rss_mb"] = _rss_mb()
        rec["runtime"] = self.bundles["forecast"].runtime or self.device.type
        self.cycles[rec["status"]] += 1
        self.rec.put("cycle", rec)
        self.last_progress = time.monotonic()
        if rec["status"] == "failed":
            self.store.set_health(status="degraded", last_cycle=rec)
        return rec

    def health(self, rec: dict, now) -> dict:
        b = self.bundles
        return {"schema": "team2.inference.v1", "status": "ok" if rec["status"] == "ok" else "degraded",
                "now": iso(now), "started_at": iso(self.started_at), "last_cycle": rec, "cycles": dict(self.cycles),
                "judgements": {k: v.availability() for k, v in b.items()},
                "models": {k: v.model_meta() for k, v in b.items() if v.ok},
                "runtime_note": b["forecast"].info.get("runtime_note"), "device": self.device.type,
                "rss_mb": _rss_mb(), "mqtt_connected": None if self.pub is None else self.pub.connected,
                "records_db_ok": self.rec.db_ok, "selfmonitor": self.monitor.rolling()}

    def _event(self, kind: str, detail: str):
        self.rec.put("event", {"at": iso(self.clock()), "kind": kind, "detail": detail})

    # ---------- 상시 ----------
    def request_reload(self):
        self._reload.set()

    def stop(self):
        self._stop.set()

    def run(self):
        off = int(self.cfg["cycle"]["offset_s"])
        self._event("service_start", f"device={self.device.type}")
        threading.Thread(target=self._watchdog, daemon=True, name="watchdog").start()
        while not self._stop.is_set():
            now = self.clock()
            nxt = now.floor("1min") + pd.Timedelta(seconds=off)
            if nxt <= now:
                nxt += pd.Timedelta(minutes=1)
            if self._stop.wait((nxt - now).total_seconds()):
                break
            rec = self.step(scheduled=nxt)
            # 주기가 다음 예정 시각을 넘겼으면 그 주기는 건너뛴다
            over = (self.clock() - nxt) // pd.Timedelta(minutes=1)
            for k in range(int(over)):
                skipped = nxt + pd.Timedelta(minutes=k + 1)
                self.cycles["skipped"] += 1
                self.rec.put("cycle", {"cycle_id": int(skipped.timestamp() // 60), "scheduled_at": iso(skipped),
                                       "status": "skipped", "error": f"이전 주기 {rec.get('total_ms')}ms"})
        self._event("service_stop", "정상 종료")

    def _watchdog(self):
        limit = 60 * int(self.cfg["cycle"]["watchdog_cycles"]) + 30
        while not self._stop.wait(15):
            if time.monotonic() - self.last_progress > limit:
                log.error("감시: %d초 동안 진행 없음 — 종료(systemd 재시작)", limit)
                self._event("watchdog_exit", f"{limit}s 진행 없음")
                os._exit(3)

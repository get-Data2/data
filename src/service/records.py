"""기록: 주기·경보·예측·장애 시험 (data-model "주기 기록"·"경보"·"예측 기록"·"장애 시험 기록").

1. 항상 보드 파일(var/service/*.jsonl)에 한 줄씩 남긴다 — `infer report`가 읽는다.
2. 팀 DB 표(svc_*)에도 쓴다. DB가 끊기면 과제 1 스풀(SQLite)에 쌓았다가 나중에 넣는다.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from collector.spool import Spool

log = logging.getLogger("service.records")

DDL = {
    "cycle": """CREATE TABLE IF NOT EXISTS {p}cycle (
  cycle_id BIGINT NOT NULL PRIMARY KEY, scheduled_at DATETIME(3) NOT NULL, started_at DATETIME(3) NULL,
  finished_at DATETIME(3) NULL, status VARCHAR(16) NOT NULL, read_ms INT NULL, prep_ms INT NULL, infer_ms INT NULL,
  publish_ms INT NULL, total_ms INT NULL, rows_read INT NULL, error VARCHAR(255) NULL, rss_mb FLOAT NULL,
  gpu_mb FLOAT NULL, runtime VARCHAR(64) NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    "alert": """CREATE TABLE IF NOT EXISTS {p}alert (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY, alert_id CHAR(12) NOT NULL, kind VARCHAR(32) NOT NULL,
  space_id VARCHAR(32) NOT NULL, device_id VARCHAR(64) NULL, event VARCHAR(16) NOT NULL, status VARCHAR(16) NOT NULL,
  started_at DATETIME(3) NOT NULL, at DATETIME(3) NOT NULL, evidence JSON NULL, action_ko VARCHAR(128) NULL,
  notify_role VARCHAR(32) NULL, KEY ix_alert (alert_id, at)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    "forecast_log": """CREATE TABLE IF NOT EXISTS {p}forecast_log (
  t0 DATETIME(3) NOT NULL, space_id VARCHAR(32) NOT NULL, item VARCHAR(32) NOT NULL, horizon_min SMALLINT NOT NULL,
  y_pred DOUBLE NOT NULL, y_true DOUBLE NULL, abs_err DOUBLE NULL,
  PRIMARY KEY (t0, space_id, item, horizon_min)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    "event": """CREATE TABLE IF NOT EXISTS {p}event (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY, at DATETIME(3) NOT NULL, kind VARCHAR(32) NOT NULL,
  detail VARCHAR(255) NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
}


def _dt(s):
    if s is None:
        return None
    return str(s).replace("T", " ").replace("Z", "").split("+")[0]


def db_writer(conn_factory, prefix: str):
    """스풀 묶음 → svc_* 표. 처음 부를 때 표를 만든다."""
    state = {"conn": None}

    def write(batch: list[tuple[str, dict]]):
        if state["conn"] is None:
            state["conn"] = conn_factory()
            with state["conn"].cursor() as cur:
                for ddl in DDL.values():
                    cur.execute(ddl.format(p=prefix))
            state["conn"].commit()
        conn = state["conn"]
        try:
            conn.ping(reconnect=True)
            with conn.cursor() as cur:
                for kind, r in batch:
                    if kind == "cycle":
                        cur.execute(f"REPLACE INTO {prefix}cycle VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                                    (r["cycle_id"], _dt(r["scheduled_at"]), _dt(r.get("started_at")), _dt(r.get("finished_at")),
                                     r["status"], r.get("read_ms"), r.get("prep_ms"), r.get("infer_ms"), r.get("publish_ms"),
                                     r.get("total_ms"), r.get("rows_read"), (r.get("error") or "")[:255] or None,
                                     r.get("rss_mb"), r.get("gpu_mb"), r.get("runtime")))
                    elif kind == "alert":
                        cur.execute(f"INSERT INTO {prefix}alert (alert_id, kind, space_id, device_id, event, status, started_at, at,"
                                    " evidence, action_ko, notify_role) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                                    (r["alert_id"], r["kind"], r["space_id"], r.get("device_id"), r["event"], r["status"],
                                     _dt(r["started_at"]), _dt(r["at"]), json.dumps(r.get("evidence"), ensure_ascii=False, default=str),
                                     r.get("action_ko"), r.get("notify_role")))
                    elif kind == "forecast_log":
                        cur.execute(f"INSERT INTO {prefix}forecast_log VALUES (%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE "
                                    "y_true=VALUES(y_true), abs_err=VALUES(abs_err)",
                                    (_dt(r["t0"]), r["space_id"], r["item"], r["horizon_min"], r["y_pred"], r.get("y_true"), r.get("abs_err")))
                    elif kind == "event":
                        cur.execute(f"INSERT INTO {prefix}event (at, kind, detail) VALUES (%s,%s,%s)",
                                    (_dt(r["at"]), r["kind"], (r.get("detail") or "")[:255]))
            conn.commit()
        except Exception:
            state["conn"] = None
            raise
    return write


class Recorder:
    def __init__(self, cfg, writer=None):
        rc = cfg["records"]
        self.dir = Path(rc["dir"])
        self.dir.mkdir(parents=True, exist_ok=True)
        self.spool = Spool(rc["spool"]) if writer is not None else None
        self.writer = writer
        self._stop = threading.Event()
        self._thread = None
        self.db_ok = None

    def put(self, kind: str, row: dict):
        with open(self.dir / f"{kind}s.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        if self.spool is not None:
            self.spool.put_many([(kind, row)])

    def flush(self, limit: int = 500) -> int:
        if self.spool is None:
            return 0
        n = 0
        while True:
            rows = self.spool.peek(limit)
            if not rows:
                return n
            try:
                self.writer([(k, r) for _, k, r in rows])
                self.db_ok = True
            except Exception as e:
                if self.db_ok is not False:
                    log.warning("기록 DB 실패 — 스풀에 보관: %s", type(e).__name__)
                self.db_ok = False
                return n
            self.spool.ack(rows[-1][0])
            n += len(rows)

    def start(self, every_s: float = 2.0):
        def loop():
            while not self._stop.wait(every_s):
                self.flush()
        self._thread = threading.Thread(target=loop, daemon=True, name="records")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(5)
        self.flush()

    def read(self, kind: str) -> list[dict]:
        p = self.dir / f"{kind}s.jsonl"
        if not p.exists():
            return []
        out = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


def now_ms() -> int:
    return int(time.time() * 1000)

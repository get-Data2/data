"""메모리 창: 최근 window_h의 1분 원본. 시작 때 한 번 읽고, 주기마다 늦게 도착한 행 보정 구간부터 증분으로 읽는다
(data-model "메모리 창", research R2)."""
from __future__ import annotations

import threading
from datetime import timezone

import pandas as pd

ITEMS = ("co2", "temperature", "humidity")
COLS = ["ts", "space_id", "device_id", "item", "value"]


class DbSource:
    """MySQL measurement에서 [since, until) 행을 읽는다. 접속은 과제 1 collector.store.connect(.env) 재사용."""

    def __init__(self, cfg, env_path: str = ".env"):
        self.cfg = cfg
        self.env_path = env_path
        self._conn = None
        self._lock = threading.Lock()

    def _connect(self):
        from collector import store
        self._conn = store.connect(self.env_path)
        # 읽기 전용이라도 자동 커밋을 켠다. 끄면(과제 1 기본) 첫 SELECT의 트랜잭션 스냅샷(REPEATABLE READ)에
        # 갇혀 이후 주기가 새 행을 못 본다(2026-10-21 끊김 오경보로 발견).
        self._conn.autocommit(True)
        timeout = int(self.cfg["cycle"]["db_timeout_s"])
        with self._conn.cursor() as cur:
            cur.execute(f"SET SESSION MAX_EXECUTION_TIME={timeout * 1000}")

    def fetch(self, since: pd.Timestamp, until: pd.Timestamp | None = None) -> pd.DataFrame:
        with self._lock:
            if self._conn is None:
                self._connect()
            sql = ("SELECT ts, space_id, device_id, item, value FROM measurement "
                   "WHERE ts >= %s AND item IN %s AND space_id IN %s" + (" AND ts < %s" if until is not None else ""))
            args = [since.tz_convert(timezone.utc).tz_localize(None).to_pydatetime(), ITEMS, tuple(self.cfg.spaces)]
            if until is not None:
                args.append(until.tz_convert(timezone.utc).tz_localize(None).to_pydatetime())
            try:
                self._conn.ping(reconnect=True)
                with self._conn.cursor() as cur:
                    cur.execute(sql, args)
                    rows = cur.fetchall()
            except Exception:
                self._conn = None          # 다음 주기에 새로 접속
                raise
        df = pd.DataFrame(rows, columns=COLS)
        if len(df):
            df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize("UTC")
            df["value"] = df["value"].astype(float)
        return df


class Window:
    def __init__(self, source, window_h: float, late_min: int):
        self.source = source
        self.window = pd.Timedelta(hours=window_h)
        self.late = pd.Timedelta(minutes=late_min)
        self.raw = pd.DataFrame(columns=COLS)
        self.last_ts: pd.Timestamp | None = None
        self.last_read_rows = 0

    def update(self, now: pd.Timestamp) -> int:
        """처음엔 now − window_h부터, 이후엔 마지막 ts − late_min부터 읽어 합친다(같은 행은 하나로). 읽은 행 수."""
        since = now - self.window if self.last_ts is None else self.last_ts - self.late
        new = self.source.fetch(since, None)
        self.last_read_rows = len(new)
        if len(new):
            base = self.raw[self.raw["ts"] < since] if len(self.raw) else self.raw
            merged = pd.concat([base, new], ignore_index=True) if len(base) else new
            merged = merged.drop_duplicates(["device_id", "item", "ts"], keep="last")
            self.raw = merged[merged["ts"] >= now - self.window].sort_values("ts").reset_index(drop=True)
            self.last_ts = self.raw["ts"].max() if len(self.raw) else self.last_ts
        elif len(self.raw):
            self.raw = self.raw[self.raw["ts"] >= now - self.window].reset_index(drop=True)
        return len(new)

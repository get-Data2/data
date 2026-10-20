"""상시 수집기: MQTT 구독 → 해석 → 스풀 → MySQL 일괄 기록.

복구(research R3): paho 자동 재접속(최대 reconnect_max_s 간격), 무수신 감시, DB 실패 시 스풀 보관 후 백오프 재시도.
"""
from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

from . import store
from .config import Config, ConfigError, load_config
from .dedup import FrameDedup
from .parser import dbtime, parse
from .spool import Spool
from .timeutil import UTC

log = logging.getLogger("collector")

Writer = Callable[[list[tuple[str, dict]]], None]


class MySQLWriter:
    """스풀 묶음을 MySQL에 기록한다. 실패하면 연결을 버리고 예외를 올린다(다음 시도에 다시 접속)."""

    def __init__(self, env_path: str | Path, prefix: str):
        self.env_path, self.prefix, self.conn = env_path, prefix, None

    def __call__(self, batch):
        if self.conn is None:
            self.conn = store.connect(self.env_path)
        try:
            store.write_batch(self.conn, self.prefix, batch)
        except Exception as e:
            if not self.is_data_error(e):
                self.close()
            raise

    @staticmethod
    def is_data_error(e: Exception) -> bool:
        """행 내용 때문에 거부된 오류(다시 해도 안 됨)인가. 접속·시간 초과는 False."""
        import pymysql
        return isinstance(e, (pymysql.err.DataError, pymysql.err.IntegrityError,
                              pymysql.err.ProgrammingError, pymysql.err.NotSupportedError))

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Collector:
    def __init__(self, cfg: Config, env_path: str | Path = ".env", config_path: str | Path | None = None,
                 writer: Writer | None = None, mqtt_host: str | None = None, mqtt_port: int | None = None,
                 mqtt_auth: bool = True, host: str | None = None):
        self.cfg = cfg
        self.config_path = config_path
        self.host = host or socket.gethostname()
        load_dotenv(env_path, override=False)
        self.mqtt_host = mqtt_host or os.environ["MQTT_HOST"]
        self.mqtt_port = int(mqtt_port or os.environ.get("MQTT_PORT", "1883"))
        self.mqtt_auth = mqtt_auth
        self.writer = writer or MySQLWriter(env_path, cfg.storage["table_prefix"])
        self.spool = Spool(cfg.spool["path"], cfg.spool["max_rows"])

        self._lock = threading.Lock()
        self._stats = self._new_stats()
        self._minute = self._minute_of(_utcnow())
        self.dedup = FrameDedup(window_s=7200)
        self._stop = threading.Event()
        self._writer_thread: threading.Thread | None = None
        self.connected = threading.Event()
        self._session_present = False
        self._disconnects: list[float] = []
        self.last_message_at = time.monotonic()
        self.db_ok = True
        self._db_down_since: float | None = None
        self.stored_total = 0          # MySQL에 기록된 측정값 수(시험·상태 확인용)
        self.client = self._make_client()

    # ---------- MQTT ----------

    def _make_client(self) -> mqtt.Client:
        m = self.cfg.mqtt
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=m["client_id"], clean_session=False)
        if self.mqtt_auth and os.environ.get("MQTT_USER"):
            c.username_pw_set(os.environ["MQTT_USER"], os.environ.get("MQTT_PASS"))
        c.reconnect_delay_set(min_delay=1, max_delay=m["reconnect_max_s"])
        c.on_connect = self._on_connect
        c.on_subscribe = self._on_subscribe
        c.on_disconnect = self._on_disconnect
        c.on_message = self._on_message
        return c

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code.is_failure:
            self.event("mqtt_connect_failed", str(reason_code))
            return
        topics = self.cfg.all_topics()
        client.subscribe([(t, self.cfg.mqtt["qos"]) for t in topics])
        self._session_present = flags.session_present

    def _on_subscribe(self, client, userdata, mid, reason_codes, properties=None):
        """구독 확인까지 받아야 '접속됨'이다(그 전에 발행된 메시지는 받지 못한다)."""
        if self.connected.is_set():          # 설정 다시 읽기 때의 추가 구독
            return
        bad = [str(rc) for rc in reason_codes if rc.is_failure]
        self.last_message_at = time.monotonic()
        self.connected.set()
        self.event("mqtt_connected", f"구독 {len(reason_codes)}개(거부 {len(bad)}), "
                                     f"session_present={self._session_present}")
        log.info("MQTT 접속, 구독 %d개%s", len(reason_codes), f", 거부 {bad}" if bad else "")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        was = self.connected.is_set()
        self.connected.clear()
        if was:
            self.event("mqtt_disconnected", str(reason_code))
            log.warning("MQTT 끊김: %s", reason_code)
            # 1분에 5번 넘게 끊기면 같은 client_id를 쓰는 다른 클라이언트와 서로 밀어내는 경우가 많다(2026-10-20 실측)
            now = time.monotonic()
            self._disconnects = [t for t in self._disconnects if now - t < 60] + [now]
            if len(self._disconnects) == 5:
                self.event("mqtt_flapping", f"1분에 5회 끊김 — client_id '{self.cfg.mqtt['client_id']}' 충돌 의심")
                log.error("MQTT 접속이 반복해서 끊김 — 브로커에 같은 client_id가 있는지 확인")

    def _on_message(self, client, userdata, msg):
        received = _utcnow()
        self.last_message_at = time.monotonic()
        try:
            r = parse(msg.topic, msg.payload, received, self.cfg)
        except Exception as e:      # 해석기 버그가 수집을 멈추지 않게
            log.exception("해석 오류: %s", msg.topic)
            r = None
            failed = f"{type(e).__name__}"
        with self._lock:
            self._roll_minute(received)
            s = self._stats
            s["messages"] += 1
            if r is None:
                s["failed"] += 1
                return
            if r.failed:
                s["failed"] += 1
                log.debug("해석 실패(%s): %s", r.failed, msg.topic)
                return
            if self.dedup.is_duplicate(r.frame_key):
                s["duplicate"] += 1
                return
            s["ignored"] += r.ignored + (1 if r.ignored_topic else 0)
            s["stored"] += len(r.measurements)
            if r.lag_s is not None:
                s["max_lag_s"] = max(s["max_lag_s"] or 0.0, r.lag_s)
        rows = [("measurement", m) for m in r.measurements] + [("device", d) for d in r.devices]
        self._put(rows)

    # ---------- 스풀·사건·통계 ----------

    def _put(self, rows):
        dropped = self.spool.put_many(rows)
        if dropped:
            self.spool.put_many([("event", self._event_row(
                "spool_drop", f"{dropped['dropped']}건 {dropped['from']} ~ {dropped['to']}"))])
            log.error("스풀 한도 초과: %s", dropped)

    def _event_row(self, kind, detail=None):
        return {"ts": dbtime(_utcnow()), "kind": kind, "detail": detail, "host": self.host}

    def event(self, kind: str, detail: str | None = None) -> None:
        self._put([("event", self._event_row(kind, detail))])

    @staticmethod
    def _new_stats():
        return {"messages": 0, "stored": 0, "ignored": 0, "failed": 0, "duplicate": 0, "max_lag_s": None}

    @staticmethod
    def _minute_of(t: datetime) -> datetime:
        return t.replace(second=0, microsecond=0)

    def _roll_minute(self, now: datetime, force: bool = False):
        """분이 바뀌면 지난 분 통계를 한 줄 남긴다(메시지가 0건이어도). 호출자가 _lock을 잡는다."""
        cur = self._minute_of(now)
        if cur == self._minute and not force:
            return
        row = {"minute": self._minute.strftime("%Y-%m-%d %H:%M:00"), "host": self.host, **self._stats}
        self._stats = self._new_stats()
        self._minute = cur
        self.spool.put_many([("stats", row)])

    # ---------- 기록 스레드 ----------

    def _writer_loop(self):
        backoff = 1.0
        interval = self.cfg.storage["flush_interval_s"]
        while not self._stop.is_set():
            with self._lock:
                self._roll_minute(_utcnow())
            self._watchdog()
            wrote = self._flush_once()
            if wrote is None:      # DB 실패
                self._stop.wait(backoff)
                backoff = min(backoff * 2, self.cfg.mqtt["reconnect_max_s"])
                continue
            backoff = 1.0
            if wrote < self.cfg.storage["batch_rows"]:
                self._stop.wait(interval)

    def _flush_once(self) -> int | None:
        """스풀에서 한 묶음을 기록한다. 기록 건수, 실패면 None."""
        batch = self.spool.peek(self.cfg.storage["batch_rows"])
        if not batch:
            return 0
        try:
            self.writer([(k, r) for _, k, r in batch])
        except Exception as e:
            if getattr(self.writer, "is_data_error", lambda _e: False)(e):
                return self._write_one_by_one(batch)
            if self.db_ok:
                self.db_ok = False
                self._db_down_since = time.monotonic()
                detail = f"{type(e).__name__} {getattr(e, 'args', ('',))[0] if e.args else ''}"
                self.event("db_down", detail[:200])
                log.error("DB 기록 실패: %s", type(e).__name__)
            return None
        self.spool.ack(batch[-1][0])
        self.stored_total += sum(1 for _, k, _r in batch if k == "measurement")
        if not self.db_ok:
            self.db_ok = True
            down = time.monotonic() - (self._db_down_since or time.monotonic())
            self.event("db_up", f"{down:.0f}초 끊김, 밀린 건 {self.spool.count()}")
            log.info("DB 복구 (%.0f초)", down)
        return len(batch)

    def _write_one_by_one(self, batch) -> int | None:
        """묶음 안의 거부되는 행을 골라내 버린다. 한 행 때문에 스풀 전체가 막히지 않게 한다."""
        bad = []
        for seq, kind, row in batch:
            try:
                self.writer([(kind, row)])
            except Exception as e:
                if not self.writer.is_data_error(e):
                    return None            # 도중에 접속 오류 — 다음 시도에서 이어 간다(기록분은 멱등)
                bad.append((kind, row, f"{type(e).__name__} {e.args[0] if e.args else ''}"))
        self.spool.ack(batch[-1][0])
        for kind, row, why in bad[:5]:
            log.error("거부된 행 버림(%s): %s %s", why, kind, {k: row.get(k) for k in ("device_id", "item", "ts")})
        if bad:
            self.event("bad_row", f"{len(bad)}건 버림, 첫 사유 {bad[0][2]}"[:500])
        self.stored_total += sum(1 for _, k, _r in batch if k == "measurement") - \
            sum(1 for k, _r, _w in bad if k == "measurement")
        return len(batch)

    def _watchdog(self):
        """접속은 됐는데 오래 아무것도 안 오면 연결이 조용히 죽은 것으로 보고 다시 접속한다."""
        limit = self.cfg.mqtt["stale_reconnect_s"]
        if self.connected.is_set() and time.monotonic() - self.last_message_at > limit:
            self.event("stale_reconnect", f"{limit}초 무수신")
            log.warning("%d초 무수신 → 재접속", limit)
            self.last_message_at = time.monotonic()
            self._restart_mqtt()

    def _restart_mqtt(self):
        try:
            self.client.disconnect()
        except Exception:
            pass
        self.client.loop_stop()
        self.connected.clear()
        self.client.connect_async(self.mqtt_host, self.mqtt_port, keepalive=self.cfg.mqtt["keepalive_s"])
        self.client.loop_start()

    # ---------- 시작·종료 ----------

    def start(self):
        self.event("start", f"pid={os.getpid()}, 스풀 잔여 {self.spool.count()}")
        self.client.connect_async(self.mqtt_host, self.mqtt_port, keepalive=self.cfg.mqtt["keepalive_s"])
        self.client.loop_start()
        self._writer_thread = threading.Thread(target=self._writer_loop, name="writer", daemon=True)
        self._writer_thread.start()

    def stop(self, drain_timeout: float = 10.0):
        """구독을 끊고, 받은 측정값을 최대 drain_timeout초 동안 기록한 뒤 끝낸다(FR-012)."""
        try:
            self.client.disconnect()
        except Exception:
            pass
        self.client.loop_stop()
        self._stop.set()
        if self._writer_thread:
            self._writer_thread.join(timeout=5)
        with self._lock:
            self._roll_minute(_utcnow(), force=True)
        self.event("stop", f"스풀 잔여 {self.spool.count()}")
        deadline = time.monotonic() + drain_timeout
        while self.spool.count() and time.monotonic() < deadline:
            if self._flush_once() is None:
                time.sleep(1)
        if isinstance(self.writer, MySQLWriter):
            self.writer.close()
        self.spool.close()

    def reload_config(self):
        """SIGHUP: 설정을 다시 읽는다. 실패하면 기존 설정을 유지한다."""
        if not self.config_path:
            return
        try:
            new = load_config(self.config_path)
        except ConfigError as e:
            self.event("config_rejected", "; ".join(e.errors)[:500])
            log.error("설정 거부: %s", e)
            return
        old_topics, new_topics = set(self.cfg.all_topics()), set(new.all_topics())
        self.cfg = new
        if self.connected.is_set():
            gone = sorted(old_topics - new_topics)
            if gone:
                self.client.unsubscribe(gone)
            added = sorted(new_topics - old_topics)
            if added:
                self.client.subscribe([(t, new.mqtt["qos"]) for t in added])
        if isinstance(self.writer, MySQLWriter):
            try:
                if self.writer.conn is None:
                    self.writer.conn = store.connect(self.writer.env_path)
                store.sync_spaces(self.writer.conn, new, self.writer.prefix)
            except Exception as e:
                log.warning("공간 동기화 실패(다음 기회에): %s", type(e).__name__)
                self.writer.close()
        self.event("config_reload", f"구독 +{len(new_topics - old_topics)} -{len(old_topics - new_topics)}")
        log.info("설정 다시 읽음")

    def run_forever(self):
        logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                            format="%(asctime)s %(levelname)s %(message)s")
        done = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: done.set())
        signal.signal(signal.SIGINT, lambda *_: done.set())
        signal.signal(signal.SIGHUP, lambda *_: threading.Thread(target=self.reload_config).start())
        self.start()
        log.info("수집 시작 (host=%s)", self.host)
        while not done.wait(1):
            pass
        log.info("종료 신호 — 스풀 비우는 중")
        self.stop()
        log.info("종료")

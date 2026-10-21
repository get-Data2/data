"""시험 브로커(amqtt)·시험용 기록기·발행기 고정물."""
import asyncio
import dataclasses
import json
import socket
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import paho.mqtt.client as mqtt
import pytest
from amqtt.broker import Broker

from tests.conftest import ROOT


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestBroker:
    """별도 스레드의 이벤트 루프에서 도는 amqtt 브로커. stop()/start()로 같은 포트에서 재시작한다."""

    def __init__(self, port: int):
        self.port = port
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.broker = None

    def _run(self, coro, timeout=15):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def start(self):
        cfg = {"listeners": {"default": {"type": "tcp", "bind": f"127.0.0.1:{self.port}"}},
               "plugins": {"amqtt.plugins.authentication.AnonymousAuthPlugin": {"allow_anonymous": True}}}

        async def go():
            self.broker = Broker(cfg)
            await self.broker.start()
        self._run(go())

    def stop(self):
        if self.broker:
            self._run(self.broker.shutdown())
            self.broker = None

    def close(self):
        self.stop()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)


@pytest.fixture
def broker():
    b = TestBroker(free_port())
    b.start()
    yield b
    b.close()


class FakeWriter:
    """MySQL 대신 기록 묶음을 메모리에 쌓는다. fail=True면 DB 장애를 흉내 낸다."""

    def __init__(self):
        self.fail = False
        self.rows: list[tuple[str, dict]] = []
        self.first_meas_after: dict[float, float] = {}
        self.lock = threading.Lock()

    def __call__(self, batch):
        if self.fail:
            raise ConnectionError("시험용 DB 장애")
        with self.lock:
            self.rows.extend(batch)

    def measurements(self):
        with self.lock:
            return [r for k, r in self.rows if k == "measurement"]

    def kinds(self, kind):
        with self.lock:
            return [r for k, r in self.rows if k == kind]


@pytest.fixture
def writer():
    return FakeWriter()


@pytest.fixture
def make_collector(config, tmp_path, broker, writer):
    """시험 브로커에 붙는 수집기를 만든다. 재접속 상한을 운영값(30초) 그대로 둔다."""
    from collector.service import Collector
    made = []

    def make(spool_name="spool.db", **mqtt_over):
        cfg = dataclasses.replace(
            config,
            mqtt={**config.mqtt, "client_id": f"t-{uuid.uuid4().hex[:6]}", **mqtt_over},
            spool={**config.spool, "path": str(tmp_path / spool_name)},
            storage={**config.storage, "flush_interval_s": 0.5},
        )
        c = Collector(cfg, env_path=ROOT / ".env", writer=writer, mqtt_host="127.0.0.1",
                      mqtt_port=broker.port, mqtt_auth=False, host="test")
        made.append(c)
        return c

    yield make
    for c in made:
        if not c._stop.is_set():
            c.stop(drain_timeout=2)


class Publisher:
    """실측 표본 메시지를 시각만 지금으로 바꿔 시험 브로커에 계속 발행한다."""

    def __init__(self, port, messages):
        self.msgs = [m for m in messages if m["topic"].startswith(("application/", "xflow/"))]
        self.c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"pub-{uuid.uuid4().hex[:6]}")
        self.c.reconnect_delay_set(1, 2)
        self.ready = threading.Event()
        self.c.on_connect = lambda *a: self.ready.set()
        self.c.on_disconnect = lambda *a: self.ready.clear()
        self.c.connect_async("127.0.0.1", port)
        self.c.loop_start()
        assert self.ready.wait(10), "발행기 접속 실패"
        self.i = 0
        self.fcnt = 100000
        self.last_ts = None

    def publish_one(self):
        self.ready.wait(5)
        m = self.msgs[self.i % len(self.msgs)]
        self.i += 1
        p = json.loads(m["payload"])
        # 시각을 발행마다 최소 1ms 이상 늘린다. 같은 장치 메시지가 연달아 같은 밀리초에 나가면
        # (장치, 항목, 시각)이 겹쳐 시험이 가끔 실패했다(2026-10-21, 12회 중 1회).
        now = datetime.now(timezone.utc)
        now = now.replace(microsecond=now.microsecond // 1000 * 1000)
        if self.last_ts is not None and now <= self.last_ts:
            now = self.last_ts + timedelta(milliseconds=1)
        self.last_ts = now
        if "rxInfo" in p:
            for rx in p["rxInfo"]:
                rx["nsTime"] = now.isoformat()
            p["time"] = now.isoformat()
            self.fcnt += 1
            p["fCnt"] = self.fcnt
        if "ts" in p:
            p["ts"] = int(now.timestamp() * 1000)
        return self.c.publish(m["topic"], json.dumps(p, ensure_ascii=False), qos=1)

    def close(self):
        self.c.disconnect()
        self.c.loop_stop()


@pytest.fixture
def publisher(broker, fixture_messages):
    p = Publisher(broker.port, fixture_messages)
    yield p
    p.close()


def wait_for(pred, timeout, step=0.2):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(step)
    return False

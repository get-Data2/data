"""보드 로컬 브로커 발행 (spec US2, tasks T025): 새 구독자가 최신값을 바로 받음, 익명 발행 거부, 브로커 재시작 뒤 재발행."""
import asyncio
import json
import socket
import threading
import time

import paho.mqtt.client as mqtt
import pytest
import torch

from service.broker import broker_config, serve
from service.loop import Service
from service.publish import Publisher
from tests.service.conftest import FakeSource, at

pytestmark = pytest.mark.broker
USER, PW = "inference-test", "pw-test"


class LocalBroker:
    def __init__(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.stop_ev = None
        self.fut = None

    def start(self):
        ready = threading.Event()

        async def go():
            self.stop_ev = asyncio.Event()
            r = asyncio.Event()

            async def waiter():
                await r.wait()
                ready.set()
            asyncio.ensure_future(waiter())
            await serve(broker_config("127.0.0.1", self.port, USER, PW, ["team2/inference/v1/#"]), r, self.stop_ev)
        self.fut = asyncio.run_coroutine_threadsafe(go(), self.loop)
        assert ready.wait(10)

    def stop(self):
        self.loop.call_soon_threadsafe(self.stop_ev.set)
        self.fut.result(10)


def subscribe(port, topic="team2/inference/v1/#", wait=1.5):
    got = []
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sub-{time.time_ns()}")
    c.on_message = lambda cl, u, m: got.append((m.topic, m.payload, m.retain))
    c.connect("127.0.0.1", port)
    c.subscribe(topic, qos=1)
    c.loop_start()
    time.sleep(wait)
    c.loop_stop()
    c.disconnect()
    return got


@pytest.fixture
def broker():
    b = LocalBroker()
    b.start()
    yield b
    try:
        b.stop()
    except Exception:
        pass


@pytest.fixture
def publisher(scfg, broker, monkeypatch):
    monkeypatch.setenv("INFER_MQTT_USER", USER)
    monkeypatch.setenv("INFER_MQTT_PASS", PW)
    scfg.raw["mqtt"] = dict(scfg.raw["mqtt"], local={"host": "127.0.0.1", "port": broker.port},
                            client_id=f"svc-{time.time_ns()}")
    p = Publisher(scfg, env_path="/nonexistent")
    p.start()
    for _ in range(50):
        if p.connected:
            break
        time.sleep(0.1)
    assert p.connected
    yield p
    p.stop()


def test_new_subscriber_gets_latest(scfg, raw, publisher, broker):
    svc = Service(scfg, FakeSource(raw), torch.device("cpu"), publisher=publisher)
    svc.step(now=at("10:03:20"))
    time.sleep(0.5)
    got = subscribe(broker.port)
    topics = {t for t, _, retain in got if retain}
    assert "team2/inference/v1/lab/forecast" in topics and "team2/inference/v1/_service/health" in topics
    body = json.loads(next(p for t, p, _ in got if t == "team2/inference/v1/meeting/forecast"))
    assert body["schema"] == "team2.inference.v1" and body["space_id"] == "meeting"


def test_anonymous_publish_dropped(broker):
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="anon-pub")
    c.connect("127.0.0.1", broker.port)
    c.loop_start()
    c.publish("team2/inference/v1/lab/forecast", b"spoof", qos=1, retain=True)
    time.sleep(0.5)
    c.loop_stop()
    c.disconnect()
    assert not any(p == b"spoof" for _, p, _ in subscribe(broker.port))


def test_republish_after_broker_restart(scfg, raw, publisher, broker):
    svc = Service(scfg, FakeSource(raw), torch.device("cpu"), publisher=publisher)
    svc.step(now=at("10:03:20"))
    time.sleep(0.5)
    broker.stop()                                     # 브로커 재시작 → 보존 메시지 사라짐(메모리 브로커)
    time.sleep(1)
    broker.start()
    for _ in range(80):                               # 발행기가 다시 붙어 최신값을 다시 보존
        if publisher.connected:
            break
        time.sleep(0.25)
    time.sleep(1)
    got = subscribe(broker.port)
    assert any(t == "team2/inference/v1/lab/forecast" and r for t, _, r in got)

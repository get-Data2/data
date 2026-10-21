"""MQTT 발행: 보드 로컬 브로커(wonil-broker)에 보존(retained) 발행, 재접속 시 최신값 다시 발행 (contracts/mqtt.md).

공용 MQTT 서버(.env의 MQTT_*)에는 발행하지 않는다(2026-10-21 결정). 발행 계정은 .env의 INFER_MQTT_USER/PASS.
"""
from __future__ import annotations

import json
import logging
import os
import threading

import paho.mqtt.client as mqtt

log = logging.getLogger("service.publish")


class Publisher:
    def __init__(self, cfg, env_path: str = ".env"):
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
        m = cfg["mqtt"]
        if m["target"] != "local":
            raise ValueError("mqtt.target: 이 서비스는 보드 로컬 브로커(local)에만 발행한다")
        self.prefix = m["topic_prefix"].rstrip("/")
        self.qos = int(m["qos"])
        self.host, self.port = m["local"]["host"], int(m["local"]["port"])
        self._retained: dict[str, bytes] = {}
        self._lock = threading.Lock()
        self.connected = False
        self.c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=m["client_id"])
        user, pw = os.environ.get(m["user_env"]), os.environ.get(m["pass_env"])
        if user:
            self.c.username_pw_set(user, pw)
        self.c.reconnect_delay_set(1, 30)
        self.c.on_connect = self._on_connect
        self.c.on_disconnect = self._on_disconnect

    def start(self):
        self.c.connect_async(self.host, self.port, keepalive=30)
        self.c.loop_start()

    def stop(self):
        self.c.loop_stop()
        try:
            self.c.disconnect()
        except Exception:
            pass

    def _on_connect(self, c, userdata, flags, rc, props=None):
        self.connected = not rc.is_failure
        if self.connected:
            with self._lock:
                items = list(self._retained.items())
            for topic, payload in items:          # 브로커 재시작 뒤 최신값 다시 보존
                c.publish(topic, payload, qos=self.qos, retain=True)
            log.info("브로커 접속 — 보존 메시지 %d개 다시 발행", len(items))
        else:
            log.warning("브로커 접속 거부: %s", rc)

    def _on_disconnect(self, c, userdata, flags, rc, props=None):
        self.connected = False

    def topic(self, *parts: str) -> str:
        return "/".join([self.prefix, *parts])

    def publish(self, topic: str, body: dict, retain: bool = True):
        payload = json.dumps(body, ensure_ascii=False, default=str).encode()
        if retain:
            with self._lock:
                self._retained[topic] = payload
        if self.connected:
            self.c.publish(topic, payload, qos=self.qos, retain=retain)

    def clear_retained(self, topic: str):
        """없어진 공간의 보존 메시지 지우기(빈 보존 메시지)."""
        with self._lock:
            self._retained.pop(topic, None)
        if self.connected:
            self.c.publish(topic, b"", qos=self.qos, retain=True)


def publish_cycle(pub: Publisher, results: dict[str, dict], active: list[dict], events: list[dict], health: dict):
    for space, r in results.items():
        base = {k: r[k] for k in ("schema", "space_id", "space_name", "t0", "t0_kst", "computed_at", "freshness_min", "stale")}
        pub.publish(pub.topic(space, "forecast"), {**base, "availability": r["availability"].get("forecast"),
                                                   **(r["forecast"] or {}), "co2_over": r["co2_over"],
                                                   "summary_ko": r["summary_ko"], "model": r["models"].get("forecast")})
        pub.publish(pub.topic(space, "anomaly"), {**base, "availability": r["availability"].get("anomaly"),
                                                  **(r["anomaly"] or {}), "model": r["models"].get("anomaly")})
        pub.publish(pub.topic(space, "rules"), {**base, "availability": r["availability"].get("rules"), **r["rules"]})
        pub.publish(pub.topic(space, "state"), {**base, "availability": r["availability"].get("state"),
                                                **(r["state"] or {}), "model": r["models"].get("state")})
        pub.publish(pub.topic(space, "alerts", "active"),
                    {**base, "alerts": [a for a in active if a["space_id"] == space]})
    for e in events:
        pub.publish(pub.topic(e["space_id"], "alerts", "events"), {"schema": "team2.inference.v1", **e}, retain=False)
    pub.publish(pub.topic("_service", "health"), health)

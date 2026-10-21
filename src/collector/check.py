"""연결 점검: MQTT 브로커, MySQL, InfluxDB, 수집 서비스를 한 번에 확인한다(`python -m collector check`).

비밀 값(주소·계정·토큰)은 출력하지 않는다. MQTT는 운영 수집기와 겹치지 않도록 일회용 client_id로 잠깐 접속한다.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv


def check_mqtt(cfg, env_path, seconds: float = 10) -> tuple[bool, str]:
    import paho.mqtt.client as mqtt
    from paho.mqtt.client import topic_matches_sub
    load_dotenv(env_path, override=False)
    got: Counter = Counter()
    ready, result = threading.Event(), {}

    def on_connect(c, u, flags, rc, props=None):
        result["rc"] = str(rc)
        if not rc.is_failure:
            c.subscribe([(t, 0) for t in cfg.all_topics()])
        ready.set()

    def on_message(c, u, msg):
        src = next((s.name for s in cfg.sources if any(topic_matches_sub(t, msg.topic) for t in s.topics)), "무시 토픽")
        got[src] += 1

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"team2-check-{uuid.uuid4().hex[:6]}")
    if os.environ.get("MQTT_USER"):
        c.username_pw_set(os.environ["MQTT_USER"], os.environ.get("MQTT_PASS"))
    c.on_connect, c.on_message = on_connect, on_message
    t0 = time.monotonic()
    try:
        c.connect(os.environ["MQTT_HOST"], int(os.environ.get("MQTT_PORT", "1883")), keepalive=30)
    except Exception as e:
        return False, f"접속 실패: {type(e).__name__}"
    c.loop_start()
    if not ready.wait(10):
        c.loop_stop()
        return False, "CONNACK 없음(10초)"
    if result["rc"] != "Success":
        c.loop_stop()
        return False, f"접속 거부: {result['rc']}"
    took = time.monotonic() - t0
    time.sleep(seconds)
    c.disconnect()
    c.loop_stop()
    detail = ", ".join(f"{k} {v}건" for k, v in got.most_common()) or "메시지 없음"
    return sum(got.values()) > 0, f"접속 {took:.1f}초, {seconds:g}초 동안 {detail}"


def check_mysql(env_path, prefix="") -> tuple[bool, str]:
    from . import store
    try:
        conn = store.connect(env_path)
    except Exception as e:
        return False, f"접속 실패: {type(e).__name__}"
    with conn.cursor() as cur:
        cur.execute("SELECT VERSION()")
        ver = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*), TIMESTAMPDIFF(SECOND, MAX(received_at), UTC_TIMESTAMP(3)) "
                    f"FROM `{prefix}measurement` WHERE received_at > UTC_TIMESTAMP() - INTERVAL 5 MINUTE")
        n, age = cur.fetchone()
    conn.close()
    ok = bool(n)
    return ok, f"MySQL {ver.split('-')[0]}, 최근 5분 저장 {n}건" + (f", 마지막 저장 {age}초 전" if age is not None else "")


def check_influx(cfg, env_path) -> tuple[bool, str]:
    import json
    import urllib.request
    from .influx import InfluxReader
    load_dotenv(env_path, override=False)
    try:
        url = os.environ["INFLUX_URL"].rstrip("/")
        health = json.loads(urllib.request.urlopen(url + "/health", timeout=5).read())
        r = InfluxReader(env_path, cfg.backfill["bucket"], timeout=30)
        last = r.query(f'from(bucket:"{r.bucket}") |> range(start: -1h) |> filter(fn:(r)=> r._field=="value") '
                       f'|> group() |> last()')
    except Exception as e:
        return False, f"접속 실패: {type(e).__name__} (사설망 — 유선 랜 경로 확인)"
    t = last[0]["_time"][:19].replace("T", " ") + " UTC" if last else "최근 1시간 데이터 없음"
    return True, f"InfluxDB {health.get('version', '?')} {health.get('status')}, 버킷 '{r.bucket}' 최근 값 {t}"


def check_service(unit: str) -> tuple[bool, str]:
    try:
        out = subprocess.run(["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "NRestarts",
                              "-p", "MemoryCurrent"], capture_output=True, text=True, timeout=10).stdout
    except Exception as e:
        return False, f"확인 실패: {type(e).__name__}"
    kv = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
    mem = kv.get("MemoryCurrent", "")
    mem_txt = f", 메모리 {int(mem) / 2**20:.0f}MB" if mem.isdigit() else ""
    return kv.get("ActiveState") == "active", f"{kv.get('ActiveState')}, 재시작 {kv.get('NRestarts')}회{mem_txt}"


def run_checks(cfg, env_path: str | Path, seconds: float = 10, skip_influx: bool = False) -> tuple[bool, str]:
    rows = [("수집 서비스", *check_service("wonil-collector.service")),
            ("MQTT 브로커", *check_mqtt(cfg, env_path, seconds)),
            ("MySQL", *check_mysql(env_path, cfg.storage["table_prefix"]))]
    if not skip_influx:
        rows.append(("InfluxDB(과거 데이터)", *check_influx(cfg, env_path)))
    lines = ["| 대상 | 결과 | 내용 |", "|---|---|---|"]
    lines += [f"| {name} | {'✅' if ok else '❌'} | {msg} |" for name, ok, msg in rows]
    essential = all(ok for name, ok, _ in rows if not name.startswith("InfluxDB"))
    return essential, "\n".join(lines)

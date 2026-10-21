"""수집 정확성 독립 검증: 원본 MQTT 메시지에서 기대 행을 직접 계산해 MySQL과 양방향 대조한다.

수집기의 해석 코드(parser·config)를 쓰지 않는다 — 같은 실수를 같이 하지 않기 위해서다. DB 접속만 collector.store를 쓴다.
사용: .venv/bin/python tools/verify_ingest.py [초=180]
"""
import json, math, os, sys, time, uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

from collector import store

# 기대값 표(설정과 별도로 손으로 적음 — 설정이 바뀌면 함께 갱신)
SPACE = {"실습실": "lab", "사무실": "office", "사무실 밖": "office_out", "대표실": "ceo_room", "개발실": "dev_room",
         "회의실": "meeting", "집중회의실": "focus_meeting", "controller": "plant", "창고": "storage"}
OVERRIDE = {"24e124785c389818": "lab"}
ALIAS = {"current_temperature": "temperature", "current_humidity": "humidity"}


def iso(s):
    s = s.replace("Z", "+00:00")
    head, _, rest = s.partition(".")
    if rest:
        n = 0
        while n < len(rest) and rest[n].isdigit():
            n += 1
        s = f"{head}.{rest[:n][:6].ljust(6, '0')}{rest[n:]}"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def capture(seconds):
    load_dotenv(".env")
    msgs = []
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"wonil-verify-{uuid.uuid4().hex[:6]}")
    c.username_pw_set(os.environ["MQTT_USER"], os.environ["MQTT_PASS"])
    c.on_connect = lambda c, u, f, rc, p=None: c.subscribe(
        [("application/+/device/+/event/up", 1), ("xflow/hvacr/+/+/status", 1), ("xflow/controller/+/status", 1)])
    c.on_message = lambda c, u, m: msgs.append((time.time(), m.topic, m.payload.decode("utf-8", "replace")))
    c.connect(os.environ["MQTT_HOST"], int(os.environ["MQTT_PORT"]))
    c.loop_start(); time.sleep(seconds); c.loop_stop(); c.disconnect()
    return msgs


def expected(msgs):
    exp, frames = [], defaultdict(int)
    for _, topic, payload in msgs:
        p = json.loads(payload)
        if topic.startswith("application/"):
            dev = p["deviceInfo"]["devEui"]; loc = p["deviceInfo"].get("tags", {}).get("location")
            ts = min(iso(r["nsTime"]) for r in p["rxInfo"]); fields = p.get("object", {})
            frames[(dev, p.get("fCnt"))] += 1
            if frames[(dev, p.get("fCnt"))] > 1:
                continue                                  # 같은 업링크 이중 발행 → 첫 판만 기대
        else:
            dev = topic.split("/")[-2]; name, fields = next(iter(p["values"].items()))
            loc = name.split("-")[0]; ts = datetime.fromtimestamp(p["ts"] / 1000, timezone.utc)
        space = OVERRIDE.get(dev) or SPACE.get(loc, "unmapped")
        ts = ts.replace(microsecond=ts.microsecond // 1000 * 1000, tzinfo=None)
        for f, v in fields.items():
            if isinstance(v, bool):
                v = 1.0 if v else 0.0
            elif not (isinstance(v, (int, float)) and math.isfinite(v)):
                continue
            exp.append((dev, ALIAS.get(f, f), ts, space, float(v)))
    return exp, sum(1 for n in frames.values() if n > 1)


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 180
    msgs = capture(seconds)
    time.sleep(10)                                        # 수집기 일괄 기록(2초 주기) 여유
    exp, dup = expected(msgs)
    cur = store.connect(".env").cursor()
    res, bad = Counter(), []
    for dev, item, ts, space, v in exp:
        cur.execute("SELECT space_id, value FROM measurement WHERE device_id=%s AND item=%s AND ts=%s", [dev, item, ts])
        got = cur.fetchall()
        if len(got) != 1:
            res["누락" if not got else "중복"] += 1; bad.append((dev, item, ts, len(got)))
        elif got[0][0] != space:
            res["공간 불일치"] += 1; bad.append((dev, item, got[0][0], space))
        elif abs(got[0][1] - v) > 1e-9:
            res["값 불일치"] += 1; bad.append((dev, item, got[0][1], v))
        else:
            res["일치"] += 1
    lo = datetime.fromtimestamp(msgs[0][0] + 5, timezone.utc).replace(tzinfo=None)
    hi = datetime.fromtimestamp(msgs[-1][0] - 5, timezone.utc).replace(tzinfo=None)
    cur.execute("SELECT device_id, item, ts FROM measurement WHERE source<>'influx' AND received_at BETWEEN %s AND %s", [lo, hi])
    want = {(d, i, t) for d, i, t, _, _ in exp}
    db = cur.fetchall()
    extra = [r for r in db if r not in want]
    print(f"측정 조건: {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M %Z')}, 원본 {seconds:g}초, 실데이터")
    print(f"원본 메시지 {len(msgs)}건 → 기대 행 {len(exp)}개: {dict(res)} (이중 발행 업링크 {dup}건은 첫 판만 기대)")
    print(f"반대 방향: 그 시간대 DB 행 {len(db)}개 중 원본에 없는 행 {len(extra)}개")
    print("공간별 기대 행:", dict(Counter(r[3] for r in exp)))
    for b in (bad + extra)[:10]:
        print("  문제:", b)
    sys.exit(0 if res["일치"] == len(exp) and not extra else 1)


if __name__ == "__main__":
    main()

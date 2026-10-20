import json
from datetime import datetime, timedelta, timezone

import pytest

from collector.parser import parse
from collector.timeutil import parse_time

pytestmark = pytest.mark.unit
UTC = timezone.utc


def recv(m):
    return datetime.fromtimestamp(m["t"], UTC)


def find(msgs, pred):
    return next(m for m in msgs if pred(m))


def test_chirpstack_measurements_and_time(fixture_messages, config):
    # 시계가 빠른 게이트웨이(…dccc)를 먼저 거친 EM300-TH 메시지
    m = find(fixture_messages, lambda m: "24e124136d151836" in m["topic"])
    r = parse(m["topic"], m["payload"].encode(), recv(m), config)
    assert r.source == "chirpstack" and r.failed is None
    items = {x["item"]: x for x in r.measurements}
    assert set(items) == {"temperature", "humidity"}
    t = items["temperature"]
    assert t["value"] == 20.7 and t["unit"] == "degC"
    assert t["space_id"] == "office_out" and t["device_id"] == "24e124136d151836"
    ts = parse_time(t["ts"])
    assert abs((ts - recv(m)).total_seconds()) < 5          # nsTime 기준 → 수신 시각과 몇 초 안
    src = parse_time(t["source_ts"])
    assert (src - recv(m)).total_seconds() > 250             # 원천 time은 약 5분 빠름(보존만)
    assert t["ts_corrected"] == 0
    assert isinstance(t["frame_cnt"], int)
    d = r.devices[0]
    assert d["model"] == "EM300-TH" and d["spot"] == "입구 오른쪽" and d["mapped_by"] == "alias"
    assert d["raw_location"] == "사무실 밖"


def test_chirpstack_co2_sensor(fixture_messages, config):
    m = find(fixture_messages, lambda m: "24e124128c067999" in m["topic"])   # AM107
    r = parse(m["topic"], m["payload"].encode(), recv(m), config)
    items = {x["item"]: x["value"] for x in r.measurements}
    assert items["co2"] == 1025 and "tvoc" in items and "activity" in items
    assert all(x["space_id"] == "lab" for x in r.measurements)


def test_device_override_wins(fixture_messages, config):
    m = find(fixture_messages, lambda m: "24e124785c389818" in m["topic"])
    r = parse(m["topic"], m["payload"].encode(), recv(m), config)
    assert r.devices[0]["mapped_by"] == "device" and r.devices[0]["space_id"] == "lab"


def test_xflow_idu(fixture_messages, config):
    m = find(fixture_messages, lambda m: "/idu/4a848cc0" in m["topic"])
    r = parse(m["topic"], m["payload"].encode(), recv(m), config)
    assert r.source == "xflow"
    items = {x["item"]: x["value"] for x in r.measurements}
    assert items["power"] == 0.0 and items["temperature"] == 20.7 and items["humidity"] == 55
    assert all(x["space_id"] == "ceo_room" for x in r.measurements)
    d = r.devices[0]
    assert d["device_id"] == "4a848cc0-8598-11f1-bcac-bdfb397bbe8a" and d["model"] == "idu"
    assert d["name"] == "대표실-실내기"


def test_xflow_controller_string_value_ignored(fixture_messages, config):
    m = find(fixture_messages, lambda m: "/controller/" in m["topic"])
    r = parse(m["topic"], m["payload"].encode(), recv(m), config)
    assert r.ignored == 1                                     # op_mode: "unknown"
    assert {x["space_id"] for x in r.measurements} == {"plant"}
    assert "compressor_run" in {x["item"] for x in r.measurements}


def test_xflow_simple_rooms(fixture_messages, config):
    spaces = set()
    for m in fixture_messages:
        if m["topic"].startswith("xflow/"):
            r = parse(m["topic"], m["payload"].encode(), recv(m), config)
            spaces |= {x["space_id"] for x in r.measurements}
    assert {"ceo_room", "dev_room", "meeting", "focus_meeting", "office", "plant"} <= spaces


def test_no_unmapped_in_fixture(fixture_messages, config):
    for m in fixture_messages:
        r = parse(m["topic"], m["payload"].encode(), recv(m), config)
        assert all(x["space_id"] != "unmapped" for x in r.measurements), m["topic"]


def test_gateway_state_ignored(fixture_messages, config):
    m = find(fixture_messages, lambda m: m["topic"].startswith("gateway/"))
    r = parse(m["topic"], m["payload"].encode(), recv(m), config)
    assert r.ignored_topic and not r.measurements and r.failed is None


def test_broken_json_fails(config):
    r = parse("application/a/device/b/event/up", b"{not json", datetime.now(UTC), config)
    assert r.failed and not r.measurements


def test_missing_device_id_fails(config):
    r = parse("application/a/device/b/event/up", b'{"object": {"co2": 1}}', datetime.now(UTC), config)
    assert r.failed == "device_id 없음"


def _cs(ns_time, obj=None):
    return json.dumps({
        "deviceInfo": {"devEui": "dev1", "tags": {"location": "실습실"}},
        "rxInfo": [{"nsTime": ns_time}], "time": ns_time, "fCnt": 7,
        "object": obj or {"co2": 0},
    }).encode()


@pytest.mark.parametrize("offset_s, corrected", [(0, 0), (100, 0), (200, 1), (-3000, 0), (-4000, 1)])
def test_time_range_correction(config, offset_s, corrected):
    now = datetime(2026, 10, 20, 6, 0, tzinfo=UTC)
    ns = (now + timedelta(seconds=offset_s)).isoformat()
    r = parse("application/a/device/dev1/event/up", _cs(ns), now, config)
    m = r.measurements[0]
    assert m["ts_corrected"] == corrected
    expect = now if corrected else now + timedelta(seconds=offset_s)
    assert parse_time(m["ts"]) == expect


def test_broken_value_kept(config):
    """0ppm 같은 깨진 값도 수집 단계에서는 그대로 저장한다(FR-006)."""
    now = datetime(2026, 10, 20, 6, 0, tzinfo=UTC)
    r = parse("application/a/device/dev1/event/up", _cs(now.isoformat()), now, config)
    assert r.measurements[0]["value"] == 0


def test_non_finite_rejected(config):
    now = datetime(2026, 10, 20, 6, 0, tzinfo=UTC)
    payload = _cs(now.isoformat()).replace(b'"co2": 0', b'"co2": NaN, "t": Infinity')
    r = parse("application/a/device/dev1/event/up", payload, now, config)
    assert not r.measurements and r.ignored == 2


def test_disabled_device(config):
    from collector.config import DeviceOverride
    import dataclasses
    cfg = dataclasses.replace(config, devices={**config.devices, "dev1": DeviceOverride(enabled=False)})
    now = datetime(2026, 10, 20, 6, 0, tzinfo=UTC)
    r = parse("application/a/device/dev1/event/up", _cs(now.isoformat()), now, cfg)
    assert not r.measurements and r.ignored_topic

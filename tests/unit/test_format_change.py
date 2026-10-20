"""형식 변경을 설정만 바꿔 대응한다 (US4, SC-006). 코드(src/)는 그대로 두고 설정 조각만 덧붙인다."""
import copy
import json
from datetime import datetime, timezone

import pytest
import yaml

from collector.config import build_config
from collector.parser import parse
from tests.conftest import CONFIG_PATH, FIXTURES

pytestmark = pytest.mark.unit
FC = FIXTURES / "format_change"


def base_raw():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def patched(name):
    """설정 조각을 기본 설정에 덧붙인다: items는 덮어쓰기, *_add는 목록에 추가."""
    raw = copy.deepcopy(base_raw())
    patch = yaml.safe_load((FC / f"{name}.yaml").read_text(encoding="utf-8"))
    raw["items"].update(patch.get("items", {}))
    raw["spaces"] += patch.get("spaces_add", [])
    raw["sources"] += patch.get("sources_add", [])
    return build_config(raw)


def run(cfg, name):
    out = []
    for line in (FC / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
        m = json.loads(line)
        out.append(parse(m["topic"], m["payload"].encode(), datetime.fromtimestamp(m["t"], timezone.utc), cfg))
    return out


def items(results):
    return [x["item"] for r in results for x in r.measurements]


def test_renamed_field_before_and_after():
    before = run(build_config(base_raw()), "co2_renamed")
    assert "co2" not in items(before) and "co2_ppm" in items(before)   # 비표준 이름으로 새어 들어감
    after = run(patched("co2_renamed"), "co2_renamed")
    assert "co2" in items(after) and "co2_ppm" not in items(after)
    assert all(x["unit"] == "ppm" for r in after for x in r.measurements if x["item"] == "co2")


def test_new_source_before_and_after():
    before = run(build_config(base_raw()), "new_source")
    assert all(r.ignored_topic for r in before)                     # 규칙이 없으면 받지 않음
    after = run(patched("new_source"), "new_source")
    rows = [x for r in after for x in r.measurements]
    assert {x["item"] for x in rows} == {"co2", "temperature", "rssi"}
    assert {x["space_id"] for x in rows} == {"meeting", "lounge"}   # 기존 공간 별칭 + 새 공간
    assert all(x["source"] == "labv2" and x["ts_corrected"] == 0 for x in rows)


def test_new_device_auto_mapped_and_unknown_unmapped(config):
    p = {"deviceInfo": {"devEui": "new-dev", "tags": {"location": "실습실"}}, "rxInfo": [], "object": {"co2": 700}}
    now = datetime.now(timezone.utc)
    r = parse("application/a/device/new-dev/event/up", json.dumps(p).encode(), now, config)
    assert r.measurements[0]["space_id"] == "lab"
    p["deviceInfo"]["tags"]["location"] = "보일러실"
    r = parse("application/a/device/new-dev/event/up", json.dumps(p).encode(), now, config)
    assert r.measurements[0]["space_id"] == "unmapped" and r.devices[0]["mapped_by"] == "none"

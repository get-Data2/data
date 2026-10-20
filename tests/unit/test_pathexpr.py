import json

import pytest

from collector.pathexpr import PathExprError, compile_expr

pytestmark = pytest.mark.unit

CS = {
    "deviceInfo": {"devEui": "24e1", "tags": {"location": "실습실"}},
    "rxInfo": [
        {"gatewayId": "dccc", "nsTime": "2026-10-20T06:42:04.228815318+00:00"},
        {"gatewayId": "9304", "nsTime": "2026-10-20T06:42:04.2247+00:00"},
    ],
    "object": {"co2": 879, "temperature": 24.4},
}
XF = {"ts": 1792478477802, "values": {"대표실-실내기": {"power": False, "current_temperature": 20.7}}}
XF_TOPIC = "xflow/hvacr/idu/4a848cc0/status"


def ev(src, payload, topic=""):
    return compile_expr(src).eval(payload, topic)


def test_nested_key():
    assert ev("deviceInfo.tags.location", CS) == "실습실"
    assert ev("deviceInfo.tags.spot", CS) is None


def test_array_star_returns_list():
    assert ev("rxInfo[*].gatewayId", CS) == ["dccc", "9304"]


def test_array_index():
    assert ev("rxInfo[0].gatewayId", CS) == "dccc"
    assert ev("rxInfo[-1].gatewayId", CS) == "9304"
    assert ev("rxInfo[5].gatewayId", CS) is None


def test_min_compares_times_not_strings():
    # 소수 자릿수가 다른 ISO 문자열: 문자열 비교로는 첫 번째가 작지만 실제로는 두 번째가 이르다
    assert ev("min(rxInfo[*].nsTime)", CS) == "2026-10-20T06:42:04.2247+00:00"


def test_first():
    assert ev("first(rxInfo[*].gatewayId)", CS) == "dccc"


def test_topic_parts():
    assert ev("topic[3]", {}, XF_TOPIC) == "4a848cc0"
    assert ev("topic[-2]", {}, XF_TOPIC) == "4a848cc0"
    assert ev("topic[-3]", {}, XF_TOPIC) == "idu"


def test_values_star_key_and_split():
    assert ev("values.*.$key", XF) == "대표실-실내기"
    assert ev('split(values.*.$key, "-", 0)', XF) == "대표실"
    assert ev("values.*", XF) == {"power": False, "current_temperature": 20.7}


def test_eval_items_keeps_key():
    items = compile_expr("values.*").eval_items(XF)
    assert items == [({"power": False, "current_temperature": 20.7}, "대표실-실내기")]


@pytest.mark.parametrize("bad, reason", [
    ("max2(ts)", "알 수 없는 함수"),
    ("rxInfo[x].a", "배열 첨자"),
    ("min(ts", "')'"),
    ('split(a, "-")', "인자는 3개"),
    ("a b", "남은 문자"),
    ("", "빈 식"),
])
def test_syntax_errors(bad, reason):
    with pytest.raises(PathExprError) as e:
        compile_expr(bad)
    assert reason in str(e.value)


def test_real_fixture_nstime(fixture_messages):
    """실측: 시계가 빠른 게이트웨이(…dccc)를 거친 메시지도 min(nsTime)은 수신 시각과 몇 초 안."""
    from collector.timeutil import parse_time
    e = compile_expr("min(rxInfo[*].nsTime)")
    for m in fixture_messages:
        if not m["topic"].startswith("application/"):
            continue
        t = parse_time(e.eval(json.loads(m["payload"])))
        assert abs(t.timestamp() - m["t"]) < 10

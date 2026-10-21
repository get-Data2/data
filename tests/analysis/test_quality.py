import pytest

from analysis.quality import apply_axis_a
from tests.analysis.conftest import T0

pytestmark = pytest.mark.unit


def test_axis_a_rules(acfg, make_raw):
    raw = make_raw([
        (T0, "lab", "d1", "co2", 65535.0),          # A1
        (T0, "lab", "d1", "co2", 0.0),              # A1
        (T0, "lab", "d2", "temperature", -0.1),     # A2 쌍
        (T0, "lab", "d2", "humidity", 127.5),       # A2 쌍
        (T0, "lab", "d3", "temperature", -0.1),     # 쌍 없는 −0.1 → 실제 영하일 수 있으므로 ok
        (T0, "lab", "d4", "humidity", 101.0),       # A3
        (T0, "lab", "d5", "co2", 10000.0),          # A4 saturated(범위 끝값)
        (T0, "lab", "d7", "co2", 7029.0),           # 회의실 실측 고농도 — 상한 아님 → ok
        (T0, "lab", "d5", "co2", 3200.0),           # 실제 밀집 → ok
        (T0, "lab", "d6", "temperature", 45.0),     # 화재 수준이어도 범위 안 → ok
    ])
    q = apply_axis_a(raw, acfg)
    got = list(zip(q["quality"].astype(str), q["rule"].astype(str)))
    assert got == [("error", "A1"), ("error", "A1"), ("error", "A2"), ("error", "A2"), ("ok", ""),
                   ("error", "A3"), ("saturated", "A4"), ("ok", ""), ("ok", ""), ("ok", "")]
    assert len(q) == len(raw)                         # 원본 행 보존(SC-006)


def test_model_override(acfg, make_raw):
    import dataclasses
    raw_cfg = {**acfg.raw, "axis_a": {**acfg.raw["axis_a"], "models": {"AM103": {"temperature": [0, 50]}}}}
    cfg = dataclasses.replace(acfg, raw=raw_cfg)
    raw = make_raw([(T0, "office", "am", "temperature", 55.0), (T0, "lab", "em", "temperature", 55.0)])
    q = apply_axis_a(raw, cfg, models={"am": "AM103", "em": "EM300-TH"})
    assert list(q["quality"].astype(str)) == ["error", "ok"]


def test_pair_with_1ms_offset(acfg, make_raw):
    """InfluxDB 과거분: 온도·습도 행이 1ms 차이로 저장된 쌍도 오류 코드 쌍이다(실측 83쌍)."""
    from datetime import timedelta
    raw = make_raw([(T0, "office_out", "x", "temperature", -0.1),
                    (T0 + timedelta(milliseconds=1), "office_out", "x", "humidity", 127.5),
                    (T0 + timedelta(seconds=5), "office_out", "y", "temperature", -0.1),
                    (T0 + timedelta(seconds=9), "office_out", "y", "humidity", 127.5)])   # 4초 차이 → 쌍 아님
    q = apply_axis_a(raw, acfg)
    assert list(q["rule"].astype(str)) == ["A2", "A2", "", "A3"]

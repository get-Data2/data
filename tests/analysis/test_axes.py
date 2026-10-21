"""센서 오류(축 A)와 실제 사건(축 B)이 분리되는지 합성 시나리오로 확인(SC-007, SC-008)."""
from datetime import timedelta

import numpy as np
import pytest

from analysis.events import detect
from analysis.quality import apply_axis_a
from analysis.series import sensor_bins, space_series
from tests.analysis.conftest import T0

pytestmark = pytest.mark.unit
STATS = {"spaces": {"lab": {"co2": {"p99.9": 1800.0, "mad": 30.0}, "temperature": {"mad": 0.3},
                            "humidity": {"mad": 1.0}}},
         "sensors": {d: {"co2": {"bias": 0.0, "mad": 30.0}} for d in ("a", "b", "c")}}


def run(acfg, rows):
    from tests.analysis.conftest import raw_frame
    raw = raw_frame(rows)
    q = apply_axis_a(raw, acfg)
    sb = sensor_bins(raw, q, acfg)
    return q, detect(sb, space_series(sb, acfg), acfg, STATS)


def series(dev, item, vals, step_s=60):
    return [(T0 + timedelta(seconds=step_s * i), "lab", dev, item, float(v)) for i, v in enumerate(vals)]


def test_fire_is_event_not_error(acfg):
    n = 120
    temp = list(np.r_[np.full(40, 25.0), np.linspace(25, 45, 30), np.full(n - 70, 45.0)])
    hum = list(np.r_[np.full(40, 50.0), np.linspace(50, 30, 30), np.full(n - 70, 30.0)])
    rows = []
    for d in ("a", "b", "c"):
        rows += series(d, "temperature", temp) + series(d, "humidity", hum)
    q, ev = run(acfg, rows)
    assert (q["quality"].astype(str) == "error").sum() == 0
    assert (ev["kind"] == "B1").sum() >= 1


def test_error_code_pair_is_error_not_event(acfg):
    rows = []
    for d in ("a", "b", "c"):
        rows += series(d, "temperature", [24.0] * 30) + series(d, "humidity", [50.0] * 30)
    rows += [(T0 + timedelta(seconds=60 * 31), "lab", "a", "temperature", -0.1),
             (T0 + timedelta(seconds=60 * 31), "lab", "a", "humidity", 127.5)]
    q, ev = run(acfg, rows)
    assert (q["rule"].astype(str) == "A2").sum() == 2
    assert ev.empty


def test_single_spike_is_suspect(acfg):
    # 1분짜리 튐은 5분 칸 평균에서 1/5로 희석된다(+800 → +160, 기준 MAD×10=300 미만).
    # 칸 수준 B0는 큰 튐만 잡는다 — 작은 튐은 과제 4에서 원본(1분) 수준으로 탐지해야 한다.
    base = [500.0] * 30
    spike = base[:]
    spike[12] = 3000.0
    rows = series("a", "co2", base) + series("b", "co2", [510.0] * 30) + series("c", "co2", spike)
    q, ev = run(acfg, rows)
    assert list(ev["kind"]) == ["B0"] and ev["devices"].iloc[0] == "c"


def test_crowding_two_sensors(acfg):
    vals = [800.0] * 20 + [3000.0] * 15 + [800.0] * 10
    rows = series("a", "co2", vals) + series("b", "co2", vals) + series("c", "co2", [800.0] * 45)
    q, ev = run(acfg, rows)
    b3 = ev[ev["kind"] == "B3"]
    assert len(b3) == 1 and "3000" in b3["evidence"].iloc[0]


def test_axes_independent(acfg):
    """SC-008: 축 B는 축 A 설정을 읽지 않는다 — 축 A 설정을 바꿔도 같은 센서 칸이면 사건 결과가 같다."""
    import dataclasses
    vals = [800.0] * 20 + [3000.0] * 15
    rows = series("a", "co2", vals) + series("b", "co2", vals)
    from tests.analysis.conftest import raw_frame
    raw = raw_frame(rows)
    sb = sensor_bins(raw, apply_axis_a(raw, acfg), acfg)
    ev1 = detect(sb, space_series(sb, acfg), acfg, STATS)
    other = dataclasses.replace(acfg, raw={**acfg.raw, "axis_a": {**acfg.raw["axis_a"], "error_codes": []}})
    ev2 = detect(sb, space_series(sb, other), other, STATS)
    assert ev1.drop(columns="rule_version").equals(ev2.drop(columns="rule_version"))

from datetime import datetime, timezone

import pytest

from collector.report import Data, Event, build_report, cause_hint, coverage, gaps

pytestmark = pytest.mark.unit

S = 29_000_000          # 임의 epoch 분(시작)
E = S + 60              # 60분 구간


def test_coverage_95():
    mins = set(range(S, E)) - {S + 10, S + 11, S + 40}
    assert coverage(mins, S, E) == pytest.approx(57 / 60)


def test_coverage_ignores_outside():
    assert coverage({S - 1, E, E + 5}, S, E) == 0.0


def test_coverage_union_of_devices():
    a = set(range(S, S + 30))
    b = set(range(S + 30, E))
    assert coverage(a | b, S, E) == 1.0


def test_coverage_long_period():
    # 10분 주기 장치: 기대 칸 6개, 그중 5개에 값
    mins = {S + 1, S + 12, S + 23, S + 34, S + 45}
    assert coverage(mins, S - S % 10, S - S % 10 + 60, 600) == pytest.approx(5 / 6)


def test_gaps_min_length():
    mins = set(range(S, E)) - {S + 5} - set(range(S + 20, S + 23))
    assert gaps(mins, S, E, 60, 2) == [(S + 20, S + 23)]       # 1분 빈칸은 무시
    assert gaps(mins, S, E, 60, 1) == [(S + 5, S + 6), (S + 20, S + 23)]


def test_gaps_edges():
    mins = set(range(S + 3, E - 4))
    assert gaps(mins, S, E, 60, 2) == [(S, S + 3), (E - 4, E)]
    assert gaps(set(), S, E) == [(S, E)]


def dev_minutes(missing_a=(), missing_b=()):
    full = set(range(S, E))
    return {"a": full - set(missing_a), "b": full - set(missing_b), "c": set(full)}


def test_cause_collector_down():
    stats = set(range(S, E)) - set(range(S + 10, S + 20))
    h = cause_hint((S + 10, S + 20), None, dev_minutes(), stats, [Event(S + 10.1, "stop")])
    assert h.startswith("수집기 미가동") and "stop" in h


def test_cause_broker():
    gap = range(S + 10, S + 20)
    dm = {k: v - set(gap) for k, v in dev_minutes().items()}
    h = cause_hint((S + 10, S + 20), None, dm, set(range(S, E)), [Event(S + 10.2, "mqtt_disconnected")])
    assert "브로커·네트워크" in h


def test_cause_source_side():
    gap = range(S + 10, S + 20)
    dm = {k: v - set(gap) for k, v in dev_minutes().items()}
    assert "원천 측" in cause_hint((S + 10, S + 20), None, dm, set(range(S, E)), [])


def test_cause_single_device_battery():
    dm = dev_minutes(missing_a=range(S + 10, S + 50))
    h = cause_hint((S + 10, S + 50), "a", dm, set(range(S, E)), [], battery=1)
    assert "해당 장치만" in h and "배터리 1%" in h


def test_build_report(config):
    start = datetime.fromtimestamp(S * 60, timezone.utc)
    end = datetime.fromtimestamp(E * 60, timezone.utc)
    last = datetime.fromtimestamp((E - 1) * 60, timezone.utc).replace(tzinfo=None)
    d = Data(start, end)
    d.spaces = {"lab": ("실습실", "classroom"), "unmapped": ("미지정", "unmapped")}
    d.devices = {
        "a": {"device_id": "a", "model": "EM500-CO2", "space_id": "lab", "raw_location": "실습실",
              "mapped_by": "alias", "expected_period_s": 60, "last_seen": last},
        "z": {"device_id": "z", "model": "?", "space_id": "unmapped", "raw_location": "창고",
              "mapped_by": "none", "expected_period_s": 60, "last_seen": last},
    }
    mins = set(range(S, E)) - set(range(S + 20, S + 25))
    d.device_minutes = {"a": mins, "z": set(range(S, E))}
    d.device_counts = {"a": len(mins)}
    d.space_item_minutes = {("lab", "co2"): mins}
    d.series = {("lab", "co2"): {"count": len(mins), "last": last, "devices": 1}}
    d.stats_minutes = set(range(S, E))
    text = build_report(d, config, now=end)
    assert "실데이터" in text and "공간 미지정(원천 표기 '창고')" in text
    assert "| 실습실(lab) | co2 | 1 | 55 |" in text and "91.7%" in text
    assert "| a | 실습실(lab) |" in text and "5분" in text and "해당 장치만 결측" in text


def test_observed_period():
    from collector.report import observed_period_min
    assert observed_period_min({S, S + 10, S + 20, S + 31}) == 10
    assert observed_period_min({S, S + 1}) is None

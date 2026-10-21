import json

import numpy as np
import pandas as pd
import pytest

from anomaly.alarms import (ThresholdMissingError, alarms_from, choose_threshold, combine, freeze_thresholds,
                            require_thresholds, to_quantile)
from anomaly.evaluate import criteria, false_alarms, match, scorecard

pytestmark = pytest.mark.unit
T = pd.Timestamp("2026-06-01 00:00", tz="UTC")


def pts(space, offsets_min, q=0.999, cause="D1/co2"):
    return pd.DataFrame({"space_id": space, "t": [T + pd.Timedelta(minutes=m) for m in offsets_min], "q": q, "cause": cause})


def test_merge_within_30min():
    a = alarms_from(pts("lab", [0, 5, 25, 60]), 0.99, 30)
    assert len(a) == 2 and a.iloc[0]["end"] == T + pd.Timedelta(minutes=25)


def test_quantile():
    q = to_quantile(pd.Series([0.5, 2.0, 10.0]), pd.Series(np.arange(1, 5, dtype=float)))
    assert list(q) == [0.0, 0.5, 1.0]


def test_threshold_uses_validation_only():
    """임계값은 검증 점수만으로 정해진다: 시험 점수를 어떻게 바꿔도 같다(누수 없음)."""
    rng = np.random.default_rng(0)
    val = pd.DataFrame({"space_id": "lab", "t": pd.date_range(T, periods=5000, freq="5min"), "q": rng.random(5000),
                        "cause": "D1/co2"})
    cands = 1 - np.logspace(-2, -5, 20)
    p1, rows1, _ = choose_threshold(val, cands, 0.5, 17, 30)
    p2, rows2, _ = choose_threshold(val.copy(), cands, 0.5, 17, 30)
    assert p1 == p2 and rows1 == rows2
    assert p1["val_per_day"] <= 0.5


def test_threshold_file_gate(tmp_path):
    p = tmp_path / "thresholds.json"
    with pytest.raises(ThresholdMissingError):
        require_thresholds(p)
    freeze_thresholds(p, {"ae": {"threshold_q": 0.999}})
    require_thresholds(p)
    b = json.loads(p.read_text())
    b["ae"]["threshold_q"] = 0.9
    p.write_text(json.dumps(b))
    with pytest.raises(ThresholdMissingError):
        require_thresholds(p)


def test_metrics_by_hand():
    faults = pd.DataFrame([
        {"fault_id": 1, "kind": "F1", "space_id": "lab", "device_id": "D1", "items": "co2", "start": T,
         "end": T + pd.Timedelta(minutes=5), "match_end": T + pd.Timedelta(minutes=35)},
        {"fault_id": 2, "kind": "F3", "space_id": "lab", "device_id": "D2", "items": "co2", "start": T + pd.Timedelta(hours=10),
         "end": T + pd.Timedelta(hours=14), "match_end": T + pd.Timedelta(hours=34)},
        {"fault_id": 3, "kind": "F4", "space_id": "office", "device_id": None, "items": "temperature",
         "start": T + pd.Timedelta(hours=2), "end": T + pd.Timedelta(hours=4), "match_end": T + pd.Timedelta(hours=4, minutes=30)},
    ])
    alarms = pd.DataFrame([
        {"space_id": "lab", "start": T + pd.Timedelta(minutes=5), "end": T + pd.Timedelta(minutes=5), "peak_q": 1, "cause": "D1/co2"},
        {"space_id": "lab", "start": T + pd.Timedelta(hours=30), "end": T + pd.Timedelta(hours=30), "peak_q": 1, "cause": "D3/co2"},
        {"space_id": "office", "start": T + pd.Timedelta(hours=8), "end": T + pd.Timedelta(hours=8), "peak_q": 1, "cause": "O1/co2"},
        {"space_id": "meeting", "start": T, "end": T, "peak_q": 1, "cause": "M1/co2"},
    ])
    m = match(alarms, faults)
    assert list(m["fault_id"]) == [1, 2, 0, 0]
    card = scorecard(m, faults).set_index("kind")
    assert card.loc["F1", "detected"] == 1 and card.loc["F1", "delay_median_min"] == 5
    assert card.loc["F3", "detected"] == 1 and card.loc["F3", "delay_median_min"] == 20 * 60     # 24시간 인정 구간
    assert card.loc["F3", "cause_acc"] == 0                                                         # 다른 센서를 지목
    assert card.loc["F4", "detected"] == 0
    assert card.loc["전체", "rate"] == pytest.approx(2 / 3)
    ev = pd.DataFrame({"space_id": ["meeting"], "start": [T - pd.Timedelta(minutes=10)], "end": [T + pd.Timedelta(minutes=10)]})
    fa = false_alarms(m, ev, days=2)
    assert fa["false_alarms"] == 2 and fa["per_day"] == 1.0 and fa["real_event_overlap"] == 1
    crit = criteria(scorecard(m, faults), fa)
    assert list(crit["판정"]) == ["❌", "✅"]


def test_combine_union():
    a = alarms_from(pts("lab", [0], 0.995, "A/co2"), 0.99, 30)
    b = alarms_from(pts("lab", [10, 200], 0.999, "B/co2"), 0.99, 30)
    c = combine([a, b], 30)
    assert len(c) == 2 and c.iloc[0]["cause"] == "B/co2"


def test_adaptive_quantile_is_causal():
    """적응 임계값: 그 날 분위수는 이전 날 점수로만 정해진다(뒤에 큰 점수가 와도 앞날 q가 그대로)."""
    from anomaly.adaptive import rolling_points
    t = pd.date_range("2026-06-01", periods=288 * 10, freq="5min", tz="Asia/Seoul").tz_convert("UTC")
    base = pd.DataFrame({"space_id": "lab", "t": t, "score": np.random.default_rng(0).random(len(t)), "cause": "D/co2"})
    late = base.copy()
    late.loc[late.index[-288:], "score"] = 1e6
    a, b = rolling_points(base, 3), rolling_points(late, 3)
    day9 = a["t"] < t[-288]
    assert np.allclose(a.loc[day9, "q"].to_numpy(), b.loc[b["t"] < t[-288], "q"].to_numpy())

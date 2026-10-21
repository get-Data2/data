import numpy as np
import pandas as pd
import pytest

from anomaly.data import sensor_stats
from anomaly.rules import spike, stuck

pytestmark = pytest.mark.unit


def test_single_spike_detected_with_5min_delay(ncfg, boards):
    w = boards["lab"].copy()
    st = sensor_stats(w)
    c = "L1/co2"
    i = 2000
    w.iloc[i, w.columns.get_loc(c)] += 900
    r = spike(w, st, ncfg)
    hit = r[r["cause"] == c]
    assert len(hit) == 1 and hit["t"].iloc[0] == w.index[i + 1]


def test_class_start_rise_is_not_spike(ncfg, boards):
    """수업 시작의 단조 상승은 튐이 아니다(합성 원본에 튐 없음)."""
    w = boards["lab"]
    assert spike(w, sensor_stats(w), ncfg).empty


def test_stuck_with_changing_neighbours(ncfg, boards):
    w = boards["lab"].copy()
    # 수업 중 2시간 L1 전 항목 고정
    k = w.index.tz_convert("Asia/Seoul")
    day = np.where((k.weekday < 5) & (k.hour == 9) & (k.minute == 0))[0][0]
    for c in ("L1/co2", "L1/temperature", "L1/humidity"):
        w.iloc[day:day + 24, w.columns.get_loc(c)] = w.iloc[day, w.columns.get_loc(c)]
    r = stuck(w, ncfg, ncfg.acfg)
    assert set(r["cause"]) >= {"L1/co2"}
    assert (r["t"] >= w.index[day + 12]).all()                # 1시간 지나야 판정


def test_night_quantized_humidity_not_stuck(ncfg, boards):
    """밤에는 이웃도 안 변하므로 습도 양자화 반복을 값 고정으로 보지 않는다."""
    w = boards["lab"].copy()
    k = w.index.tz_convert("Asia/Seoul")
    night = np.where((k.hour >= 1) & (k.hour < 5))[0]
    for c in [c for c in w.columns if c.endswith("/humidity")]:
        w.iloc[night, w.columns.get_loc(c)] = 40.0
    r = stuck(w, ncfg, ncfg.acfg)
    assert not r["cause"].str.endswith("/humidity").any()

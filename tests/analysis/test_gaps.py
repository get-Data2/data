from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from analysis.gaps import fill_bins, find_gaps, interpolation_test
from tests.analysis.conftest import T0

pytestmark = pytest.mark.unit
nan = np.nan


def test_find_gaps_by_own_period():
    ts = pd.Series([T0, T0 + timedelta(seconds=110), T0 + timedelta(seconds=260)])
    g = find_gaps(ts, 60)
    assert len(g) == 1 and g["minutes"].iloc[0] == pytest.approx(150 / 60)   # 110초는 결측 아님, 150초는 결측
    ts10 = pd.Series([T0, T0 + timedelta(seconds=900)])
    assert find_gaps(ts10, 600).empty                                       # 10분 장치의 15분 간격은 결측 아님


def test_fill_short_gap_interpolates():
    v, f = fill_bins(np.array([10, nan, nan, 40.0]), 5, 60, 15)
    assert list(v) == [10, 20, 30, 40] and list(f) == [False, True, True, False]


def test_long_gap_left_empty():
    v, f = fill_bins(np.array([10, nan, nan, nan, nan, 40.0]), 5, 60, 15)    # 20분 > 15분
    assert np.isnan(v[1:5]).all() and not f.any()


def test_long_period_device_carries_within_period():
    # 10분 주기 장치를 5분 칸으로: 사이 칸은 직전 값(결측 아님)
    v, f = fill_bins(np.array([1, nan, 2, nan, 3.0]), 5, 600, 15)
    assert list(v) == [1, 1, 2, 2, 3] and list(f) == [False, True, False, True, False]


def test_edges_not_extrapolated():
    v, _ = fill_bins(np.array([nan, 5, nan]), 5, 60, 15)
    assert np.isnan(v[0]) and np.isnan(v[2])


def test_interpolation_test_table():
    x = np.sin(np.arange(2000) / 20) * 100 + 500
    t = interpolation_test(x, 5, [5, 15, 60], n_trials=50)
    assert list(t["gap_min"]) == [5, 15, 60] and (t["trials"] == 50).all()
    assert t["mae"].is_monotonic_increasing                                 # 길수록 오차가 커진다

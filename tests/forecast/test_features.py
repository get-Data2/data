from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from forecast.features import booking_features, estimate_bookings, schedule_features, time_features

pytestmark = pytest.mark.unit
KST = timezone(timedelta(hours=9))


def kst_range(start, n):
    return pd.date_range(start.astimezone(timezone.utc), periods=n, freq="5min")


def test_holiday_is_offday(fcfg):
    t = kst_range(datetime(2026, 5, 5, 10, 0, tzinfo=KST), 1)          # 어린이날(화)
    assert time_features(t, fcfg.acfg)[0, 4] == 1
    sch = schedule_features(t, "lab", fcfg.acfg)[0]
    assert sch[3] == 1 and sch[0] == 0


def test_schedule_state_and_next_change(fcfg):
    t = kst_range(datetime(2026, 3, 4, 8, 50, tzinfo=KST), 1)          # 수 08:50
    s = schedule_features(t, "lab", fcfg.acfg)[0]
    assert s[2] == 1 and s[4] == pytest.approx(10 / 60)
    t = kst_range(datetime(2026, 3, 4, 12, 0, tzinfo=KST), 1)          # 점심
    s = schedule_features(t, "lab", fcfg.acfg)[0]
    assert s[1] == 1 and s[4] == pytest.approx(1.0)
    m = schedule_features(t, "meeting", fcfg.acfg)[0]                  # 일정 없는 공간
    assert m[:4].sum() == 0 and m[4] == 3.0


def test_future_features_independent_of_sensors(fcfg, series):
    """특징은 시각만으로 정해진다: 센서값을 바꿔도 같다(미래 칸에 써도 누수 아님)."""
    from forecast.data import build_arrays
    nb = pd.DataFrame(columns=["space_id", "start", "end", "source"])
    a = build_arrays(fcfg, series, bookings=nb)
    s2 = series.copy()
    s2["value"] *= 3
    b = build_arrays(fcfg, s2, bookings=nb)
    for k in a.cov:
        np.testing.assert_array_equal(a.cov[k], b.cov[k])


def test_estimate_booking_from_rise():
    t = kst_range(datetime(2026, 3, 4, 13, 0, tzinfo=KST), 48)          # 13:00 ~ 17:00, 10분 주기
    v = np.full(48, 500.0)
    v[6:18] = np.linspace(500, 1300, 12)                                # 13:30부터 1시간 상승
    v[18:24] = 1300
    v[24:30] = np.linspace(1300, 700, 6)                                # 15:00부터 하락
    s = pd.Series(v, index=t)
    s[1::2] = np.nan
    rule = {"rise_ppm": 300, "rise_window_min": 30, "end_drop_ppm_per_10min": 50, "round_min": 15, "min_len_min": 30}
    df = estimate_bookings(s, rule)
    assert len(df) == 1
    st, en = df.iloc[0]["start"].tz_convert(KST), df.iloc[0]["end"].tz_convert(KST)
    assert st.strftime("%H:%M") == "13:30" and en.strftime("%H:%M") == "15:00"


def test_booking_features():
    t = kst_range(datetime(2026, 3, 4, 13, 0, tzinfo=KST), 4)
    iv = pd.DataFrame({"start": [t[2]], "end": [t[2] + pd.Timedelta(hours=1)]})
    f = booking_features(t, iv)
    assert list(f[:, 0]) == [0, 0, 1, 1]
    assert f[0, 1] == pytest.approx(10 / 60) and f[2, 2] == pytest.approx(1.0)

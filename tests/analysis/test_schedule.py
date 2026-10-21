from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from analysis.config import KST
from analysis.schedule import events_for, mismatch_days, reaction_lags

pytestmark = pytest.mark.unit


def test_events_weekday_and_holiday(acfg):
    ev = events_for(date(2026, 10, 8), date(2026, 10, 10), acfg)      # 목 / 금 한글날 / 토
    lab = ev[ev.space_id == "lab"]
    assert set(lab["date"]) == {date(2026, 10, 8)}
    assert list(lab.sort_values("t")["hhmm"]) == ["09:00", "11:30", "13:00", "18:00"]
    assert list(lab.sort_values("t")["name"]) == ["수업 시작", "점심 시작", "점심 끝(오후 시작)", "퇴실"]


def series_day(d: date, rise_at: str | None, level=420.0, rise=200.0):
    idx = pd.date_range(datetime(d.year, d.month, d.day, 0, 0, tzinfo=KST), periods=288, freq="5min").tz_convert("UTC")
    v = np.full(288, level)
    if rise_at:
        k = idx.tz_convert(KST)
        m = (k.hour * 60 + k.minute) >= int(rise_at[:2]) * 60 + int(rise_at[3:])
        v[m] = level + rise
    return pd.Series(v, index=idx)


def test_reaction_lag_and_no_reaction(acfg):
    d = date(2026, 10, 8)
    ev = events_for(d, d, acfg)
    ev = ev[(ev.space_id == "lab") & (ev.hhmm == "09:00")]
    lag = reaction_lags(series_day(d, "09:10"), ev, acfg)
    assert lag["lag_min"].iloc[0] == 10
    none = reaction_lags(series_day(d, None), ev, acfg)
    assert np.isnan(none["lag_min"].iloc[0]) and bool(none["measured"].iloc[0])


def test_mismatch_days(acfg):
    thu, sat = date(2026, 10, 8), date(2026, 10, 10)
    s = pd.concat([series_day(thu, None), series_day(date(2026, 10, 9), None), series_day(sat, "14:00")])
    ev = events_for(thu, sat, acfg)
    lags = reaction_lags(s, ev[ev.space_id == "lab"], acfg)
    mm = mismatch_days(s, lags, thu, sat, acfg)
    assert list(mm["type"]) == ["수업일 무반응", "비수업일 사용(주말)"]
    assert list(mm["date"]) == [thu, sat]


def test_weekend_midnight_residue_is_not_use(acfg):
    """금요일 밤 남은 CO₂가 토요일 자정 넘어 빠지는 것은 주말 사용이 아니다."""
    sat = date(2026, 10, 10)
    idx = pd.date_range(datetime(2026, 10, 10, 0, 0, tzinfo=KST), periods=288, freq="5min").tz_convert("UTC")
    v = np.full(288, 450.0)
    v[:24] = np.linspace(700, 450, 24)           # 00:00~02:00 감쇠
    s = pd.Series(v, index=idx)
    mm = mismatch_days(s, pd.DataFrame(columns=["kind", "hhmm", "measured", "lag_min", "date", "name"]), sat, sat, acfg)
    assert mm.empty

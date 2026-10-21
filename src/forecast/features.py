"""시간·공휴일·일과표·예약 특징과 회의실 추정 예약 (research R6·R10).

모두 "그 칸의 시각"만으로 정해지는 값이라 t0 이후 칸에도 쓸 수 있다(미래에 알 수 있는 정보).
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from analysis.config import KST, WEEKDAYS
from analysis.schedule import schedule_rows

TIME_COLS = ["hod_sin", "hod_cos", "dow_sin", "dow_cos", "offday"]
SCHED_COLS = ["sch_on", "sch_lunch", "sch_off", "sch_offday", "sch_next_h"]
BOOK_COLS = ["bk_on", "bk_to_start_h", "bk_to_end_h"]
CAP_H = 3.0


def _kst(t: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return t.tz_convert(KST)


def offday_mask(t: pd.DatetimeIndex, acfg) -> np.ndarray:
    k = _kst(t)
    kr = holidays.KR(years=range(k.year.min(), k.year.max() + 1))
    hol = pd.Index(k.date).isin(list(kr.keys())) if acfg["schedule"].get("exclude_holidays") else np.zeros(len(k), bool)
    return np.asarray((k.weekday >= 5) | hol)


def time_features(t: pd.DatetimeIndex, acfg) -> np.ndarray:
    k = _kst(t)
    h = (k.hour + k.minute / 60) / 24 * 2 * np.pi
    w = k.weekday / 7 * 2 * np.pi
    return np.column_stack([np.sin(h), np.cos(h), np.sin(w), np.cos(w), offday_mask(t, acfg)]).astype(np.float32)


def _hm(s: str) -> int:
    return int(s[:2]) * 60 + int(s[3:])


def schedule_features(t: pd.DatetimeIndex, space: str, acfg) -> np.ndarray:
    """상태 원-핫(일정 중 / 점심(일정 사이) / 일과 외 / 비수업일) + 다음 변화까지 남은 시간(시간, 상한 3).
    일정이 없는 공간은 상태 0, 남은 시간 상한."""
    out = np.zeros((len(t), len(SCHED_COLS)), np.float32)
    out[:, 4] = CAP_H
    rows = schedule_rows(acfg)
    rows = rows[rows["space"] == space]
    if rows.empty:
        return out
    k = _kst(t)
    mins = np.asarray(k.hour * 60 + k.minute)
    wd = np.asarray(k.weekday)
    off = offday_mask(t, acfg)
    for d in range(7):
        dayrows = sorted((_hm(r.start), _hm(r.end)) for r in rows.itertuples() if WEEKDAYS[d] in r.days)
        m = (wd == d) & ~off
        if not dayrows or not m.any():
            continue
        bounds = sorted({x for a, b in dayrows for x in (a, b)})
        mm = mins[m]
        on = np.zeros(len(mm), bool)
        for a, b in dayrows:
            on |= (mm >= a) & (mm < b)
        lunch = ~on & (mm >= dayrows[0][0]) & (mm < dayrows[-1][1])
        idx = np.where(m)[0]
        out[idx, 0] = on
        out[idx, 1] = lunch
        out[idx, 2] = ~on & ~lunch
        nb = np.array([min([x - v for x in bounds if x > v], default=10**6) for v in mm], np.float32)
        out[idx, 4] = np.minimum(nb / 60, CAP_H)
    out[off, 3] = 1
    return out


def booking_features(t: pd.DatetimeIndex, intervals: pd.DataFrame) -> np.ndarray:
    """intervals: start·end(UTC). 예약 중 여부, 다음 시작까지, 현재 예약 끝까지(시간, 상한 3)."""
    out = np.zeros((len(t), len(BOOK_COLS)), np.float32)
    out[:, 1] = CAP_H
    if intervals is None or intervals.empty:
        return out
    tv = t.as_unit("ns").asi8
    st = np.sort(pd.DatetimeIndex(intervals["start"]).as_unit("ns").asi8)
    iv = intervals.sort_values("start")
    s_arr, e_arr = pd.DatetimeIndex(iv["start"]).as_unit("ns").asi8, pd.DatetimeIndex(iv["end"]).as_unit("ns").asi8
    ns_h = 3600 * 10**9
    i = np.searchsorted(s_arr, tv, side="right") - 1          # 시작이 t 이하인 마지막 예약
    on = (i >= 0) & (tv < e_arr[np.clip(i, 0, None)])
    out[:, 0] = on
    out[on, 2] = np.minimum((e_arr[i[on]] - tv[on]) / ns_h, CAP_H)
    j = np.searchsorted(st, tv, side="right")                  # t 다음 시작
    has = j < len(st)
    out[has, 1] = np.minimum((st[np.clip(j[has], 0, len(st) - 1)] - tv[has]) / ns_h, CAP_H)
    return out


def estimate_bookings(co2: pd.Series, rule: dict) -> pd.DataFrame:
    """회의실 관측 CO₂(UTC 5분 칸, 빈 칸 NaN)에서 회의 구간을 추정한다(research R10).
    시작: 앞으로 rise_window_min 안에 rise_ppm 이상 오르는 첫 칸. 끝: 최고점 뒤 10분 하락이 end_drop 이상인 첫 칸."""
    s = co2.dropna()
    if s.empty:
        return pd.DataFrame(columns=["start", "end"])
    rnd = pd.Timedelta(minutes=rule["round_min"])
    win = pd.Timedelta(minutes=rule["rise_window_min"])
    t, v = s.index, s.to_numpy()
    out, i = [], 0
    while i < len(t):
        j = np.searchsorted(t, t[i] + win, side="right")
        if j > i + 1 and v[i + 1:j].max() - v[i] >= rule["rise_ppm"]:
            start = t[i]
            k = i
            peak = i
            # 최고점을 따라가다가 그 칸에서 10분 하락이 기준 이상이면 그 칸이 끝(하락 시작)
            while k + 1 < len(t) and (t[k + 1] - t[k]) <= pd.Timedelta(minutes=30):
                k += 1
                if v[k] >= v[peak]:
                    peak = k
                ten = np.searchsorted(t, t[k] + pd.Timedelta(minutes=10))
                if k >= peak and ten < len(t) and v[k] - v[ten] >= rule["end_drop_ppm_per_10min"]:
                    break
            end = t[k]
            st_r, en_r = start.floor(rnd), end.ceil(rnd)
            if en_r - st_r >= pd.Timedelta(minutes=rule["min_len_min"]):
                out.append((st_r, en_r))
            i = max(k + 1, i + 1)
        else:
            i += 1
    df = pd.DataFrame(out, columns=["start", "end"])
    if len(df) > 1:   # 겹치는 구간 합치기
        df = df.sort_values("start").reset_index(drop=True)
        merged = [list(df.iloc[0])]
        for a, b in df.iloc[1:].itertuples(index=False):
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        df = pd.DataFrame(merged, columns=["start", "end"])
    return df

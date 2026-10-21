"""결측 판정과 짧은 결측 보간(spec "기록 주기와 결측").

결측 = 같은 장치의 연속 두 보고 간격이 자기 기대 주기 × missing_factor(2)를 넘는 것.
분석 칸(5분)에서는: 주기가 칸보다 긴 장치의 빈 칸은 결측이 아니라 자기 주기 안에서 직전 값을 쓰고,
그 밖의 빈 칸 연속 길이가 short_max_min(15분) 이하면 선형 보간, 넘으면 비워 둔다.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def find_gaps(ts: pd.Series, period_s: int, factor: float = 2.0) -> pd.DataFrame:
    """정렬된 보고 시각 → 결측 구간 [start, end, minutes]."""
    t = pd.to_datetime(ts).sort_values().reset_index(drop=True)
    d = t.diff().dt.total_seconds()
    m = d > period_s * factor
    return pd.DataFrame({"start": t.shift(1)[m].to_numpy(), "end": t[m].to_numpy(),
                         "minutes": (d[m] / 60).to_numpy()})


def fill_bins(values: np.ndarray, interval_min: int, period_s: int, short_max_min: int) -> tuple[np.ndarray, np.ndarray]:
    """칸 값 배열(빈 칸 NaN) → (채운 값, 채움 표시).

    1) 주기가 칸보다 길면 자기 주기 안의 빈 칸은 직전 값(결측 아님)
    2) 남은 빈 칸 연속이 short_max_min 이하이고 앞뒤에 값이 있으면 선형 보간
    """
    v = values.astype(float).copy()
    filled = np.zeros(len(v), dtype=bool)
    carry = max(0, math.ceil(period_s / 60 / interval_min) - 1)
    if carry:
        last, last_i = np.nan, -10**9
        for i in range(len(v)):
            if not np.isnan(v[i]):
                last, last_i = v[i], i
            elif i - last_i <= carry and not np.isnan(last):
                v[i] = last
                filled[i] = True
    isn = np.isnan(v)
    if isn.any():
        max_bins = short_max_min // interval_min
        i, n = 0, len(v)
        while i < n:
            if not isn[i]:
                i += 1
                continue
            j = i
            while j < n and isn[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= max_bins:
                v[i:j] = np.interp(np.arange(i, j), [i - 1, j], [v[i - 1], v[j]])
                filled[i:j] = True
            i = j
    return v, filled


def interpolation_test(series: np.ndarray, interval_min: int, lengths_min: list[int], n_trials: int = 300,
                       seed: int = 0) -> pd.DataFrame:
    """결측 없는 구간에 인위 결측을 넣고 선형 보간 오차를 잰다(FR-003a).

    비교 기준 noise = 이웃 칸 차이 절댓값의 중앙값(칸 사이 자연 변동).
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(series, dtype=float)
    ok = ~np.isnan(v)
    noise = float(np.nanmedian(np.abs(np.diff(v))))
    rows = []
    for L in lengths_min:
        k = max(1, L // interval_min)
        errs = []
        cand = [s for s in range(1, len(v) - k - 1) if ok[s - 1:s + k + 1].all()]
        if not cand:
            rows.append({"gap_min": L, "trials": 0, "mae": np.nan, "p90": np.nan, "noise": noise})
            continue
        for s in rng.choice(cand, size=min(n_trials, len(cand)), replace=False):
            est = np.interp(np.arange(s, s + k), [s - 1, s + k], [v[s - 1], v[s + k]])
            errs.append(np.mean(np.abs(est - v[s:s + k])))
        rows.append({"gap_min": L, "trials": len(errs), "mae": float(np.mean(errs)),
                     "p90": float(np.quantile(errs, 0.9)), "noise": noise})
    return pd.DataFrame(rows)

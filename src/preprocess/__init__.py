"""학습(과제 2·3)과 서비스(과제 8)가 함께 쓰는 전처리 (specs/008 research R2·R3).

지금은 과제 2·3의 기존 함수를 감싸 그대로 부른다. 학습 코드(`analysis`·`forecast`)를 이 모듈로 옮기는 정리는
과제 2 평가 시점 해시(2f878c11…) 회귀 확인과 함께 따로 한다(specs/008 tasks T005·T006).

서비스 경로의 차이는 하나뿐이다: 학습 사본은 짧은 결측(≤15분)을 앞뒤 값으로 보간했지만, 서비스는 미래 값을 모르므로
창 끝의 짧은 결측은 보간하지 않는다. 과제 3 입력은 관측 칸(filled=False)만 쓰므로 모델 입력에는 영향이 없다.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from analysis.quality import apply_axis_a
from analysis.series import aggregate_bins, space_series
from forecast.data import ITEMS, _causal
from forecast.features import BOOK_COLS, SCHED_COLS, schedule_features, time_features

__all__ = ["ITEMS", "axis_a", "closed_bin_end", "sensor_bins_closed", "space_bins", "causal", "covariates",
           "BOOK_COLS", "SCHED_COLS"]


def axis_a(raw: pd.DataFrame, acfg, models: dict[str, str] | None = None) -> pd.DataFrame:
    """축 A 품질 표시(과제 2 규칙 그대로)."""
    return apply_axis_a(raw, acfg, models)


def closed_bin_end(now: pd.Timestamp, interval_min: int = 5) -> pd.Timestamp:
    """지금 시각에서 마지막으로 닫힌 칸의 끝. 10:03:20 → 10:00 (칸 [09:55, 10:00)까지 사용)."""
    return pd.Timestamp(now).floor(f"{interval_min}min")


def _carry(values: np.ndarray, interval_min: int, period_s: int) -> tuple[np.ndarray, np.ndarray]:
    """주기가 칸보다 긴 장치: 자기 주기 안의 빈 칸은 직전 값(결측 아님, filled 표시). 과제 2 gaps.fill_bins의 1단계와 같다.
    2단계(짧은 결측 보간)는 미래 값이 필요하므로 서비스에서는 하지 않는다."""
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
    return v, filled


def sensor_bins_closed(raw: pd.DataFrame, quality: pd.DataFrame, acfg, end: pd.Timestamp,
                       interval_min: int | None = None) -> pd.DataFrame:
    """닫힌 칸까지만의 센서 칸 평균(축 A error 제외) + 주기 안 직전값.
    [device_id, space_id, item, bin, value, filled, saturated] — 과제 2 sensor_5min과 같은 열."""
    interval_min = interval_min or acfg["interval"]["minutes"]
    keep = raw["ts"] < end
    agg = aggregate_bins(raw[keep], quality[keep], interval_min)
    cols = ["device_id", "space_id", "item", "bin", "value", "filled", "saturated"]
    if agg.empty:
        return pd.DataFrame(columns=cols)
    out = []
    for (dev, item), s in agg.groupby(["device_id", "item"], observed=True):
        s = s.set_index("bin").sort_index()
        full = pd.date_range(s.index.min(), end - pd.Timedelta(minutes=interval_min), freq=f"{interval_min}min")
        v, filled = _carry(s["value"].reindex(full).to_numpy(), interval_min, acfg.period_s(str(dev)))
        ok = ~np.isnan(v)
        space = s["space_id"].astype(str).reindex(full).ffill().bfill().to_numpy()
        sat = s["saturated"].reindex(full).fillna(False).astype(bool).to_numpy()
        out.append(pd.DataFrame({"device_id": str(dev), "space_id": space[ok], "item": str(item), "bin": full[ok],
                                 "value": v[ok], "filled": filled[ok], "saturated": sat[ok]}))
    return pd.concat(out, ignore_index=True)[cols]


def space_bins(sensor: pd.DataFrame, acfg) -> pd.DataFrame:
    """공간 대표값(센서 중앙값, 제외 장치 반영) — 과제 2 space_series 그대로."""
    if sensor.empty:
        return pd.DataFrame(columns=["space_id", "item", "bin", "value", "n_sensors", "n_observed", "filled"])
    return space_series(sensor, acfg)


def causal(values: np.ndarray, cap_h: float):
    """앞으로만 채우기·관측 표시·마지막 관측 뒤 시간(과제 3 forecast.data와 같은 함수)."""
    return _causal(values, cap_h)


def covariates(t: pd.DatetimeIndex, space: str, features, acfg, booking: np.ndarray | None = None) -> np.ndarray:
    """시간·일과표(·예약) 특징. 묶음 순서는 과제 3과 같다(time → schedule → booking)."""
    blocks = []
    if "time" in features:
        blocks.append(time_features(t, acfg))
    if "schedule" in features:
        blocks.append(schedule_features(t, space, acfg))
    if "booking" in features:
        blocks.append(booking if booking is not None else np.zeros((len(t), len(BOOK_COLS)), np.float32))
    return np.concatenate(blocks, 1).astype(np.float32) if blocks else np.zeros((len(t), 0), np.float32)

"""센서 5분 칸 → 공간 중앙값 시계열(research R4).

센서마다: 축 A error를 뺀 칸 평균 → 결측 처리(gaps.fill_bins) → 공간별 센서 중앙값.
공간 칸의 filled는 그 칸에 실제 관측한 센서가 하나도 없을 때만 True다(평가 정답에서 제외하는 기준).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .gaps import fill_bins

SERIES_ITEMS = ("co2", "temperature", "humidity")


def aggregate_bins(raw: pd.DataFrame, quality: pd.DataFrame, interval_min: int) -> pd.DataFrame:
    """센서별 칸 평균(축 A error 제외) — 달 단위로 불러 메모리를 아낀다. [device_id, item, bin, value, saturated, space_id]"""
    m = raw["item"].astype(str).isin(SERIES_ITEMS) & (quality["quality"].astype(str) != "error")
    df = raw.loc[m, ["ts", "space_id", "device_id", "item", "value"]].copy()
    df["saturated"] = (quality.loc[m, "quality"].astype(str) == "saturated").to_numpy()
    df = df[df["space_id"].astype(str) != "unmapped"]
    df["bin"] = df["ts"].dt.floor(f"{interval_min}min")
    g = df.groupby(["device_id", "item", "bin"], observed=True)
    return g.agg(value=("value", "mean"), saturated=("saturated", "max"), space_id=("space_id", "last")).reset_index()


def finish_bins(agg: pd.DataFrame, cfg, interval_min: int) -> pd.DataFrame:
    """칸 평균들을 이어 붙여 결측 처리 → [device_id, space_id, item, bin, value, filled, saturated]."""
    agg = agg.groupby(["device_id", "item", "bin"], observed=True).agg(
        value=("value", "mean"), saturated=("saturated", "max"), space_id=("space_id", "last")).reset_index()
    out = []
    gap_cfg = cfg["gaps"]
    for (dev, item), s in agg.groupby(["device_id", "item"], observed=True):
        s = s.set_index("bin").sort_index()
        full = pd.date_range(s.index.min(), s.index.max(), freq=f"{interval_min}min")
        vals = s["value"].reindex(full).to_numpy()
        v, filled = fill_bins(vals, interval_min, cfg.period_s(str(dev)), gap_cfg["short_max_min"])
        keep = ~np.isnan(v)
        space = s["space_id"].astype(str).reindex(full).ffill().bfill().to_numpy()
        sat = s["saturated"].reindex(full).fillna(False).astype(bool).to_numpy()
        out.append(pd.DataFrame({"device_id": str(dev), "space_id": space[keep], "item": str(item),
                                 "bin": full[keep], "value": v[keep], "filled": filled[keep], "saturated": sat[keep]}))
    res = pd.concat(out, ignore_index=True)
    for c in ("device_id", "space_id", "item"):
        res[c] = res[c].astype("category")
    return res


def sensor_bins(raw: pd.DataFrame, quality: pd.DataFrame, cfg, interval_min: int | None = None) -> pd.DataFrame:
    """한 번에(작은 데이터·시험용). 큰 데이터는 aggregate_bins를 달마다 부르고 finish_bins로 합친다."""
    interval_min = interval_min or cfg["interval"]["minutes"]
    return finish_bins(aggregate_bins(raw, quality, interval_min), cfg, interval_min)


def space_series(sb: pd.DataFrame, cfg) -> pd.DataFrame:
    """공간 대표값: [bin, space_id, item, value, n_sensors, n_observed, filled]."""
    excl = set(cfg["space"]["exclude_from_space"])
    d = sb[~sb["device_id"].astype(str).isin(excl)]
    g = d.groupby(["space_id", "item", "bin"], observed=True)
    res = g.agg(value=("value", "median"), n_sensors=("value", "size"),
                n_observed=("filled", lambda x: int((~x).sum()))).reset_index()
    res["filled"] = res["n_observed"] == 0
    return res


def to_wide(ss: pd.DataFrame, space: str, item: str, observed_only: bool = False) -> pd.Series:
    """한 공간·항목의 연속 칸 시계열(빈 칸 NaN)."""
    s = ss[(ss["space_id"].astype(str) == space) & (ss["item"].astype(str) == item)]
    if observed_only:
        s = s[~s["filled"]]
    s = s.set_index("bin")["value"].sort_index()
    if s.empty:
        return s
    step = (s.index[1:] - s.index[:-1]).min() if len(s) > 1 else pd.Timedelta("5min")
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq=step))

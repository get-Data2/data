"""정답: 사용 여부(일정 기반·관찰), 환기 경보(양성·음성·이미 초과·판단 불가) (research R1·R4, data-model "정답")."""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecast.features import schedule_features

from . import observe as ob
from .features import space_grid

VENT = ("positive", "negative", "exceeded", "undetermined")


def _intervals_utc(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ("start", "end"):
        d[c] = pd.to_datetime(d[c])
        if d[c].dt.tz is None:
            d[c] = d[c].dt.tz_localize("Asia/Seoul")
        d[c] = d[c].dt.tz_convert("UTC")
    return d


def in_intervals(t: pd.DatetimeIndex, iv: pd.DataFrame) -> np.ndarray:
    m = np.zeros(len(t), bool)
    for a, b in zip(iv["start"], iv["end"]):
        m |= (t >= a) & (t < b)
    return m


def occupancy_schedule(bins: pd.DatetimeIndex, space: str, cfg, bookings: pd.DataFrame | None) -> tuple[np.ndarray, str]:
    """일정 기반 정답(칸 [t, t+5분)): 실습실·사무실은 일과표 '일정 중', 회의실은 설정한 예약 출처. 없으면 NaN."""
    src = cfg["labels"]["occupancy"].get("meeting_source", "none")
    if space == "meeting":
        if src == "none" or bookings is None or bookings.empty:
            return np.full(len(bins), np.nan), "none"
        b = _intervals_utc(bookings[bookings["space_id"] == space])
        return in_intervals(bins, b).astype(float), src
    sf = schedule_features(bins, space, cfg.acfg)
    return sf[:, 0].astype(float), "timetable"


def observation_labels(bins: pd.DatetimeIndex, space: str, obs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """칸 [t, t+5분) 안의 관찰 상태 다수결(관찰이 칸의 절반 이상 덮을 때만). 인원은 그 칸에 걸친 구간의 최댓값."""
    n = len(bins)
    if obs is None or obs.empty:
        return np.full(n, np.nan), np.full(n, np.nan)
    iv = _intervals_utc(obs[obs["space_id"] == space])
    covered, occ, ppl = np.zeros(n), np.zeros(n), np.full(n, -1.0)
    step = pd.Timedelta(minutes=5)
    for a, b, st, p in zip(iv["start"], iv["end"], iv["state"], iv["people"]):
        lo = np.searchsorted(bins, a - step, side="right")
        hi = np.searchsorted(bins, b, side="left")
        for i in range(lo, hi):
            cov = max(0.0, (min(b, bins[i] + step) - max(a, bins[i])) / step)
            covered[i] += cov
            if st == "occupied":
                occ[i] += cov
            if p is not None and not pd.isna(p):
                ppl[i] = max(ppl[i], float(p))
    lab = np.where(covered >= 0.5, (occ / np.maximum(covered, 1e-9) >= 0.5).astype(float), np.nan)
    ppl = np.where(np.isnan(lab) | (ppl < 0), np.nan, ppl)
    return lab, ppl


def vent_labels(co2: pd.Series, threshold: float, horizon_min: int, min_future_obs: float) -> pd.DataFrame:
    """칸 t: CO₂(t) < 기준이고 (t, t+30분]에 관측 최댓값 ≥ 기준 → 양성. 이미 ≥ 기준 → 초과. 미래 관측 부족 → 판단 불가."""
    H = horizon_min // 5
    v = co2.to_numpy(float)
    n = len(v)
    fut = np.full((n, H), np.nan)
    for k in range(1, H + 1):
        fut[: n - k, k - 1] = v[k:]
    cov = (~np.isnan(fut)).mean(axis=1)
    mx = np.where(cov > 0, np.nanmax(np.where(np.isnan(fut), -np.inf, fut), axis=1), np.nan)
    first = np.full(n, np.nan)
    hit = fut >= threshold
    any_hit = hit.any(axis=1)
    first[any_hit] = hit[any_hit].argmax(axis=1) + 1
    label = np.where(np.isnan(v) | (cov < min_future_obs), "undetermined",
                     np.where(v >= threshold, "exceeded", np.where(mx >= threshold, "positive", "negative")))
    t = pd.DatetimeIndex(co2.index)
    return pd.DataFrame({"bin": t, "co2_now": v, "max_30": mx,
                         "first_exceed": [t[i] + pd.Timedelta(minutes=5 * int(first[i])) if not np.isnan(first[i]) else pd.NaT
                                          for i in range(n)],
                         "label": label})


def build_labels(series: pd.DataFrame, cfg, obs: pd.DataFrame | None, bookings: pd.DataFrame | None):
    occ, vent = [], []
    lc = cfg["labels"]
    for space in cfg.spaces:
        w = space_grid(series, space)
        if w.empty:
            continue
        bins = pd.DatetimeIndex(w.index)
        lab, src = occupancy_schedule(bins, space, cfg, bookings)
        ol, op = observation_labels(bins, space, obs)
        occ.append(pd.DataFrame({"space_id": space, "bin": bins, "label": lab, "source": src,
                                 "obs_label": ol, "obs_people": op}))
        v = vent_labels(w["co2"], lc["vent"]["threshold_ppm"], lc["vent"]["horizon_min"], lc["vent"]["min_future_obs"])
        v.insert(0, "space_id", space)
        vent.append(v)
    return pd.concat(occ, ignore_index=True), pd.concat(vent, ignore_index=True)


def load_observations(cfg) -> pd.DataFrame:
    df = ob.read(cfg["inputs"]["observations"])
    return ob.intervals(df) if len(df) else pd.DataFrame(columns=["space_id", "start", "end", "state", "people", "note"])

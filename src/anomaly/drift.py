"""영점 틀어짐 전용 탐지기 (심화 2, research R8).

1. 새벽 기준(drift_night): 사람이 없는 새벽 CO₂는 외기 수준과 비슷해야 한다. 센서별 새벽 하위 분위수가 학습 기간 기준보다 night_delta 이상 높으면 경보.
2. 이웃 비교(drift_peer): CO₂ 센서가 2대 이상인 공간에서, 센서와 이웃 중앙값의 차이(이동 중앙값)가 학습 기간 분포를 벗어나면 점수가 오른다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.config import KST

from .data import split_col


def _hm(s):
    return int(s[:2]) * 60 + int(s[3:])


def _night_lows(v: pd.Series, p) -> pd.Series:
    """날마다 새벽 구간(KST) 관측값의 하위 분위수. 인덱스 = 그날 새벽 구간 끝(UTC) — 판정 가능한 첫 시각."""
    a, b = (_hm(x) for x in p["weekday_hours"])
    k = v.index.tz_convert(KST)
    m = (k.hour * 60 + k.minute >= a) & (k.hour * 60 + k.minute < b)
    s = v[m].dropna()
    if s.empty:
        return pd.Series(dtype=float)
    day = s.index.tz_convert(KST).normalize()
    low = s.groupby(day).quantile(p["low_quantile"])
    low.index = (pd.DatetimeIndex(low.index) + pd.Timedelta(minutes=b)).tz_convert("UTC")
    return low


def night_baseline(train_w: pd.DataFrame, cfg) -> dict:
    p = cfg["drift"]["night"]
    out = {}
    for c in train_w.columns:
        if split_col(c)[1] == "co2":
            low = _night_lows(train_w[c], p)
            if len(low):
                out[c] = float(low.median())
    return out


def night_alarms(w: pd.DataFrame, base: dict, cfg, space: str) -> pd.DataFrame:
    p = cfg["drift"]["night"]
    out = []
    for c, b in base.items():
        if c not in w.columns:
            continue
        low = _night_lows(w[c], p)
        for t, v in low.items():
            if v - b > p["night_delta"]:
                out.append({"space_id": space, "t": t, "score": float(v - b), "cause": c})
    return pd.DataFrame(out, columns=["space_id", "t", "score", "cause"])


def _peer_resid(w: pd.DataFrame, cfg) -> pd.DataFrame:
    cols = [c for c in w.columns if split_col(c)[1] == "co2"]
    if len(cols) < 2:
        return pd.DataFrame(index=w.index)
    win = int(cfg["drift"]["peer"]["window_h"] * 12)
    r = {}
    for c in cols:
        others = w[[o for o in cols if o != c]].median(axis=1)
        d = (w[c] - others)
        r[c] = d.rolling(win, min_periods=win // 2).median()     # 과거 창만(인과적)
    return pd.DataFrame(r)


def peer_stats(train_w: pd.DataFrame, cfg) -> dict:
    r = _peer_resid(train_w, cfg)
    return {c: (float(r[c].median()), float((r[c] - r[c].median()).abs().median() * 1.4826) or 1.0) for c in r.columns}


def peer_scores(w: pd.DataFrame, stats: dict, cfg, space: str) -> pd.DataFrame:
    r = _peer_resid(w, cfg)
    if r.empty or not stats:
        return pd.DataFrame(columns=["space_id", "t", "score", "cause"])
    z = pd.DataFrame({c: (r[c] - stats[c][0]) / stats[c][1] for c in r.columns if c in stats})
    zz = z.fillna(-np.inf)
    valid = z.notna().any(axis=1)
    return pd.DataFrame({"space_id": space, "t": z.index[valid], "score": z[valid].max(axis=1).to_numpy(),
                         "cause": zz[valid].idxmax(axis=1).to_numpy()})

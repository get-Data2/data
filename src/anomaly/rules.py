"""규칙 탐지기: 급변(한 칸 튐)·값 고정 (research R4). 경보 시각은 판정에 필요한 마지막 관측 칸(인과적)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.config import KST

from .data import split_col


def spike(w: pd.DataFrame, stats: pd.DataFrame, cfg) -> pd.DataFrame:
    """관측값 x_t가 앞 관측과 뒤 관측 모두에서 k·s 이상 반대 방향으로 벗어나면 튄 값. 경보 시각 = 다음 관측 칸."""
    p = cfg["rules"]["spike"]
    out = []
    for c in w.columns:
        v = w[c].dropna()
        if len(v) < 3:
            continue
        item = split_col(c)[1]
        s = max(float(stats.loc[c, "mad_diff"]) if c in stats.index else 0.0, p["floor"][item])
        d1 = v.diff().to_numpy()
        d2 = -v.diff(-1).to_numpy()            # x_{t+1} − x_t
        hit = (np.abs(d1) > p["k"] * s) & (np.abs(d2) > p["k"] * s) & (np.sign(d1) != np.sign(d2))
        idx = np.where(hit)[0]
        for i in idx:
            out.append({"t": v.index[i + 1], "cause": c, "q": 1.0, "score": float(abs(d1[i]) / s)})
    return pd.DataFrame(out, columns=["t", "cause", "q", "score"])


def _runs_constant(v: pd.Series, min_len: pd.Timedelta):
    """관측값이 바뀌지 않는 구간 [시작, 끝] 중 min_len 이상."""
    if v.empty:
        return []
    chg = v.ne(v.shift()).cumsum()
    out = []
    for _, g in v.groupby(chg):
        if g.index[-1] - g.index[0] >= min_len:
            out.append((g.index[0], g.index[-1]))
    return out


def stuck(w: pd.DataFrame, cfg, acfg) -> pd.DataFrame:
    """값 고정: 같은 값이 min_hours 이상 + (같은 공간 다른 장치의 같은 항목이 그동안 변함 | 참조가 없으면 사용 시간대 alone_hours 이상).
    경보 시각 = 조건이 처음 성립한 칸."""
    p = cfg["rules"]["stuck"]
    mh, ah = pd.Timedelta(hours=p["min_hours"]), pd.Timedelta(hours=p["alone_hours"])
    out = []
    for c in w.columns:
        dev, item = split_col(c)
        refs = [r for r in w.columns if split_col(r)[1] == item and split_col(r)[0] != dev]
        v = w[c].dropna()
        for a, b in _runs_constant(v, mh):
            if refs:
                seg = w.loc[a:b, refs]
                # 참조 장치들의 중앙값이 고정 구간 안에서 기준 이상 변했나(처음 성립 시각)
                med = seg.median(axis=1).dropna()
                if med.empty:
                    continue
                rng_ = (med.cummax() - med.cummin())
                ok = rng_[(rng_ >= p["others_min_change"][item]) & (rng_.index >= a + mh)]
                if len(ok):
                    out.append({"t": ok.index[0], "cause": c, "q": 1.0, "score": (b - a) / pd.Timedelta(hours=1)})
            else:
                if b - a < ah:
                    continue
                k = pd.date_range(a + ah, b, freq="5min").tz_convert(KST)
                m = (k.weekday < 5) & (k.hour >= 9) & (k.hour < 18)
                if m.any():
                    out.append({"t": k[np.argmax(m)].tz_convert("UTC"), "cause": c, "q": 1.0,
                                "score": (b - a) / pd.Timedelta(hours=1)})
    return pd.DataFrame(out, columns=["t", "cause", "q", "score"])

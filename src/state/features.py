"""인과적 특징 묶음 (research R5, data-model "특징").

공간 5분 시계열의 관측 칸만 쓰고 빈 칸은 앞으로만 채운다. 칸 t의 특징은 t까지의 값만 본다(자동 시험).
TODO(과제 8 T006): src/preprocess/가 생기면 공간 시계열·시간·일정 특징을 그 모듈에서 가져온다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecast.features import SCHED_COLS, TIME_COLS, schedule_features, time_features

ITEMS = ("co2", "temperature", "humidity")
SHORT = {"co2": "co2", "temperature": "temp", "humidity": "hum"}


def group_columns() -> dict[str, list[str]]:
    g = {"level": [], "delta": [], "shape": [], "obs": ["obs_ratio30"], "time": [f"t_{c}" for c in TIME_COLS],
         "schedule": [f"sch_{c}" for c in SCHED_COLS]}
    for it in ITEMS:
        s = SHORT[it]
        g["level"].append(f"lvl_{s}")
        g["delta"] += [f"d5_{s}", f"d15_{s}", f"d30_{s}"]
        g["shape"] += [f"slope30_{s}", f"sd30_{s}", f"rise30_{s}"]
        g["obs"].append(f"age_{s}")
    return g


def columns_for(groups: list[str]) -> list[str]:
    gc = group_columns()
    return [c for g in ("level", "delta", "shape", "obs", "time", "schedule") if g in groups for c in gc[g]]


def space_grid(series: pd.DataFrame, space: str, t0=None, t1=None) -> pd.DataFrame:
    """공간 넓은 표(5분 칸 × 항목), 관측 칸만(나머지 NaN)."""
    s = series[(series["space_id"].astype(str) == space) & (~series["filled"])]
    w = s.pivot_table(index="bin", columns="item", values="value", aggfunc="first", observed=True)
    w.columns = [str(c) for c in w.columns]
    w.index = pd.DatetimeIndex(w.index).as_unit("ns")
    a = pd.Timestamp(t0) if t0 is not None else w.index.min()
    b = pd.Timestamp(t1) if t1 is not None else w.index.max()
    t = pd.date_range(a.floor("5min"), b, freq="5min").as_unit("ns")
    return w.reindex(index=t, columns=list(ITEMS))


def _slope(v: np.ndarray, W: int) -> np.ndarray:
    """최근 W칸 관측값의 선형 기울기(칸당). 관측 3개 미만이면 NaN."""
    n = len(v)
    out = np.full(n, np.nan)
    x = np.arange(W, dtype=float)
    for i in range(W - 1, n):
        y = v[i - W + 1:i + 1]
        m = ~np.isnan(y)
        if m.sum() >= 3:
            xx, yy = x[m], y[m]
            xx = xx - xx.mean()
            out[i] = float((xx * (yy - yy.mean())).sum() / (xx ** 2).sum())
    return out


def space_features(w: pd.DataFrame, space: str, norm: dict, acfg, window_min: int = 30, age_cap_h: float = 6.0) -> pd.DataFrame:
    W = window_min // 5 + 1                 # 칸 t 포함 최근 30분 = 7칸
    out = {}
    anyobs = w.notna().any(axis=1).astype(float)
    out["obs_ratio30"] = anyobs.rolling(W, min_periods=1).mean().to_numpy()
    for it in ITEMS:
        s = SHORT[it]
        st = norm["spaces"][space][it] if space in norm["spaces"] and it in norm["spaces"][space] else {"mean": 0.0, "std": 1.0}
        mu, sd = st["mean"], st["std"] or 1.0
        v = w[it]
        f = v.ffill()                                         # 앞으로만 채움(인과적)
        out[f"lvl_{s}"] = ((f - mu) / sd).to_numpy()
        for k, name in ((1, "d5"), (3, "d15"), (6, "d30")):
            out[f"{name}_{s}"] = ((f - f.shift(k)) / sd).to_numpy()
        out[f"slope30_{s}"] = _slope(v.to_numpy(float), W) / sd
        out[f"sd30_{s}"] = (v.rolling(W, min_periods=2).std() / sd).to_numpy()
        out[f"rise30_{s}"] = ((f - f.rolling(W, min_periods=1).min()) / sd).to_numpy()
        idx = np.where(v.notna().to_numpy(), np.arange(len(v)), -1)
        last = np.maximum.accumulate(idx)
        out[f"age_{s}"] = np.where(last >= 0, np.minimum((np.arange(len(v)) - last) * 5 / 60, age_cap_h), age_cap_h) / age_cap_h
    t = pd.DatetimeIndex(w.index)
    tf = time_features(t, acfg)
    for i, c in enumerate(TIME_COLS):
        out[f"t_{c}"] = tf[:, i]
    sf = schedule_features(t, space, acfg)
    for i, c in enumerate(SCHED_COLS):
        out[f"sch_{c}"] = sf[:, i]
    df = pd.DataFrame(out, index=t)
    df.insert(0, "space_id", space)
    return df.rename_axis("bin").reset_index()


def assign_split(bins: pd.Series, cfg) -> np.ndarray:
    sp = cfg.split
    out = np.full(len(bins), "other", dtype=object)
    for part in ("train", "val", "test"):
        a, b = sp.utc(part)
        out[(bins >= pd.Timestamp(a)) & (bins < pd.Timestamp(b))] = part
    live = pd.Timestamp(cfg["observe"]["live_start"], tz="Asia/Seoul").tz_convert("UTC")
    out[(bins >= live) & (out == "other")] = "live"
    return out


def build_features(series: pd.DataFrame, norm: dict, cfg) -> pd.DataFrame:
    fc = cfg["features"]
    parts = []
    for space in cfg.spaces:
        w = space_grid(series, space)
        if w.empty:
            continue
        parts.append(space_features(w, space, norm, cfg.acfg, fc["window_min"]))
    df = pd.concat(parts, ignore_index=True)
    df["split"] = assign_split(df["bin"], cfg)
    df["input_ok"] = df["obs_ratio30"] >= fc["min_input_obs"]
    return df

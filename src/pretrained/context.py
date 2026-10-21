"""컨텍스트 만들기: 관측 칸만, t0까지(포함), 빈 칸 NaN(또는 앞으로 채움) (research R3).

과제 2 보간 칸(filled=True)은 결측 뒤쪽 관측(t0 이후일 수 있음)을 써서 쓰지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.baseline import usage_mask
from analysis.pipeline import sha256_file

STEP = pd.Timedelta(minutes=5)


class InputHashError(RuntimeError):
    pass


def load_series(cfg) -> pd.DataFrame:
    s = pd.read_parquet(cfg["inputs"]["series"], columns=["space_id", "item", "bin", "value", "filled"])
    s["space_id"] = s["space_id"].astype(str)
    s["item"] = s["item"].astype(str)
    return s[s["space_id"].isin(cfg["spaces"])]


def load_eval(cfg) -> pd.DataFrame:
    p = cfg["inputs"]["eval_points"]
    sha = sha256_file(p)
    if sha != cfg["inputs"]["eval_points_sha256"]:
        raise InputHashError(f"평가 시점 파일 해시 불일치: {sha[:16]} — 과제 2를 확인")
    return pd.read_parquet(p)


def grid(series: pd.DataFrame, space: str, item: str, t0=None, t1=None) -> pd.Series:
    """관측값만, 연속 5분 칸(빈 칸 NaN)."""
    s = series[(series["space_id"] == space) & (series["item"] == item) & (~series["filled"])]
    s = s.set_index("bin")["value"].sort_index()
    s = s[~s.index.duplicated()]
    s.index = pd.DatetimeIndex(s.index).as_unit("ns")
    a = pd.Timestamp(t0) if t0 is not None else s.index.min()
    b = pd.Timestamp(t1) if t1 is not None else s.index.max()
    return s.reindex(pd.date_range(a.floor("5min"), b, freq="5min").as_unit("ns"))


def contexts(g: pd.Series, t0s, L: int, missing: str = "nan") -> tuple[np.ndarray, np.ndarray]:
    """t0마다 [t0−L+1, t0] 창. 돌려줌: (n, L) float32, 관측 비율(n,). t0 이후 값은 쓰지 않는다(인과적)."""
    v = g.to_numpy(np.float64)
    obs = ~np.isnan(v)
    if missing == "ffill":
        last = np.maximum.accumulate(np.where(obs, np.arange(len(v)), -1))
        v = np.where(last >= 0, v[np.clip(last, 0, None)], np.nan)
    idx = g.index.get_indexer(pd.DatetimeIndex(t0s).as_unit("ns"))
    if (idx < 0).any():
        raise KeyError("t0가 격자 밖")
    win = idx[:, None] + np.arange(-L + 1, 1)[None]
    ok = win >= 0
    winc = np.clip(win, 0, None)
    ctx = np.where(ok, v[winc], np.nan).astype(np.float32)
    ratio = np.where(ok, obs[winc], False).mean(1)
    return ctx, ratio


def val_t0(cfg, series: pd.DataFrame, space: str) -> pd.DatetimeIndex:
    """검증 표본: 과제 3 규칙 — t0 CO₂ 관측, t0+60분이 기간 안."""
    a, b = cfg.split.utc("val")
    g = grid(series, space, cfg["item"], a, b)
    t = g.index[g.notna().to_numpy()]
    return t[(t >= pd.Timestamp(a)) & (t + pd.Timedelta(minutes=60) < pd.Timestamp(b))]


def eval_keys(cfg, ev: pd.DataFrame) -> pd.DataFrame:
    """시험 표본: 평가 시점 파일 CO₂ 행의 고유 (space_id, t0)."""
    e = ev[(ev["item"] == cfg["item"]) & ev["space_id"].isin(cfg["spaces"])]
    return e[["space_id", "t0"]].drop_duplicates().reset_index(drop=True)


def val_targets(cfg, series: pd.DataFrame, space: str, t0s: pd.DatetimeIndex) -> pd.DataFrame:
    """검증 정답: 지평별 관측값(없으면 행 없음), 사용 시간대 표시."""
    a, b = cfg.split.utc("val")
    g = grid(series, space, cfg["item"], a, b)
    rows = []
    for h in cfg.horizons:
        y = g.reindex(pd.DatetimeIndex(t0s) + pd.Timedelta(minutes=h)).to_numpy()
        rows.append(pd.DataFrame({"space_id": space, "item": cfg["item"], "t0": t0s, "horizon_min": h, "y_true": y}))
    df = pd.concat(rows, ignore_index=True).dropna(subset=["y_true"])
    df["usage_hours"] = usage_mask(pd.DatetimeIndex(df["t0"]), cfg.acfg)
    return df

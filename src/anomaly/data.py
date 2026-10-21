"""센서 넓은 표·학습 기간 통계·공간 대표값 (data-model "센서 넓은 표")."""
from __future__ import annotations

import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=5)


def load_sensor(cfg) -> pd.DataFrame:
    s = pd.read_parquet(cfg["inputs"]["sensor_series"],
                        columns=["device_id", "space_id", "item", "bin", "value", "filled"])
    for c in ("device_id", "space_id", "item"):
        s[c] = s[c].astype(str)
    return s[s["space_id"].isin(cfg.spaces) & s["item"].isin(cfg["items"])]


def col_name(device: str, item: str) -> str:
    return f"{device}/{item}"


def split_col(c: str) -> tuple[str, str]:
    d, i = c.rsplit("/", 1)
    return d, i


def space_cols(sensor: pd.DataFrame, space: str, cfg) -> list[str]:
    ex = set(cfg["spaces"][space].get("exclude_devices") or [])
    d = sensor[(sensor["space_id"] == space) & ~sensor["device_id"].isin(ex)][["device_id", "item"]].drop_duplicates()
    return sorted(col_name(a, b) for a, b in d.itertuples(index=False))


def wide(sensor: pd.DataFrame, space: str, cols: list[str], t0, t1, value_col: str = "value") -> pd.DataFrame:
    """공간 넓은 표: 5분 칸 × 열(장치/항목). 관측 칸만(filled 칸은 NaN)."""
    t = pd.date_range(pd.Timestamp(t0), pd.Timestamp(t1), freq="5min", inclusive="left")
    s = sensor[(sensor["space_id"] == space) & (~sensor["filled"]) & (sensor["bin"] >= t[0]) & (sensor["bin"] <= t[-1])]
    s = s.assign(col=s["device_id"] + "/" + s["item"])
    w = s.pivot_table(index="bin", columns="col", values=value_col, aggfunc="first")
    w.index = pd.DatetimeIndex(w.index).as_unit("ns")
    return w.reindex(index=t.as_unit("ns"), columns=cols)


def obs_diffs(col: pd.Series) -> pd.Series:
    """관측값끼리의 이웃 차이(10분 주기 센서는 관측 사이 간격으로)."""
    v = col.dropna()
    return v.diff().dropna()


def sensor_stats(w: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for c in w.columns:
        v = w[c].dropna()
        d = obs_diffs(w[c])
        mad = float((d - d.median()).abs().median() * 1.4826) if len(d) else np.nan
        rows[c] = {"mean": float(v.mean()), "std": float(v.std()) or 1.0, "mad_diff": mad,
                   "p05": float(v.quantile(0.05)), "p95": float(v.quantile(0.95)), "n": int(len(v))}
    return pd.DataFrame(rows).T


def space_median(w: pd.DataFrame, item: str) -> pd.Series:
    """공간 대표값: 같은 항목 장치들의 중앙값(과제 2 규칙)."""
    cols = [c for c in w.columns if split_col(c)[1] == item]
    return w[cols].median(axis=1, skipna=True)

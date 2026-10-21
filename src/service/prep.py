"""주기마다: 1분 원본 창 → 공통 전처리(축 A·닫힌 5분 칸·공간 중앙값) → 판단별 입력 (research R2·R3)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import preprocess as pp

STEP = pd.Timedelta(minutes=5)


@dataclass
class Prepared:
    now: pd.Timestamp
    end: pd.Timestamp                    # 마지막 닫힌 칸의 끝(공개 t0)
    t0_bin: pd.Timestamp                 # 마지막 닫힌 칸의 시작(학습의 t0 칸)
    raw: pd.DataFrame
    quality: pd.DataFrame
    sensor: pd.DataFrame                 # 닫힌 칸 센서 평균(관측·주기 안 직전값)
    space: pd.DataFrame                  # 공간 대표값
    freshness_min: dict = field(default_factory=dict)


def prepare(raw: pd.DataFrame, acfg, now: pd.Timestamp, spaces: list[str]) -> Prepared:
    end = pp.closed_bin_end(now, acfg["interval"]["minutes"])
    if raw.empty:
        q = pd.DataFrame({"quality": pd.Categorical([], ["ok", "saturated", "error"]), "rule": pd.Categorical([])})
        sb = pp.sensor_bins_closed(raw, q, acfg, end)
    else:
        raw = raw.reset_index(drop=True)
        q = pp.axis_a(raw, acfg)
        sb = pp.sensor_bins_closed(raw, q, acfg, end)
    ss = pp.space_bins(sb, acfg)
    fresh = {}
    for s in spaces:
        o = ss[(ss["space_id"].astype(str) == s) & (~ss["filled"].astype(bool))] if len(ss) else ss
        fresh[s] = round(float((now - (o["bin"].max() + STEP)) / pd.Timedelta(minutes=1)), 1) if len(o) else None
    return Prepared(now, end, end - STEP, raw, q, sb, ss, fresh)


def forecast_input(prep: Prepared, runner, space: str, acfg, min_input_obs: float = 0.1):
    """과제 3 forecast.dataset.Windows.batch와 같은 열 순서의 (past, future, current). 입력 관측 비율이 낮으면 None."""
    spec = runner.spec
    L, H = runner.L, runner.H
    first = prep.space["bin"].min() if len(prep.space) else prep.t0_bin
    t = pd.date_range(min(first, prep.t0_bin - (L - 1) * STEP), prep.t0_bin, freq="5min")
    ss = prep.space[(prep.space["space_id"].astype(str) == space) & (~prep.space["filled"].astype(bool))]
    mu, sd = runner.mu_sd(space)
    vals, obs, age = [], [], []
    for item in pp.ITEMS:
        v = ss[ss["item"].astype(str) == item].set_index("bin")["value"]
        v = v[~v.index.duplicated()].reindex(t).to_numpy(np.float64)
        f, o, a = pp.causal(v, runner.age_cap)
        vals.append(f), obs.append(o), age.append(a)
    raw_ffill = np.stack(vals, 1)
    o = np.stack(obs, 1).astype(np.float32)
    a = np.stack(age, 1).astype(np.float32)
    xval = np.nan_to_num((raw_ffill - mu) / sd, nan=0.0).astype(np.float32)
    feats = runner.exp.features
    cov_p = pp.covariates(t, space, feats, acfg)
    past = np.concatenate([xval, o, a / runner.age_cap, cov_p], 1)[-L:].astype(np.float32)
    if past.shape[1] != len(spec["past_columns"]):
        raise ValueError(f"입력 열 수 {past.shape[1]} ≠ 규약 {len(spec['past_columns'])}")
    if o[-L:].max(axis=1).mean() < min_input_obs:
        return None
    tf = pd.date_range(prep.t0_bin + STEP, periods=H, freq="5min")
    fut = pp.covariates(tf, space, feats, acfg)
    current = raw_ffill[-1, :2]
    if np.isnan(current).any():
        return None
    return past, fut, current

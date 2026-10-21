"""과제 2 산출을 읽어 공간 배열을 만든다 (data-model "공간 배열", research R4).

입력은 관측 칸(filled=False)만 쓰고, 빈 칸은 앞으로만 채운다. 과제 2의 보간 칸은 결측 뒤쪽 관측(미래)을 쓰므로 쓰지 않는다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pipeline import sha256_file

from .features import BOOK_COLS, SCHED_COLS, TIME_COLS, booking_features, estimate_bookings, schedule_features, time_features

ITEMS = ("co2", "temperature", "humidity")       # 입력 센서 순서
STEP = pd.Timedelta(minutes=5)


class InputHashError(RuntimeError):
    pass


@dataclass
class Arrays:
    t: pd.DatetimeIndex            # (N,) UTC 5분 칸
    spaces: tuple
    raw_ffill: np.ndarray          # (S, N, 3) 앞으로 채운 원값(첫 관측 전 NaN)
    obs: np.ndarray                # (S, N, 3) 관측 표시
    age_h: np.ndarray              # (S, N, 3) 마지막 관측 뒤 시간(상한)
    y: np.ndarray                  # (S, N, 2) CO₂·온도 관측 원값(빈 칸 NaN) — 정답
    cov: dict                      # 묶음 이름 → (S, N, k)
    bookings: pd.DataFrame         # space_id, start, end, source
    age_cap: float = 6.0

    def index_of(self, ts) -> np.ndarray:
        return ((pd.DatetimeIndex(ts) - self.t[0]) // STEP).astype(int).to_numpy()


def load_inputs(cfg) -> dict:
    inp = cfg["inputs"]
    p = Path(inp["eval_points"])
    sha = sha256_file(p)
    if sha != inp["eval_points_sha256"]:
        raise InputHashError(f"평가 시점 파일 해시 불일치: {sha[:16]} ≠ {inp['eval_points_sha256'][:16]} — 과제 2를 확인")
    return {"series": pd.read_parquet(inp["series"]), "eval": pd.read_parquet(p),
            "norm": json.loads(Path(inp["norm"]).read_text()), "eval_sha256": sha,
            "series_sha16": sha256_file(Path(inp["series"]))[:16], "norm_sha16": sha256_file(Path(inp["norm"]))[:16]}


def _causal(values: np.ndarray, cap_h: float):
    """앞으로만 채우기 + 관측 표시 + 마지막 관측 뒤 시간."""
    obs = ~np.isnan(values)
    n = len(values)
    last = np.where(obs, np.arange(n), -1)
    last = np.maximum.accumulate(last)
    filled = np.where(last >= 0, values[np.clip(last, 0, None)], np.nan)
    age = np.where(last >= 0, (np.arange(n) - last) * 5 / 60, cap_h)
    return filled, obs, np.minimum(age, cap_h)


def grid_bounds(cfg, max_lookback_min: int = 1440):
    sp = cfg.split
    a, _ = sp.utc("train")
    _, b = sp.utc("test")
    return (pd.Timestamp(a) - pd.Timedelta(minutes=max_lookback_min + 5),
            pd.Timestamp(b) + pd.Timedelta(minutes=5 * (cfg["horizon_bins"] + 1)))


def load_bookings(cfg, series: pd.DataFrame) -> pd.DataFrame:
    """실제 예약 파일이 있으면 그것, 없으면 추정 예약(research R10)."""
    bk = cfg["booking"]
    if bk.get("file"):
        df = pd.read_csv(bk["file"])
        for c in ("start", "end"):
            df[c] = pd.to_datetime(df[c]).dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC")
        df["source"] = "file"
        return df[["space_id", "start", "end", "source"]]
    out = []
    for space in bk["estimate"]["spaces"]:
        s = series[(series["space_id"].astype(str) == space) & (series["item"].astype(str) == "co2") & (~series["filled"])]
        co2 = s.set_index("bin")["value"].sort_index()
        df = estimate_bookings(co2, bk["estimate"])
        df["space_id"] = space
        df["source"] = "estimated"
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["space_id", "start", "end", "source"])


def build_arrays(cfg, series: pd.DataFrame, t_start=None, t_end=None, bookings: pd.DataFrame | None = None) -> Arrays:
    if t_start is None:
        t_start, t_end = grid_bounds(cfg)
    t = pd.date_range(pd.Timestamp(t_start).floor("5min"), pd.Timestamp(t_end), freq="5min")
    spaces = tuple(cfg["spaces"])
    S, N = len(spaces), len(t)
    raw_ffill = np.full((S, N, 3), np.nan, np.float32)
    obs = np.zeros((S, N, 3), np.uint8)
    age = np.zeros((S, N, 3), np.float32)
    y = np.full((S, N, 2), np.nan, np.float32)
    ser = series[~series["filled"]]
    if bookings is None:
        bookings = load_bookings(cfg, series)
    tf = time_features(t, cfg.acfg)
    cov = {"time": np.repeat(tf[None], S, 0), "schedule": np.zeros((S, N, len(SCHED_COLS)), np.float32),
           "booking": np.zeros((S, N, len(BOOK_COLS)), np.float32)}
    for si, space in enumerate(spaces):
        sub = ser[ser["space_id"].astype(str) == space]
        for ii, item in enumerate(ITEMS):
            v = sub[sub["item"].astype(str) == item].set_index("bin")["value"].sort_index()
            v = v[~v.index.duplicated()].reindex(t).to_numpy(np.float64)
            f, o, a = _causal(v, cfg["age_cap_h"])
            raw_ffill[si, :, ii], obs[si, :, ii], age[si, :, ii] = f, o, a
            if ii < 2:
                y[si, :, ii] = v
        cov["schedule"][si] = schedule_features(t, space, cfg.acfg)
        cov["booking"][si] = booking_features(t, bookings[bookings["space_id"] == space])
    return Arrays(t, spaces, raw_ffill, obs, age, y, cov, bookings, float(cfg["age_cap_h"]))


def norm_table(norm: dict, spaces: tuple, mode: str, train_spaces: tuple) -> np.ndarray:
    """(S, 3, 2) [평균, 표준편차]. pooled는 학습 공간들의 칸 수 가중 합친 통계(뺀 공간 정보를 쓰지 않음)."""
    out = np.zeros((len(spaces), len(ITEMS), 2), np.float32)
    for ii, item in enumerate(ITEMS):
        if mode == "pooled":
            st = [norm["spaces"][s][item] for s in train_spaces]
            w = np.array([x["bins"] for x in st], float)
            mu = float(np.sum(w * [x["mean"] for x in st]) / w.sum())
            var = float(np.sum(w * ([x["std"] ** 2 for x in st] + (np.array([x["mean"] for x in st]) - mu) ** 2)) / w.sum())
            out[:, ii] = (mu, var ** 0.5)
        else:
            for si, s in enumerate(spaces):
                out[si, ii] = (norm["spaces"][s][item]["mean"], norm["spaces"][s][item]["std"])
    return out


def split_t0(arr: Arrays, cfg, kind: str, lookback_bins: int, spaces=None) -> np.ndarray:
    """학습·검증 표본 (공간 인덱스, t 인덱스). t0와 t0+60분이 같은 기간 안, t0 현재값(CO₂·온도) 관측, 입력 관측 비율 충족."""
    a, b = cfg.split.utc(kind)
    H = cfg["horizon_bins"]
    ti = np.arange(len(arr.t))
    tt = arr.t
    inside = (tt >= a) & (tt + H * STEP < b) & (ti >= lookback_bins - 1)
    out = []
    for si, space in enumerate(arr.spaces):
        if spaces is not None and space not in spaces:
            continue
        cur_ok = ~np.isnan(arr.y[si, :, 0]) & ~np.isnan(arr.y[si, :, 1])
        ratio = input_obs_ratio(arr, si, lookback_bins)
        m = inside & cur_ok & (ratio >= cfg["min_input_obs"])
        idx = np.where(m)[0]
        out.append(np.column_stack([np.full(len(idx), si), idx]))
    return np.concatenate(out) if out else np.zeros((0, 2), int)


def input_obs_ratio(arr: Arrays, si: int, lookback_bins: int) -> np.ndarray:
    """칸마다 과거 lookback 구간에서 센서 하나라도 관측된 칸의 비율(인과적)."""
    anyobs = arr.obs[si].max(axis=1).astype(np.float64)
    c = np.concatenate([[0], np.cumsum(anyobs)])
    n = np.arange(1, len(anyobs) + 1)
    lo = np.clip(n - lookback_bins, 0, None)
    return (c[n] - c[lo]) / np.minimum(n, lookback_bins)


def eval_samples(arr: Arrays, eval_points: pd.DataFrame, cfg, lookback_bins: int, spaces=None):
    """평가 시점 파일의 고유 (공간, t0) → (공간 인덱스, t 인덱스), 예측 불가 표시."""
    u = eval_points[["space_id", "t0"]].drop_duplicates()
    if spaces is not None:
        u = u[u["space_id"].isin(spaces)]
    si = u["space_id"].map({s: i for i, s in enumerate(arr.spaces)}).to_numpy()
    ti = arr.index_of(u["t0"])
    ratios = [input_obs_ratio(arr, s, lookback_bins) for s in range(len(arr.spaces))]
    ratio = np.array([ratios[s][t] for s, t in zip(si, ti)]) if len(u) else np.zeros(0)
    unpred = (ratio < cfg["min_input_obs"]) | (ti < lookback_bins - 1)
    return np.column_stack([si, ti]), unpred, u.reset_index(drop=True)

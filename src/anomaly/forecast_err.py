"""심화 1: 과제 3 예측 오차를 이상 점수로 (research R7).

평가판 센서 값으로 공간 대표값(중앙값, 과제 2 규칙)을 다시 만들고, 과제 3 모델 묶음으로 "h분 전 시점의 h분 예측"을 낸다.
점수 = Σ(항목) |실제 − 예측| / 검증 기간 잔차 MAD. 원인은 항목 수준("공간/항목")까지만 가리킨다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import split_col

ITEMS = ("co2", "temperature")


def _space_series_from_wide(w: pd.DataFrame, space: str, exclude: set) -> pd.DataFrame:
    rows = []
    for item in ("co2", "temperature", "humidity"):
        cols = [c for c in w.columns if split_col(c)[1] == item and split_col(c)[0] not in exclude]
        if not cols:
            continue
        v = w[cols].median(axis=1, skipna=True)
        rows.append(pd.DataFrame({"space_id": space, "item": item, "bin": w.index, "value": v.to_numpy(),
                                  "filled": v.isna().to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def _load_forecaster(device):
    from forecast.config import load_forecast_config
    from forecast.experiments import exp_from_key
    from forecast.models import ForecastModel
    fcfg = load_forecast_config()
    root = Path("var/forecast/bundle")
    b = json.loads((root / "bundle.json").read_text())
    exp = exp_from_key(b["config"], [b["seed"]])
    norm = json.loads(Path(fcfg["inputs"]["norm"]).read_text())
    return fcfg, exp, norm, root, ForecastModel


def score_forecast_err(work, device, period: str) -> pd.DataFrame:
    from forecast.data import build_arrays, input_obs_ratio
    from forecast.dataset import Windows
    from forecast.train import predict
    fcfg, exp, norm, root, ForecastModel = _load_forecaster(device)
    h = work.cfg["forecast_err"]["horizon_min"] // 5
    a, b = work.cfg.split.utc(period)
    t0, t1 = pd.Timestamp(a) - pd.Timedelta(minutes=exp.lookback_min + 10), pd.Timestamp(b)
    exclude = set(work.cfg.acfg["space"].get("exclude_from_space") or [])
    parts = []
    for space in fcfg["spaces"]:
        w = work.w(space, "val") if period == "val" else work.board(space)
        if space not in work.cfg.spaces:
            continue
        parts.append(_space_series_from_wide(w, space, exclude))
    ser = pd.concat(parts, ignore_index=True)
    ser = ser[(ser["bin"] >= t0) & (ser["bin"] < t1)]
    arr = build_arrays(fcfg, ser, t0, t1, bookings=pd.DataFrame(columns=["space_id", "start", "end", "source"]))
    W = Windows(arr, exp, norm, fcfg["horizon_bins"], device)
    model = ForecastModel(exp, W.fp, W.ff, W.n_space, W.H).to(device)
    model.load_state_dict(torch.load(root / "model.pt", map_location=device))
    out = []
    for si, space in enumerate(arr.spaces):
        cur_ok = ~np.isnan(arr.y[si, :, 0]) | ~np.isnan(arr.y[si, :, 1])
        ratio = input_obs_ratio(arr, si, W.L)
        ti = np.where(cur_ok & (ratio >= fcfg["min_input_obs"]) & (np.arange(len(arr.t)) >= W.L - 1)
                      & (np.arange(len(arr.t)) + W.H < len(arr.t)))[0]
        if not len(ti):
            continue
        idx = np.column_stack([np.full(len(ti), si), ti])
        pred, y = predict(model, W, idx)
        res = np.abs(pred[:, h - 1] - y[:, h - 1])                       # (n, 2) 정답 칸 = t0 + h
        out.append(pd.DataFrame({"space_id": space, "t": arr.t[ti + h], "r_co2": res[:, 0], "r_temperature": res[:, 1]}))
    df = pd.concat(out, ignore_index=True)
    return work.period_slice(df, period)


def finalize_scores(val: pd.DataFrame, test: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """검증 기간 잔차 MAD로 정규화해 공간 점수와 원인 항목을 만든다(시험은 검증 통계만 사용)."""
    def mad(x):
        x = x.dropna()
        return float((x - x.median()).abs().median() * 1.4826) or 1.0
    st = {(s, it): mad(g[f"r_{it}"]) for s, g in val.groupby("space_id") for it in ITEMS}

    def conv(df):
        z = {it: df[f"r_{it}"] / df["space_id"].map(lambda s, it=it: st[(s, it)]) for it in ITEMS}
        zz = pd.DataFrame(z)
        return pd.DataFrame({"space_id": df["space_id"], "t": df["t"], "score": zz.sum(axis=1, min_count=1),
                             "cause": df["space_id"] + "/" + zz.fillna(-np.inf).idxmax(axis=1)}).dropna(subset=["score"])
    return conv(val), (conv(test) if test is not None else None)

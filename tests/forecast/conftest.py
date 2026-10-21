"""과제 3 시험 고정물: 짧은 합성 분할과 과제 2 space_5min 형식의 합성 시계열."""
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from tests.conftest import ROOT

KST = timezone(timedelta(hours=9))
# 2026-03-02(월) ~ 03-29: 학습 2주, 검증 1주, 시험 1주
SYN_SPLIT = {"train": (date(2026, 3, 2), date(2026, 3, 15)), "val": (date(2026, 3, 17), date(2026, 3, 21)),
             "test": (date(2026, 3, 23), date(2026, 3, 28))}


@pytest.fixture
def fcfg():
    from analysis.config import Split
    from forecast.config import ForecastConfig, load_forecast_config
    base = load_forecast_config(ROOT / "config" / "forecast.yaml")
    acfg = replace(base.acfg, splits={"syn": Split("syn", **SYN_SPLIT)})
    raw = dict(base.raw)
    raw["split"] = "syn"
    raw["defaults"] = dict(raw["defaults"], size={"hidden": 8, "layers": 1, "channels": 8, "kernel": 3, "head": 16,
                                                  "dropout": 0.0},
                           lookback_min=60, seeds=[0],
                           train={"lr": 0.003, "weight_decay": 0.0, "batch": 256, "max_epochs": 2,
                                  "patience_lr": 5, "patience_stop": 8})
    return ForecastConfig(base.path, raw, acfg)


def synth_series(start=datetime(2026, 2, 28, tzinfo=KST), days=31, seed=0, gaps=()):
    """평일 09:00~11:30·13:00~18:00 CO₂ 상승, 점심·퇴실 하락. 회의실은 10분 주기(5분 칸 절반은 filled).
    gaps: (space, 시작 UTC, 칸 수) — filled=True(과제 2 보간 칸 흉내)로 표시."""
    rng = np.random.default_rng(seed)
    t = pd.date_range(start.astimezone(timezone.utc), periods=days * 288, freq="5min")
    k = t.tz_convert(KST)
    m = k.hour * 60 + k.minute
    on = (k.weekday < 5) & (((m >= 540) & (m < 690)) | ((m >= 780) & (m < 1080)))
    rows = []
    for space, amp in (("lab", 900.0), ("office", 400.0), ("meeting", 700.0)):
        co2 = np.empty(len(t))
        c = 450.0
        for i, o in enumerate(on):
            c += ((450 + amp) - c) * 0.08 if o else (450 - c) * 0.05
            co2[i] = c + rng.normal(0, 5)
        temp = 20 + 3 * np.convolve(on.astype(float), np.ones(24) / 24, "same") + rng.normal(0, 0.02, len(t))
        hum = 40 + rng.normal(0, 0.5, len(t))
        filled = np.zeros(len(t), bool)
        if space == "meeting":
            filled[1::2] = True
        for gs, g0, n in gaps:
            if gs == space:
                i0 = t.get_loc(pd.Timestamp(g0))
                filled[i0:i0 + n] = True
        for item, v in (("co2", co2), ("temperature", temp), ("humidity", hum)):
            rows.append(pd.DataFrame({"space_id": space, "item": item, "bin": t, "value": v, "n_sensors": 1,
                                      "n_observed": (~filled).astype(int), "filled": filled}))
    df = pd.concat(rows, ignore_index=True)
    df["bin"] = df["bin"].astype("datetime64[ms, UTC]")
    return df


@pytest.fixture
def series():
    return synth_series()


def synth_norm(series):
    out = {"spaces": {}}
    for (s, i), g in series[~series["filled"]].groupby(["space_id", "item"]):
        out["spaces"].setdefault(s, {})[i] = {"mean": float(g["value"].mean()), "std": float(g["value"].std()),
                                               "bins": int(len(g))}
    return out


@pytest.fixture
def norm(series):
    return synth_norm(series)

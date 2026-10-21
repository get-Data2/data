"""과제 5 시험 고정물: 짧은 합성 분할, 과제 2 space_5min 형식 합성 시계열, 관찰 CSV."""
import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from tests.conftest import ROOT

KST = timezone(timedelta(hours=9))
SYN_SPLIT = {"train": (date(2026, 3, 2), date(2026, 3, 15)), "val": (date(2026, 3, 17), date(2026, 3, 22)),
             "test": (date(2026, 3, 24), date(2026, 3, 31))}


@pytest.fixture
def scfg(tmp_path):
    from analysis.config import Split
    from state.config import StateConfig, load_state_config
    base = load_state_config(ROOT / "config" / "state.yaml")
    acfg = replace(base.acfg, splits={"syn": Split("syn", **SYN_SPLIT)})
    raw = json.loads(json.dumps(base.raw))
    raw["split"] = "syn"
    raw["observe"]["live_start"] = "2026-02-01"
    raw["inputs"] = dict(raw["inputs"], observations=str(tmp_path / "observations.csv"))
    raw["models"]["candidates"] = {"hgb": {"max_depth": [3], "learning_rate": [0.1], "class_weight": "balanced"},
                                   "logreg": {"C": [1.0], "class_weight": "balanced"},
                                   "mlp": {"hidden": [16], "layers": 2, "epochs": 3, "class_weight": "balanced"}}
    raw["models"]["seeds"] = [0, 1]
    return StateConfig(base.path, raw, acfg)


def synth_space(start=datetime(2026, 2, 28, tzinfo=KST), days=36, seed=0):
    """평일 09:00~11:30·13:00~18:00 사용. 실습실은 날마다 인원이 달라 1,000ppm을 넘는 날·안 넘는 날이 섞인다.
    회의실은 10분 주기(5분 칸 절반 filled), 하루 한 번 회의."""
    rng = np.random.default_rng(seed)
    t = pd.date_range(start.astimezone(timezone.utc), periods=days * 288, freq="5min")
    k = t.tz_convert(KST)
    m = np.asarray(k.hour * 60 + k.minute)
    wd = np.asarray(k.weekday)
    day = np.asarray((k.normalize() - k.normalize()[0]).days)
    rows = []
    for space, amp_lo, amp_hi in (("lab", 300, 1100), ("office", 150, 600), ("meeting", 200, 900)):
        amp = rng.uniform(amp_lo, amp_hi, days)
        if space == "meeting":
            on = (wd < 5) & (m >= 840) & (m < 930)
        else:
            on = (wd < 5) & (((m >= 540) & (m < 690)) | ((m >= 780) & (m < 1080)))
        co2 = np.empty(len(t))
        temp = np.empty(len(t))
        c, tp = 450.0, 20.0
        for i in range(len(t)):
            tgt = 450 + amp[day[i]] if on[i] else 450
            c += (tgt - c) * (0.08 if on[i] else 0.05)
            tp += ((22.5 if on[i] else 20.0) - tp) * 0.03
            co2[i] = c + rng.normal(0, 5)
            temp[i] = tp + rng.normal(0, 0.03)
        hum = 40 + 3 * (temp - 20) + rng.normal(0, 0.5, len(t))
        filled = np.zeros(len(t), bool)
        if space == "meeting":
            filled[1::2] = True
        for item, v in (("co2", co2), ("temperature", temp), ("humidity", hum)):
            rows.append(pd.DataFrame({"space_id": space, "item": item, "bin": t, "value": v, "n_sensors": 1,
                                      "n_observed": (~filled).astype(int), "filled": filled}))
    df = pd.concat(rows, ignore_index=True)
    df["bin"] = df["bin"].astype("datetime64[ms, UTC]")
    return df


@pytest.fixture
def space_series():
    return synth_space()


def synth_norm(series):
    out = {"spaces": {}}
    for (s, i), g in series[~series["filled"]].groupby(["space_id", "item"]):
        out["spaces"].setdefault(s, {})[i] = {"mean": float(g["value"].mean()), "std": float(g["value"].std()),
                                               "bins": int(len(g))}
    return out


def synth_obs(path, rows):
    pd.DataFrame(rows, columns=["space_id", "start", "end", "state", "people", "observer", "note"]).to_csv(path, index=False)
    return path

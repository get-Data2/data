"""과제 4 시험 고정물: 짧은 합성 분할과 센서 단위 합성 시계열."""
import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from tests.conftest import ROOT

KST = timezone(timedelta(hours=9))
SYN_SPLIT = {"train": (date(2026, 3, 2), date(2026, 3, 15)), "val": (date(2026, 3, 17), date(2026, 3, 21)),
             "test": (date(2026, 3, 23), date(2026, 4, 12))}


@pytest.fixture
def ncfg():
    from analysis.config import Split
    from anomaly.config import AnomalyConfig, load_anomaly_config
    base = load_anomaly_config(ROOT / "config" / "anomaly.yaml")
    acfg = replace(base.acfg, splits={"syn": Split("syn", **SYN_SPLIT)})
    raw = json.loads(json.dumps(base.raw))
    raw["split"] = "syn"
    raw["inject"]["per_kind"] = 2
    return AnomalyConfig(base.path, raw, acfg)


DEVICES = {"lab": [("L1", ["co2", "temperature", "humidity"]), ("L2", ["co2", "temperature", "humidity"]),
                   ("L3", ["co2", "temperature", "humidity"]), ("L4", ["temperature", "humidity"])],
           "office": [("O1", ["co2", "temperature", "humidity"]), ("O2", ["temperature", "humidity"])],
           "meeting": [("M1", ["co2", "temperature", "humidity"])]}


def synth_sensors(start=datetime(2026, 2, 28, tzinfo=KST), days=45, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start.astimezone(timezone.utc), periods=days * 288, freq="5min")
    k = t.tz_convert(KST)
    m = np.asarray(k.hour * 60 + k.minute)
    on = (np.asarray(k.weekday) < 5) & (((m >= 540) & (m < 690)) | ((m >= 780) & (m < 1080)))
    co2 = np.empty(len(t))
    c = 450.0
    for i, o in enumerate(on):
        c += (1300 - c) * 0.08 if o else (450 - c) * 0.05
        co2[i] = c
    temp = 20 + 3 * np.convolve(on.astype(float), np.ones(24) / 24, "same")
    rows = []
    for space, devs in DEVICES.items():
        for dev, items in devs:
            filled = np.zeros(len(t), bool)
            if space == "meeting":
                filled[1::2] = True
            for item in items:
                base = {"co2": co2, "temperature": temp, "humidity": 40 + 0 * temp}[item]
                noise = {"co2": 8, "temperature": 0.05, "humidity": 0.6}[item]
                v = base + rng.normal(0, noise, len(t)) + rng.normal(0, noise)
                if item == "humidity":
                    v = np.round(v * 2) / 2
                rows.append(pd.DataFrame({"device_id": dev, "space_id": space, "item": item, "bin": t, "value": v,
                                          "filled": filled}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def sensors():
    return synth_sensors()


def make_boards(ncfg, sensors):
    from anomaly.data import space_cols, wide
    a, b = ncfg.split.utc("test")
    return {s: wide(sensors, s, space_cols(sensors, s, ncfg), a, b) for s in ncfg.spaces}


@pytest.fixture
def boards(ncfg, sensors):
    return make_boards(ncfg, sensors)

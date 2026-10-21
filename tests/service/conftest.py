"""과제 8 시험 고정물: 합성 1분 원본, 가짜 원천, 가짜 예측 묶음(과제 3 묶음 형식), 시험용 서비스 설정."""
import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
import torch

from tests.conftest import ROOT

KST = timezone(timedelta(hours=9))
# 실제 장치 id를 써서 과제 1 설정의 기대 주기(회의실 600초 등)가 그대로 적용되게 한다
DEVICES = {"lab": ["24e124126d152590", "24e124128c067999"], "office": ["24e124725d081175"],
           "meeting": ["24e124725d089152"]}
PERIOD_MIN = {"24e124725d089152": 10}


def synth_raw(start=datetime(2026, 3, 4, 6, 0, tzinfo=KST), hours=8, seed=0):
    """평일 09:00 이후 CO₂ 상승. 장치마다 자기 주기로 보고(회의실 10분)."""
    rng = np.random.default_rng(seed)
    rows = []
    t = pd.date_range(start.astimezone(timezone.utc), periods=hours * 60, freq="1min")
    k = t.tz_convert(KST)
    m = np.asarray(k.hour * 60 + k.minute)
    level = np.where(m >= 540, 450 + np.minimum(m - 540, 120) * 6, 450.0)
    for space, devs in DEVICES.items():
        for j, dev in enumerate(devs):
            step = PERIOD_MIN.get(dev, 1)
            for i in range(0, len(t), step):
                ts = t[i] + pd.Timedelta(seconds=7 + j)
                rows.append((ts, space, dev, "co2", float(level[i] + rng.normal(0, 4))))
                rows.append((ts, space, dev, "temperature", float(21 + (m[i] >= 540) * 1.5 + rng.normal(0, 0.03))))
                rows.append((ts, space, dev, "humidity", float(40 + rng.normal(0, 0.4))))
    return pd.DataFrame(rows, columns=["ts", "space_id", "device_id", "item", "value"])


class FakeSource:
    def __init__(self, raw):
        self.raw = raw
        self.fail = False
        self.calls = []

    def fetch(self, since, until=None):
        self.calls.append(since)
        if self.fail:
            raise ConnectionError("시험용 DB 장애")
        d = self.raw[self.raw["ts"] >= since]
        if until is not None:
            d = d[d["ts"] < until]
        return d.copy()


@pytest.fixture
def raw():
    return synth_raw()


@pytest.fixture
def scfg(tmp_path):
    """시험용 서비스 설정: 묶음·기록 경로를 임시 폴더로."""
    from service.config import ServiceConfig, load_service_config
    base = load_service_config(ROOT / "config" / "service.yaml")
    raw = json.loads(json.dumps(base.raw))
    raw["bundles"] = dict(raw["bundles"], forecast=str(tmp_path / "fc"), anomaly=str(tmp_path / "an"),
                          state=str(tmp_path / "st"), edge_selection=str(tmp_path / "none.json"))
    raw["records"] = dict(raw["records"], dir=str(tmp_path / "svc"), spool=str(tmp_path / "svc" / "spool.db"))
    return ServiceConfig(base.path, raw, base.acfg)


def make_forecast_bundle(path, scfg, seed=0, lookback_min=60):
    """과제 3 forecast.bundle.save_bundle과 같은 형식의 작은 묶음(무작위 가중치)."""
    from forecast.bundle import input_spec
    from forecast.config import ExperimentConfig
    from forecast.models import ForecastModel
    exp = ExperimentConfig(model="gru", size={"hidden": 8, "layers": 1, "channels": 8, "kernel": 3, "head": 16,
                                              "dropout": 0.0}, lookback_min=lookback_min)
    spec = input_spec(exp)
    torch.manual_seed(seed)
    m = ForecastModel(exp, len(spec["past_columns"]), len(spec["future_columns"]), 3, 12)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), path / "model.pt")
    norm = {s: {"co2": {"mean": 700.0, "std": 300.0}, "temperature": {"mean": 21.0, "std": 2.5},
                "humidity": {"mean": 45.0, "std": 15.0}} for s in ("lab", "office", "meeting")}
    body = {"config": exp.key(), "seed": seed, "spaces": ["lab", "office", "meeting"], "norm": norm,
            "age_cap_h": 6, "input_spec": spec, "model_sha256": "test", "config_hash": exp.hash,
            "test_mae": {"lab/co2/60": 80.0}}
    (path / "bundle.json").write_text(json.dumps(body))
    return exp, norm


@pytest.fixture
def fc_bundle(scfg):
    from pathlib import Path
    return make_forecast_bundle(Path(scfg["bundles"]["forecast"]), scfg)


def at(hhmm: str, day="2026-03-04"):
    return pd.Timestamp(f"{day} {hhmm}", tz="Asia/Seoul").tz_convert("UTC")

"""학습 경로(과제 2 사본 → 과제 3 배열)와 서비스 경로(1분 원본 창 → 입력)가 같은 모델 입력을 만든다 (spec US3, tasks T007)."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from analysis.quality import apply_axis_a
from analysis.series import sensor_bins, space_series
from service.bundles import ForecastRunner
from service.prep import forecast_input, prepare
from tests.service.conftest import at, make_forecast_bundle

pytestmark = pytest.mark.unit
NO_BOOK = pd.DataFrame(columns=["space_id", "start", "end", "source"])


def training_input(scfg, raw, exp, norm, t0_bin, space):
    """과제 2(sensor_bins → space_series) → 과제 3(build_arrays → Windows.batch) 경로."""
    from analysis.config import Split
    from datetime import date
    from forecast.config import ForecastConfig, load_forecast_config
    from forecast.data import build_arrays
    from forecast.dataset import Windows
    from tests.conftest import ROOT
    acfg = scfg.acfg
    q = apply_axis_a(raw.reset_index(drop=True), acfg)
    ss = space_series(sensor_bins(raw.reset_index(drop=True), q, acfg), acfg)
    base = load_forecast_config(ROOT / "config" / "forecast.yaml")
    a2 = replace(base.acfg, splits={"syn": Split("syn", (date(2026, 3, 1), date(2026, 3, 2)),
                                                 (date(2026, 3, 3), date(2026, 3, 3)), (date(2026, 3, 4), date(2026, 3, 4)))})
    fraw = dict(base.raw, split="syn")
    fcfg = ForecastConfig(base.path, fraw, a2)
    arr = build_arrays(fcfg, ss, ss["bin"].min(), ss["bin"].max() + pd.Timedelta(hours=2), bookings=NO_BOOK)
    nj = {"spaces": {s: {i: dict(v, bins=1) for i, v in norm[s].items()} for s in norm}}
    W = Windows(arr, exp, nj, 12, torch.device("cpu"))
    b = W.batch(np.array([[arr.spaces.index(space), int(arr.index_of([t0_bin])[0])]]))
    return b["past"][0].numpy(), b["future"][0].numpy(), b["current"][0].numpy()


@pytest.mark.parametrize("space", ["lab", "office", "meeting"])
def test_model_input_identical(scfg, raw, tmp_path, space):
    exp, norm = make_forecast_bundle(tmp_path / "fcb", scfg)
    runner = ForecastRunner(tmp_path / "fcb", torch.device("cpu"), [1])
    now = at("10:03:20")
    win = raw[raw["ts"] < now]
    prep = prepare(win, scfg.acfg, now, scfg.spaces)
    assert prep.t0_bin == at("09:55")                       # 마지막 닫힌 칸 [09:55, 10:00)
    past_s, fut_s, cur_s = forecast_input(prep, runner, space, scfg.acfg)
    # 학습 경로는 전체 원본(미래 포함)으로 사본을 만들지만 입력은 t0 이전만 본다
    past_t, fut_t, cur_t = training_input(scfg, raw, exp, norm, prep.t0_bin, space)
    np.testing.assert_allclose(past_s, past_t, atol=1e-4)
    np.testing.assert_allclose(fut_s, fut_t, atol=1e-6)
    np.testing.assert_allclose(cur_s, cur_t, atol=1e-4)


def test_partial_bin_not_used(scfg, raw):
    now = at("10:03:20")
    prep = prepare(raw[raw["ts"] < now], scfg.acfg, now, scfg.spaces)
    assert prep.space["bin"].max() == at("09:55")            # [10:00, 10:05)는 아직 열린 칸
    assert prep.sensor["bin"].max() < at("10:00")


def test_error_code_excluded_from_bins(scfg, raw):
    now = at("10:03:20")
    r = raw[raw["ts"] < now].copy()
    i = r[(r["device_id"] == "24e124126d152590") & (r["item"] == "co2")].index[-1]
    r.loc[i, "value"] = 65535.0
    prep = prepare(r, scfg.acfg, now, scfg.spaces)
    assert prep.sensor[prep.sensor["item"] == "co2"]["value"].max() < 5000
    from service.rules import out_of_range
    o = out_of_range(prep.raw, prep.quality, now, 15)
    assert [x["device_id"] for x in o] == ["24e124126d152590"] and o[0]["rule"] == "A1"

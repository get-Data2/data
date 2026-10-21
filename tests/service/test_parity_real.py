"""실데이터 일치 시험: MySQL 1분 원본으로 서비스 경로와 학습 경로(과제 2 → 과제 3)가 같은 모델 입력을 만든다 (tasks T008).

실제 과제 3 묶음(var/forecast/bundle)의 입력 규약·정규화를 쓴다. 묶음이나 MySQL이 없으면 건너뛴다.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from service.bundles import ForecastRunner
from service.config import load_service_config
from service.prep import forecast_input, prepare
from tests.conftest import ROOT
from tests.service.test_preprocess_parity import training_input

pytestmark = pytest.mark.mysql
BUNDLE = ROOT / "var" / "forecast" / "bundle"
TIMES = ["2026-10-21 10:03:20", "2026-10-21 13:41:20", "2026-10-21 16:12:20"]   # KST, 수업·점심 뒤·오후


@pytest.fixture(scope="module")
def real():
    if not (BUNDLE / "bundle.json").exists():
        pytest.skip("과제 3 묶음 없음")
    from service.window import DbSource
    cfg = load_service_config(ROOT / "config" / "service.yaml")
    try:
        src = DbSource(cfg, str(ROOT / ".env"))
        a = pd.Timestamp("2026-10-21 04:00", tz="Asia/Seoul").tz_convert("UTC")
        raw = src.fetch(a, pd.Timestamp("2026-10-21 17:00", tz="Asia/Seoul").tz_convert("UTC"))
    except Exception as e:
        pytest.skip(f"MySQL 접속 불가: {type(e).__name__}")
    runner = ForecastRunner(BUNDLE, torch.device("cpu"), [1])
    return cfg, raw, runner


@pytest.mark.parametrize("when", TIMES)
@pytest.mark.parametrize("space", ["lab", "office", "meeting"])
def test_real_model_input_identical(real, when, space):
    cfg, raw, runner = real
    now = pd.Timestamp(when, tz="Asia/Seoul").tz_convert("UTC")
    win = raw[raw["ts"] < now]
    prep = prepare(win, cfg.acfg, now, cfg.spaces)
    s = forecast_input(prep, runner, space, cfg.acfg)
    assert s is not None, "실데이터 입력이 비었음"
    # 학습 경로는 t0까지의 원본으로 사본을 만든다(미래 행이 칸 평균에 섞이지 않게)
    t = training_input(cfg, win, runner.exp, runner.norm, prep.t0_bin, space)
    for a, b, tol in zip(s, t, (1e-4, 1e-6, 1e-4)):
        np.testing.assert_allclose(a, b, atol=tol)

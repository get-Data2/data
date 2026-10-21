"""과제 7 시험 고정물: 합성 작은 예측 모델(과제 3 ForecastModel)·오토인코더(과제 4 ConvAE)와 가짜 묶음."""
import json
from dataclasses import replace

import pytest
import torch

from tests.conftest import ROOT

SMALL = {"hidden": 8, "layers": 1, "channels": 8, "kernel": 3, "head": 16, "dropout": 0.0}


@pytest.fixture
def ecfg():
    from edgeopt.config import load_edgeopt_config
    return load_edgeopt_config(ROOT / "config" / "edgeopt.yaml")


def small_exp(model="gru", output="direct", target="delta", lookback=60):
    from forecast.config import ExperimentConfig
    return ExperimentConfig(model=model, size=dict(SMALL), lookback_min=lookback, output=output, target=target)


def forecast_state(exp, fp=19, ff=10, n_space=3):
    from forecast.models import ForecastModel
    torch.manual_seed(0)
    return ForecastModel(exp, fp, ff, n_space, 12).state_dict()


@pytest.fixture
def fake_bundles(tmp_path):
    """var/ 구조를 흉내 낸 임시 루트: 과제 3 묶음, 과제 4 묶음, 과제 5 트리 분류기 묶음."""
    from anomaly.autoencoder import ConvAE
    exp = small_exp()
    fb = tmp_path / "var/forecast/bundle"
    fb.mkdir(parents=True)
    torch.save(forecast_state(exp), fb / "model.pt")
    (fb / "bundle.json").write_text(json.dumps({"config": exp.key(), "seed": 0, "spaces": ["lab", "office", "meeting"],
                                                "age_cap_h": 6}))
    ab = tmp_path / "var/anomaly/bundle"
    ab.mkdir(parents=True)
    spaces = {}
    for space, n in (("lab", 4), ("office", 3), ("meeting", 3)):
        torch.save(ConvAE(2 * n + 5, n, 12, 8, 4).state_dict(), ab / f"ae_{space}.pt")
        spaces[space] = {"cols": [f"D{i}/co2" for i in range(n)], "colnorm": [1.0] * n, "stats": {},
                         "score_quantiles": list(range(1001))}
    (ab / "bundle.json").write_text(json.dumps({"window": 12, "score_tail": 3, "channels": 8, "bottleneck": 4,
                                                "spaces": spaces, "thresholds": {"ae": 0.999}}))
    sb = tmp_path / "var/state/bundle"
    sb.mkdir(parents=True)
    (sb / "bundle.json").write_text(json.dumps({"models": {"occupancy": {"model_type": "hist_gradient_boosting"},
                                                           "vent": {"model_type": "logistic"}}}))
    return tmp_path

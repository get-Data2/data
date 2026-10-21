"""과제 6 시험 고정물: 과제 3 합성 시계열·짧은 분할, 가짜 분위수 모델(가중치 없이)."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from analysis.pipeline import sha256_file
from tests.conftest import ROOT
from tests.forecast.conftest import SYN_SPLIT, synth_norm, synth_series


class FakePredictor:
    """마지막 관측값 ± 고정 폭(분위수 순서 그대로). 호출 수를 센다."""

    def __init__(self, width=50.0):
        self.width, self.calls, self.device, self.dtype, self.stats = width, 0, "cpu", "float32", {"crossed": 0, "negative": 0}

    def _last(self, ctx):
        last = np.full(len(ctx), np.nan, np.float32)
        for i, c in enumerate(ctx):
            v = c[~np.isnan(c)]
            last[i] = v[-1] if len(v) else 0.0
        return last

    def predict(self, ctx, H, quantiles):
        self.calls += 1
        last = self._last(ctx)
        q = np.array(quantiles, np.float32)
        return np.repeat(last[:, None, None], H, 1) + (q - 0.5)[None, None] * 2 * self.width

    def predict_cov(self, ctx, past, fut, H, quantiles):
        assert all(v.shape[1] == H for v in fut.values())
        return self.predict(ctx, H, quantiles) + 1.0


@pytest.fixture
def pcfg(tmp_path):
    from analysis.baseline import build as baseline_build
    from analysis.config import Split
    from pretrained.config import PretrainedConfig, load_pretrained_config
    base = load_pretrained_config(ROOT / "config" / "pretrained.yaml")
    acfg = replace(base.acfg, splits={"syn": Split("syn", **SYN_SPLIT)})
    raw = json.loads(json.dumps(base.raw))
    raw["split"] = "syn"
    raw["context_min"] = [60, 240]
    s = synth_series()
    sp = tmp_path / "series.parquet"
    s.to_parquet(sp)
    cfg = PretrainedConfig(base.path, raw, acfg)
    bl = baseline_build(s, cfg.split, acfg)
    blp, evp = tmp_path / "baseline.parquet", tmp_path / "eval.parquet"
    bl.to_parquet(blp, index=False)
    bl[["space_id", "item", "t0", "horizon_min", "y_true", "usage_hours"]].to_parquet(evp, index=False)
    np_ = tmp_path / "norm.json"
    np_.write_text(json.dumps(synth_norm(s)))
    raw["inputs"] = {"series": str(sp), "eval_points": str(evp), "eval_points_sha256": sha256_file(evp),
                     "baseline": str(blp), "norm": str(np_), "forecast_root": str(tmp_path / "forecast")}
    return cfg


@pytest.fixture
def fake():
    return FakePredictor()

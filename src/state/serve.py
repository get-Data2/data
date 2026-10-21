"""과제 8 서비스 진입점(bundle.json "service_entry": "state.serve:load")과 과제 7 변환용 MLP 읽기.

서비스는 공통 전처리(src/preprocess)로 만든 공간 5분 표(prep.space)를 넘긴다. 학습과 같은 특징 함수(space_features)를 쓴다.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.config import load_analysis_config

from .features import space_features, space_grid
from .pipeline import load_bundle


class StateService:
    def __init__(self, path, device=None):
        self.path = Path(path)
        self.b = load_bundle(self.path)
        self.acfg = load_analysis_config()
        self.version = f"state-{self.b['config']}"
        self.sha256 = hashlib.sha256((self.path / "bundle.json").read_bytes()).hexdigest()[:16]
        self.norm = {"spaces": self.b["norm"]}

    def _features(self, prep, space) -> pd.DataFrame | None:
        ser = prep.space
        if ser is None or not len(ser):
            return None
        win = self.b["feature_window_min"]
        t1 = prep.t0_bin
        w = space_grid(ser, space, t1 - pd.Timedelta(minutes=win + 60), t1)
        if w.empty:
            return None
        f = space_features(w, space, self.norm, self.acfg, window_min=win)
        return f.iloc[[-1]]

    def predict(self, prep) -> dict:
        out = {}
        for space in self.b["spaces"]:
            f = self._features(prep, space)
            if f is None or float(f["obs_ratio30"].iloc[0]) < self.b["min_input_obs"]:
                continue
            r = {}
            for prob, key_p, key_flag in (("occupancy", "p_occupied", "occupied"), ("vent", "p_vent", "ventilation_soon")):
                p = self.b["problems"].get(prob)
                if not p:
                    continue
                X = f[p["features"]].to_numpy(float)
                prob_ = float(np.mean([m.predict_proba(X)[0] for m in p["models"]]))
                r[key_p] = round(prob_, 4)
                r[key_flag] = bool(prob_ >= p["threshold"])
                r[f"{prob}_threshold"] = p["threshold"]
            out[space] = r
        return out


def load(path, device=None) -> StateService:
    return StateService(path, device)


def load_torch_model(path):
    """과제 7: 최종 분류기가 MLP면 (torch 모듈, 입력 특징 수). 아니면 ValueError(변환 대상 아님)."""
    b = load_bundle(Path(path))
    for prob, p in b["problems"].items():
        if p["kind"] == "mlp":
            m = p["models"][0]
            return m.m["net"], len(p["features"])
    raise ValueError("MLP 분류기가 없음 — 트리·선형 모델은 TensorRT 변환 대상이 아니다")

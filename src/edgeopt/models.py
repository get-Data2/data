"""묶음 → (torch 모듈, 예시 입력, 입력·출력 이름, 원 단위 복원 정보) (data-model "변환 대상 모델", research R3).

엔진 안에서는 정규화 공간까지만 계산하고, 원 단위 복원(현재값 + 출력 × 표준편차)은 엔진 밖 FP32에서 한다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

SIZE_DEFAULT = {"hidden": 64, "layers": 2, "channels": 64, "kernel": 3, "head": 128, "dropout": 0.1}


@dataclass
class Loaded:
    name: str
    kind: str
    module: nn.Module
    input_names: list[str]
    output_names: list[str]
    input_shapes: dict[str, tuple]          # 배치 축을 뺀 모양
    restore: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def example(self, batch: int, seed: int = 0) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        out = {}
        for n in self.input_names:
            shp = (batch, *self.input_shapes[n])
            if n == "space":
                v = np.zeros(shp, np.float32)
                v[np.arange(batch), np.arange(batch) % shp[1]] = 1
            elif n == "sd":
                v = rng.uniform(0.5, 2.0, shp).astype(np.float32)
            else:
                v = rng.normal(0, 1, shp).astype(np.float32)
            out[n] = v
        return out


class ForecastWrapper(nn.Module):
    """ForecastModel(dict 입력)을 위치 인자 텐서로 감싼다. 반복 예측이면 반복 안에서 쓰는 값(current·mu·sd)도 입력."""

    def __init__(self, model, has_space: bool, recursive: bool, target_kind: str, age_step: float):
        super().__init__()
        self.model, self.has_space, self.recursive = model, has_space, recursive
        self.target_kind, self.age_step = target_kind, age_step

    def forward(self, past, future, *rest):
        i = 0
        space = None
        if self.has_space:
            space = rest[i]
            i += 1
        b = {"past": past, "future": future, "space": space, "target_kind": self.target_kind, "age_step": self.age_step}
        if self.recursive:
            b |= {"current": rest[i], "mu": rest[i + 1], "sd": rest[i + 2]}
        return self.model(b)


def infer_forecast_dims(state: dict, exp, n_space_total: int = 3) -> tuple[int, int, int]:
    """가중치 모양에서 과거 특징 수·미래 특징 수·공간 수를 얻는다(묶음에 입력 규약이 없어도 동작)."""
    sz = SIZE_DEFAULT | dict(exp.size)
    if exp.model == "tcn":
        fp = state["enc.net.0.c1.weight"].shape[1]
        enc_out = sz["channels"]
    else:
        fp = state["enc.rnn.weight_ih_l0"].shape[1]
        enc_out = sz["hidden"]
    n_space = n_space_total if "space" in exp.features else 0
    steps = 1 if exp.output == "recursive" else 12
    n_head = state["head.0.weight"].shape[1]
    ff = (n_head - enc_out - n_space) // steps
    return int(fp), int(ff), int(n_space)


def forecast_module(exp, state: dict, n_space_total: int = 3, age_cap_h: float = 6.0) -> Loaded:
    from forecast.models import ForecastModel
    fp, ff, ns = infer_forecast_dims(state, exp, n_space_total)
    m = ForecastModel(exp, fp, ff, ns if ns else n_space_total, 12)
    m.load_state_dict(state)
    m.eval()
    rec = exp.output == "recursive"
    w = ForecastWrapper(m, ns > 0, rec, exp.target, 5 / 60 / age_cap_h).eval()
    L = exp.lookback_min // 5
    names, shapes = ["past", "future"], {"past": (L, fp), "future": (12, ff)}
    if ns:
        names.append("space")
        shapes["space"] = (ns,)
    if rec:
        names += ["current", "mu", "sd"]
        shapes |= {"current": (2,), "mu": (2,), "sd": (2,)}
    return Loaded("", "forecast", w, names, ["pred"], shapes,
                  restore={"target": exp.target, "how": "delta: current + pred×sd / value: mu + pred×sd"},
                  meta={"config": exp.key(), "lookback_bins": L, "fp": fp, "ff": ff, "n_space": ns})


def load_forecast(spec) -> Loaded:
    from forecast.experiments import exp_from_key
    if "config" in spec.extra:
        key = spec.extra["config"]
    else:
        key = json.loads(spec.files["bundle"].read_text())["config"]
    b = json.loads(spec.files["bundle"].read_text()) if "bundle" in spec.files else {}
    exp = exp_from_key(key, [0])
    state = torch.load(spec.files["weights"], map_location="cpu")
    ld = forecast_module(exp, state, len(b.get("spaces", [None] * 3)), b.get("age_cap_h", 6.0))
    ld.name = spec.name
    ld.meta["norm"] = b.get("norm")
    ld.meta["spaces"] = b.get("spaces")
    return ld


class AEWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)


def ae_module(n_out: int, window: int, channels: int, bottleneck: int, state: dict | None = None) -> Loaded:
    from anomaly.autoencoder import TIME_F, ConvAE
    n_in = 2 * n_out + TIME_F
    m = ConvAE(n_in, n_out, window, channels, bottleneck)
    if state is not None:
        m.load_state_dict(state)
    m.eval()
    return Loaded("", "autoencoder", AEWrapper(m).eval(), ["x"], ["recon"], {"x": (window, n_in)},
                  restore={"how": "열 오차 = (recon − z)² 창 끝 score_tail칸 평균 / colnorm, 합 → 분위수(score_quantiles)"},
                  meta={"n_out": n_out, "window": window})


def load_autoencoder(spec) -> Loaded:
    b = json.loads(spec.files["bundle"].read_text())
    space = spec.extra["space"]
    cols = b["spaces"][space]["cols"]
    ld = ae_module(len(cols), b["window"], b["channels"], b["bottleneck"],
                   torch.load(spec.files["weights"], map_location="cpu"))
    ld.name = spec.name
    ld.meta |= {"space": space, "cols": cols, "colnorm": b["spaces"][space]["colnorm"],
                "score_quantiles": b["spaces"][space].get("score_quantiles", []), "score_tail": b["score_tail"],
                "threshold_q": b.get("thresholds", {}).get("ae")}
    return ld


def load_model(spec) -> Loaded:
    if spec.kind == "forecast":
        return load_forecast(spec)
    if spec.kind == "autoencoder":
        return load_autoencoder(spec)
    return load_classifier(spec)


def load_classifier(spec) -> Loaded:
    """과제 5 MLP 분류기. 과제 5 묶음이 제공하는 로더(state.bundle.load_torch_model)를 쓴다."""
    try:
        from state.bundle import load_torch_model  # type: ignore
    except Exception as e:
        raise RuntimeError(f"과제 5 MLP 로더 없음(state.bundle.load_torch_model): {e}")
    m, n_in = load_torch_model(spec.files["bundle"].parent)
    return Loaded(spec.name, "classifier", m.eval(), ["x"], ["logits"], {"x": (n_in,)}, restore={"how": "argmax"})

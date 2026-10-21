"""모델 묶음 적재·입력 규약 검사·실행 방식 선택 (research R8).

- 예측(과제 3): var/forecast/bundle/{bundle.json, model.pt}. forecast.models.ForecastModel을 그대로 만든다.
- 이상(과제 4)·상태(과제 5): bundle.json의 "service_entry": "모듈:함수"를 부른다.
  함수는 (묶음 폴더, device) → 객체를 돌려주고, 객체는 version·sha256 속성과 predict(ctx) → {공간: 결과 dict}를 가진다.
  (각 과제가 serve 모듈을 제공한다. 없으면 그 판단은 unavailable.)
- 묶음이 없거나 규약이 맞지 않으면 그 판단만 끈다. 서비스는 계속 돈다.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Loaded:
    name: str
    ok: bool
    reason: str = ""
    obj: object = None
    version: str = ""
    sha256: str = ""
    runtime: str = ""
    info: dict = field(default_factory=dict)

    def availability(self) -> str:
        return "ok" if self.ok else f"unavailable:{self.reason}"

    def model_meta(self) -> dict:
        return {"version": self.version, "sha256": self.sha256, "runtime": self.runtime}


def _sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def choose_runtime(cfg, device) -> tuple[str, str]:
    """과제 7 결과에 검증된 TensorRT 엔진이 있으면 TensorRT, 아니면 PyTorch(GPU/CPU)."""
    torch_rt = "torch-cuda" if getattr(device, "type", "cpu") == "cuda" else "torch-cpu"
    p = Path(cfg["bundles"]["edge_selection"])
    if not p.exists():
        return torch_rt, "과제 7 결과 없음"
    try:
        sel = json.loads(p.read_text())
    except Exception as e:  # 깨진 파일이어도 서비스는 PyTorch로 돈다
        return torch_rt, f"과제 7 결과 읽기 실패: {e}"
    if sel.get("forecast", {}).get("runtime") == "tensorrt":
        try:
            import tensorrt  # noqa: F401
        except ImportError:
            return torch_rt, "TensorRT 미설치 — PyTorch로 실행"
        return torch_rt, "TensorRT 엔진 연결은 과제 7 실행 묶음 확정 뒤(현재 PyTorch)"
    return torch_rt, "과제 7 선택: PyTorch"


class ForecastRunner:
    """과제 3 묶음 → 입력 규약에 맞춘 텐서로 12칸 × (CO₂, 온도) 예측."""

    def __init__(self, path: Path, device, supported: list[int]):
        import torch
        from forecast.experiments import exp_from_key
        from forecast.models import ForecastModel
        body = json.loads((path / "bundle.json").read_text())
        spec = body["input_spec"]
        if spec["version"] not in supported:
            raise ValueError(f"입력 규약 v{spec['version']} 미지원(지원 {supported})")
        self.body = body
        self.spec = spec
        self.exp = exp_from_key(body["config"], [body.get("seed", 0)])
        self.spaces = list(body["spaces"])
        self.norm = body["norm"]
        self.age_cap = float(body["age_cap_h"])
        self.L = int(spec["lookback_bins"])
        self.H = int(spec["horizon_bins"])
        self.n_past = len(spec["past_columns"])
        self.n_future = len(spec["future_columns"])
        self.device = device
        m = ForecastModel(self.exp, self.n_past, self.n_future, len(self.spaces), self.H)
        m.load_state_dict(torch.load(path / "model.pt", map_location=device))
        self.model = m.to(device).eval()
        self.test_mae = body.get("test_mae", {})

    def mu_sd(self, space: str) -> tuple[np.ndarray, np.ndarray]:
        n = self.norm[space]
        mu = np.array([n[i]["mean"] for i in ("co2", "temperature", "humidity")], np.float32)
        sd = np.array([n[i]["std"] for i in ("co2", "temperature", "humidity")], np.float32)
        return mu, sd

    def predict(self, past: np.ndarray, fut: np.ndarray, space: str, current: np.ndarray) -> np.ndarray:
        """past (L, n_past), fut (H, n_future), current (2,) 원값 → (H, 2) 원값."""
        import torch
        mu, sd = self.mu_sd(space)
        d = self.device
        b = {"past": torch.as_tensor(past[None], device=d), "future": torch.as_tensor(fut[None], device=d),
             "space": None, "current": torch.as_tensor(current[None].astype(np.float32), device=d),
             "mu": torch.as_tensor(mu[None, :2], device=d), "sd": torch.as_tensor(sd[None, :2], device=d),
             "age_step": 5 / 60 / self.age_cap, "target_kind": self.exp.target}
        if self.spec["space_onehot"]:
            oh = np.zeros((1, len(self.spaces)), np.float32)
            oh[0, self.spaces.index(space)] = 1
            b["space"] = torch.as_tensor(oh, device=d)
        with torch.no_grad():
            out = self.model(b)
        if self.exp.target == "delta":
            y = b["current"][:, None] + out * b["sd"][:, None]
        else:
            y = b["mu"][:, None] + out * b["sd"][:, None]
        return y[0].cpu().numpy()


def load_forecast(cfg, device) -> Loaded:
    p = Path(cfg["bundles"]["forecast"])
    if not (p / "bundle.json").exists():
        return Loaded("forecast", False, "no_bundle")
    try:
        r = ForecastRunner(p, device, cfg["bundles"]["supported_input_spec"])
    except Exception as e:
        return Loaded("forecast", False, f"load_failed:{type(e).__name__}", info={"error": str(e)})
    rt, note = choose_runtime(cfg, device)
    return Loaded("forecast", True, obj=r, version=f"forecast-{r.body.get('config_hash', r.exp.hash)}",
                  sha256=r.body.get("model_sha256") or _sha16(p / "model.pt"), runtime=rt, info={"runtime_note": note})


def load_entry(name: str, cfg, device) -> Loaded:
    """과제 4·5 묶음: bundle.json의 service_entry로 적재."""
    p = Path(cfg["bundles"][name])
    if not (p / "bundle.json").exists():
        return Loaded(name, False, "no_bundle")
    try:
        body = json.loads((p / "bundle.json").read_text())
        spec = body.get("input_spec", {}).get("version", 1)
        if spec not in cfg["bundles"]["supported_input_spec"]:
            return Loaded(name, False, f"unsupported_spec:v{spec}")
        entry = body.get("service_entry")
        if not entry:
            return Loaded(name, False, "no_service_entry")
        mod, fn = entry.split(":")
        obj = getattr(importlib.import_module(mod), fn)(p, device)
    except Exception as e:
        return Loaded(name, False, f"load_failed:{type(e).__name__}", info={"error": str(e)})
    rt, _ = choose_runtime(cfg, device)
    return Loaded(name, True, obj=obj, version=getattr(obj, "version", name), sha256=getattr(obj, "sha256", ""), runtime=rt)


def load_all(cfg, device) -> dict[str, Loaded]:
    return {"forecast": load_forecast(cfg, device), "anomaly": load_entry("anomaly", cfg, device),
            "state": load_entry("state", cfg, device)}

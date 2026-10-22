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


def edge_selection(cfg) -> tuple[dict | None, str]:
    """과제 7 결과(runtime.json). 없거나 깨졌으면 (None, 이유)."""
    p = Path(cfg["bundles"]["edge_selection"])
    if not p.exists():
        return None, "과제 7 결과 없음"
    try:
        return json.loads(p.read_text()), ""
    except Exception as e:  # 깨진 파일이어도 서비스는 PyTorch로 돈다
        return None, f"과제 7 결과 읽기 실패: {e}"


def ort_forecast_path(cfg, model_sha: str) -> tuple[Path | None, str]:
    """과제 7이 ONNX Runtime CPU를 골랐고, ONNX가 지금 묶음의 가중치에서 만든 것이면 그 경로.
    아니면 (None, 이유) — 그때는 PyTorch로 돈다(runtime.json의 fallback)."""
    sel, why = edge_selection(cfg)
    if sel is None:
        return None, why
    if sel.get("runner") != "ort_cpu":
        return None, f"과제 7 선택 {sel.get('runner')} — PyTorch로 실행"
    m = (sel.get("models") or {}).get("forecast_final") or {}
    if m.get("source_weights_sha256") != model_sha:
        return None, f"ONNX가 다른 가중치에서 만들어짐({m.get('source_weights_sha256')} ≠ {model_sha}) — PyTorch로 실행"
    f = Path(cfg["bundles"]["edge_selection"]).parent / m.get("file", "")
    if not f.is_file() or _sha16(f) != m.get("sha256"):
        return None, "ONNX 파일 없음·해시 불일치 — PyTorch로 실행"
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return None, "onnxruntime 미설치 — PyTorch로 실행"
    return f, "과제 7 선택: ONNX Runtime CPU"


def prefers_cpu(cfg) -> bool:
    """과제 7이 CPU 실행을 골랐으면 장치 auto를 CPU로 둔다(CUDA 문맥을 만들지 않아 메모리 절약)."""
    sel, _ = edge_selection(cfg)
    return bool(sel) and sel.get("runner") == "ort_cpu"


class ForecastRunner:
    """과제 3 묶음 → 입력 규약에 맞춘 텐서로 12칸 × (CO₂, 온도) 예측."""

    def __init__(self, path: Path, device, supported: list[int], onnx: Path | None = None):
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
        self.ort = None
        if onnx is not None:
            import onnxruntime as ort
            self.ort = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
            self.ort_inputs = [i.name for i in self.ort.get_inputs()]

    def mu_sd(self, space: str) -> tuple[np.ndarray, np.ndarray]:
        n = self.norm[space]
        mu = np.array([n[i]["mean"] for i in ("co2", "temperature", "humidity")], np.float32)
        sd = np.array([n[i]["std"] for i in ("co2", "temperature", "humidity")], np.float32)
        return mu, sd

    def predict(self, past: np.ndarray, fut: np.ndarray, space: str, current: np.ndarray) -> np.ndarray:
        """past (L, n_past), fut (H, n_future), current (2,) 원값 → (H, 2) 원값."""
        return self.predict_many(past[None], fut[None], [space], current[None])[0]

    def predict_many(self, past: np.ndarray, fut: np.ndarray, spaces: list[str], current: np.ndarray) -> np.ndarray:
        """여러 시점을 한 번에(재생·시뮬레이션용). past (N, L, n_past), fut (N, H, n_future), current (N, 2) → (N, H, 2)."""
        import torch
        ms = [self.mu_sd(s) for s in spaces]
        mu = np.stack([m[0][:2] for m in ms])
        sd = np.stack([m[1][:2] for m in ms])
        d = self.device
        b = {"past": torch.as_tensor(past, device=d), "future": torch.as_tensor(fut, device=d),
             "space": None, "current": torch.as_tensor(current.astype(np.float32), device=d),
             "mu": torch.as_tensor(mu, device=d), "sd": torch.as_tensor(sd, device=d),
             "age_step": 5 / 60 / self.age_cap, "target_kind": self.exp.target}
        if self.spec["space_onehot"]:
            oh = np.zeros((len(spaces), len(self.spaces)), np.float32)
            oh[np.arange(len(spaces)), [self.spaces.index(s) for s in spaces]] = 1
            b["space"] = torch.as_tensor(oh, device=d)
        if self.ort is not None:
            feed = {"past": past, "future": fut, "space": b["space"].numpy() if b["space"] is not None else None}
            out = torch.as_tensor(self.ort.run(None, {n: np.ascontiguousarray(feed[n], np.float32)
                                                      for n in self.ort_inputs})[0])
        else:
            with torch.no_grad():
                out = self.model(b)
        if self.exp.target == "delta":
            y = b["current"][:, None] + out * b["sd"][:, None]
        else:
            y = b["mu"][:, None] + out * b["sd"][:, None]
        return y.cpu().numpy()


def load_forecast(cfg, device) -> Loaded:
    p = Path(cfg["bundles"]["forecast"])
    if not (p / "bundle.json").exists():
        return Loaded("forecast", False, "no_bundle")
    torch_rt = "torch-cuda" if getattr(device, "type", "cpu") == "cuda" else "torch-cpu"
    try:
        body = json.loads((p / "bundle.json").read_text())
        sha = body.get("model_sha256") or _sha16(p / "model.pt")
        onnx, note = ort_forecast_path(cfg, sha)
        try:
            r = ForecastRunner(p, device, cfg["bundles"]["supported_input_spec"], onnx)
        except Exception as e:
            if onnx is None:
                raise
            note = f"ONNX Runtime 적재 실패({type(e).__name__}) — PyTorch로 실행"
            r = ForecastRunner(p, device, cfg["bundles"]["supported_input_spec"])
    except Exception as e:
        return Loaded("forecast", False, f"load_failed:{type(e).__name__}", info={"error": str(e)})
    rt = "onnxruntime-cpu" if r.ort is not None else torch_rt
    return Loaded("forecast", True, obj=r, version=f"forecast-{r.body.get('config_hash', r.exp.hash)}",
                  sha256=sha, runtime=rt, info={"runtime_note": note})


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
    rt = "torch-cuda" if getattr(device, "type", "cpu") == "cuda" else "torch-cpu"
    return Loaded(name, True, obj=obj, version=getattr(obj, "version", name), sha256=getattr(obj, "sha256", ""), runtime=rt)


def load_all(cfg, device) -> dict[str, Loaded]:
    return {"forecast": load_forecast(cfg, device), "anomaly": load_entry("anomaly", cfg, device),
            "state": load_entry("state", cfg, device)}

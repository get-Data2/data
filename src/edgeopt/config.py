"""config/edgeopt.yaml 읽기·검증, 대상 모델 목록 해석 (contracts/config.md, data-model "변환 대상 모델")."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

RUNNERS = ("torch_cpu", "torch_gpu", "ort_cpu", "ort_cuda", "ort_trt", "trt_fp16", "trt_fp32", "trt_int8")
KINDS = ("forecast", "autoencoder", "classifier")


class EdgeConfigError(ValueError):
    def __init__(self, path, errors: list[str]):
        self.errors = errors
        super().__init__(f"{path}: " + "; ".join(errors))


@dataclass(frozen=True)
class EdgeConfig:
    path: str
    raw: dict[str, Any]

    def __getitem__(self, k):
        return self.raw[k]

    def get(self, k, default=None):
        return self.raw.get(k, default)

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


@dataclass
class ModelSpec:
    name: str
    kind: str
    source: str
    status: str = "waiting"                 # ready · waiting(묶음 없음) · unsupported(변환 불가)
    reason: str = ""
    files: dict = field(default_factory=dict)   # weights, bundle(json) 등
    extra: dict = field(default_factory=dict)


def validate(raw: dict) -> list[str]:
    errs = []
    if "torch_cpu" not in raw.get("runners", []):
        errs.append("runners: 기준 torch_cpu가 없음")
    bad = [r for r in raw.get("runners", []) + raw.get("optional_runners", []) if r not in RUNNERS]
    if bad:
        errs.append(f"runners: 알 수 없는 방식 {bad}")
    if any(int(b) <= 0 for b in raw.get("batches", [])):
        errs.append("batches: 0 이하 값")
    p = raw.get("trt", {}).get("profile", {})
    if not (p.get("min", 1) <= p.get("opt", 1) <= p.get("max", 1)):
        errs.append("trt.profile: min ≤ opt ≤ max가 아님")
    elif any(b < p["min"] or b > p["max"] for b in raw.get("batches", [])):
        errs.append("batches: trt.profile 범위 밖")
    if raw.get("bench", {}).get("repeat", 0) < 100:
        errs.append("bench.repeat: 100 이상이어야 함")
    if raw.get("int8", {}).get("calibration_samples", 0) < 100:
        errs.append("int8.calibration_samples: 100 이상이어야 함")
    for n, m in (raw.get("models") or {}).items():
        if m.get("kind") not in KINDS:
            errs.append(f"models.{n}.kind: {KINDS} 중 하나여야 함")
    return errs


def load_edgeopt_config(path: str | Path = "config/edgeopt.yaml") -> EdgeConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    errs = validate(raw)
    if errs:
        raise EdgeConfigError(path, errs)
    return EdgeConfig(str(path), raw)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def resolve_models(cfg: EdgeConfig, root: Path = Path(".")) -> list[ModelSpec]:
    """설정의 모델 목록 → 실제 파일. 묶음이 없으면 waiting으로 남긴다(나중에 같은 명령으로 채움)."""
    out = []
    for name, m in cfg["models"].items():
        spec = ModelSpec(name, m["kind"], m["source"], extra=dict(m))
        try:
            if m["kind"] == "forecast" and m["source"] == "forecast_b1":
                _resolve_b1(spec, m, root)
            elif m["kind"] == "forecast":
                b = root / m["source"]
                if (b / "bundle.json").exists() and (b / "model.pt").exists():
                    spec.files = {"bundle": b / "bundle.json", "weights": b / "model.pt"}
                    spec.status = "ready"
                else:
                    spec.reason = f"묶음 없음({b})"
            elif m["kind"] == "autoencoder":
                b = root / m["source"]
                w = b / f"ae_{m['space']}.pt"
                if (b / "bundle.json").exists() and w.exists():
                    spec.files = {"bundle": b / "bundle.json", "weights": w}
                    spec.status = "ready"
                else:
                    spec.reason = f"묶음 없음({w})"
            else:
                _resolve_classifier(spec, root / m["source"])
        except Exception as e:                      # 해석 실패가 전체를 멈추지 않는다(헌법 VI)
            spec.status, spec.reason = "waiting", f"{type(e).__name__}: {e}"
        if spec.status == "ready" and "weights" in spec.files:
            spec.extra["weights_sha256"] = _sha(spec.files["weights"])
        out.append(spec)
    return out


def _resolve_b1(spec: ModelSpec, m: dict, root: Path):
    """과제 3 B1 후보 중 최종과 같은 과거 길이의 해당 모델(씨앗 0)."""
    from forecast.experiments import exp_from_key
    fr = root / "var/forecast"
    st_p, b_p = fr / "stages.json", fr / "bundle" / "bundle.json"
    if not st_p.exists() or not b_p.exists():
        spec.reason = "과제 3 stages.json·모델 묶음 없음"
        return
    final = json.loads(b_p.read_text())["config"]
    for c in json.loads(st_p.read_text()).get("B1", {}).get("candidates", []):
        k = c["key"]
        if k["model"] == m["model"] and k["lookback_min"] == final["lookback_min"]:
            exp = exp_from_key(k, [0])
            run = fr / "runs" / f"{exp.hash}-s0"
            if (run / "model.pt").exists():
                spec.files = {"run": run / "run.json", "weights": run / "model.pt", "bundle": b_p}
                spec.extra["config"] = k
                spec.status = "ready"
                return
    spec.reason = f"B1에 과거 {final['lookback_min']}분 {m['model']} 실행 없음"


def _resolve_classifier(spec: ModelSpec, b: Path):
    """과제 5 묶음. 신경망(MLP)이면 변환 대상, 트리·선형이면 CPU만(research R1·spec FR-062)."""
    bj = b / "bundle.json"
    if not bj.exists():
        spec.reason = f"묶음 없음({b})"
        return
    body = json.loads(bj.read_text())
    kinds = {}
    for task, v in _classifier_entries(body).items():
        t = str(v.get("model_type", v.get("kind", ""))).lower()
        kinds[task] = {"model_type": t, "file": v.get("file") or v.get("model_file"),
                       "convertible": t in ("mlp", "torch", "pytorch")}
    spec.files = {"bundle": bj}
    spec.extra["classifiers"] = kinds
    if any(k["convertible"] for k in kinds.values()):
        spec.status = "ready"
    else:
        spec.status, spec.reason = "unsupported", "트리·선형 분류기 — 변환 대상 아님(CPU 그대로)"


def _classifier_entries(body: dict) -> dict:
    """bundle.json 형식 차이를 흡수: {"models": {과제: {...}}} 또는 {"occupancy": {...}, "vent": {...}} 또는 단일."""
    if isinstance(body.get("models"), dict):
        return body["models"]
    subs = {k: v for k, v in body.items() if isinstance(v, dict) and ("model_type" in v or "kind" in v)}
    return subs or {"state": body}

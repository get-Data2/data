"""과제 8용 실행 묶음 `var/edgeopt/runtime/` (data-model "실행 묶음").

runtime.json: 고른 방식·이유, 모델별 파일·해시·원본 묶음 해시·입력·출력 이름과 모양, 입력 규약 버전, 생성 환경, 결과 차이 판정.
TensorRT 엔진은 보드·버전에 묶이므로 생성 환경이 다르면 서비스가 다시 만들거나 ONNX로 내려간다(check_env).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

RUNTIME_VERSION = 1


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def write_runtime(cfg, root: Path, runner: str, specs, results: dict, reason: str, env: dict) -> dict:
    """판정이 실패한 모델이 있으면 거부한다. 판정 대상이 아닌 방식(torch_cpu 기준)은 그대로 허용."""
    out = root / "runtime"
    out.mkdir(parents=True, exist_ok=True)
    models = {}
    for spec in specs:
        if spec.status != "ready" or spec.name in ("forecast_gru", "forecast_lstm", "forecast_tcn"):
            continue
        verdict = results.get(spec.name, {}).get(runner, {}) if runner != "torch_cpu" else {"ok": True, "note": "기준"}
        if runner != "torch_cpu" and verdict.get("ok") is not True:
            raise RuntimeError(f"{spec.name}: {runner} 결과 차이 판정이 통과가 아님({verdict.get('ok', verdict.get('status'))}) — 묶음 거부")
        if runner.startswith("trt_"):
            src = root / "engines" / f"{spec.name}__{runner}.plan"
        elif runner.startswith("ort_"):
            src = root / "onnx" / f"{spec.name}.onnx"
        else:
            src = spec.files["weights"]
        dst = out / f"{spec.name}{src.suffix}"
        shutil.copy(src, dst)
        conv = json.loads((root / "convert.json").read_text()).get(spec.name, {}) if (root / "convert.json").exists() else {}
        models[spec.name] = {"kind": spec.kind, "file": dst.name, "sha256": _sha(dst),
                             "source_weights_sha256": spec.extra.get("weights_sha256"),
                             "source_bundle": str(spec.files.get("bundle", "")),
                             "inputs": conv.get("onnx", {}).get("inputs"), "outputs": conv.get("onnx", {}).get("outputs"),
                             "verdict": verdict}
    body = {"runtime_version": RUNTIME_VERSION, "runner": runner, "reason": reason, "models": models, "env": env,
            "config": cfg.digest, "criteria": cfg["criteria"],
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fallback": "엔진 역직렬화 실패·환경 불일치 시 onnx(ort_cuda) → torch_gpu → torch_cpu 순으로 내려간다"}
    (out / "runtime.json").write_text(json.dumps(body, ensure_ascii=False, indent=1, default=str))
    return body


def check_env(runtime: dict, env: dict) -> list[str]:
    """현재 환경과 엔진 생성 환경 비교 — 다르면 이유 목록(서비스가 재생성·대체 결정에 씀)."""
    diffs = []
    if runtime["runner"].startswith("trt_"):
        for k in ("tensorrt", "gpu", "cc"):
            if runtime["env"].get(k) != env.get(k):
                diffs.append(f"{k}: {runtime['env'].get(k)} → {env.get(k)}")
    return diffs

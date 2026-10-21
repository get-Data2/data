"""ONNX 내보내기와 검증 (research R3): 연산 집합 17, 배치 축 동적, 시간 축 고정, ORT CPU vs PyTorch CPU ≤ 1e-5."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

EXPORT_TOL = 1e-5


def export_onnx(ld, path: Path, opset: int = 17, batch: int = 3) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    ex = ld.example(batch)
    args = tuple(torch.from_numpy(ex[n]) for n in ld.input_names)
    axes = {n: {0: "batch"} for n in ld.input_names + ld.output_names}
    rec = {"path": str(path), "opset": opset, "inputs": {n: list(ld.input_shapes[n]) for n in ld.input_names},
           "outputs": ld.output_names}
    with torch.no_grad():
        try:
            torch.onnx.export(ld.module, args, str(path), input_names=ld.input_names, output_names=ld.output_names,
                              dynamic_axes=axes, opset_version=opset, dynamo=False)
            rec["exporter"] = "torchscript"
        except TypeError:          # dynamo 인자를 모르는 판
            torch.onnx.export(ld.module, args, str(path), input_names=ld.input_names, output_names=ld.output_names,
                              dynamic_axes=axes, opset_version=opset)
            rec["exporter"] = "default"
    rec["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    rec["verify"] = verify_onnx(ld, path)
    return rec


def torch_run(ld, inputs: dict[str, np.ndarray]) -> np.ndarray:
    with torch.no_grad():
        return ld.module(*[torch.from_numpy(inputs[n]) for n in ld.input_names]).numpy()


def verify_onnx(ld, path: Path, batches=(1, 3, 4)) -> dict:
    """ORT CPU와 PyTorch CPU 출력 최대 차이. 배치마다 확인(동적 배치 축)."""
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for b in batches:
        ex = ld.example(b, seed=b)
        ref = torch_run(ld, ex)
        out = s.run(None, {n: ex[n] for n in ld.input_names})[0]
        worst = max(worst, float(np.max(np.abs(out - ref))))
    return {"max_abs_diff": worst, "tol": EXPORT_TOL, "ok": worst <= EXPORT_TOL, "batches": list(batches)}

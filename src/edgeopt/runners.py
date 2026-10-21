"""실행 방식 7가지를 같은 모양으로: load(묶음·ONNX·엔진) → run(numpy 입력) → numpy 출력. 기준 = torch_cpu(FP32)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class Runner:
    name = ""
    needs_gpu = False

    def run(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def sync(self):
        pass


class TorchRunner(Runner):
    def __init__(self, ld, device: str, threads: int | None = None):
        import torch
        self.torch, self.ld = torch, ld
        self.name = f"torch_{'gpu' if device == 'cuda' else 'cpu'}"
        self.needs_gpu = device == "cuda"
        if threads and device == "cpu":
            torch.set_num_threads(threads)
        self.dev = torch.device(device)
        if device == "cuda":
            from forecast.env import set_determinism
            set_determinism(0)          # TF32 끔(research R1)
        self.m = ld.module.to(self.dev).eval()

    def run(self, inputs):
        t = self.torch
        with t.no_grad():
            out = self.m(*[t.from_numpy(np.ascontiguousarray(inputs[n])).to(self.dev) for n in self.ld.input_names])
            return out.cpu().numpy()

    def sync(self):
        if self.needs_gpu:
            self.torch.cuda.synchronize()


class OrtRunner(Runner):
    PROVIDERS = {"ort_cpu": ["CPUExecutionProvider"], "ort_cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                 "ort_trt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]}

    def __init__(self, name: str, onnx_path: Path, input_names: list[str], threads: int | None = None):
        import onnxruntime as ort
        self.name, self.inputs = name, input_names
        self.needs_gpu = name != "ort_cpu"
        want = self.PROVIDERS[name]
        have = ort.get_available_providers()
        if want[0] not in have:
            raise RuntimeError(f"{want[0]} 없음(사용 가능: {have})")
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.s = ort.InferenceSession(str(onnx_path), so, providers=want)
        used = self.s.get_providers()[0]
        if used != want[0]:
            raise RuntimeError(f"{want[0]} 요청했지만 {used}로 대체됨")

    def run(self, inputs):
        return self.s.run(None, {n: np.ascontiguousarray(inputs[n]) for n in self.inputs})[0]


class TrtRunnerAdapter(Runner):
    needs_gpu = True

    def __init__(self, name, engine_path, input_names, output_names):
        from .trt import TrtRunner
        self.name = name
        self.r = TrtRunner(engine_path, input_names, output_names)

    def run(self, inputs):
        return self.r.run(inputs)


def make_runner(name: str, ld, art: dict, threads: int | None = None) -> Runner:
    """art: {"onnx": 경로, "trt_fp16"·"trt_fp32"·"trt_int8": 엔진 경로}."""
    if name in ("torch_cpu", "torch_gpu"):
        return TorchRunner(ld, "cpu" if name == "torch_cpu" else "cuda", threads)
    if name.startswith("ort_"):
        if "onnx" not in art:
            raise RuntimeError("ONNX 없음 — `edgeopt export` 먼저")
        return OrtRunner(name, art["onnx"], ld.input_names, threads)
    if name.startswith("trt_"):
        if name not in art:
            raise RuntimeError(f"{name} 엔진 없음 — `edgeopt build` 먼저(또는 probe 실패)")
        return TrtRunnerAdapter(name, art[name], ld.input_names, ld.output_names)
    raise ValueError(f"알 수 없는 실행 방식 {name}")


def artifacts(root: Path, model: str) -> dict:
    a = {}
    if (root / "onnx" / f"{model}.onnx").exists():
        a["onnx"] = root / "onnx" / f"{model}.onnx"
    for k in ("trt_fp16", "trt_fp32", "trt_int8"):
        p = root / "engines" / f"{model}__{k}.plan"
        if p.exists():
            a[k] = p
    return a

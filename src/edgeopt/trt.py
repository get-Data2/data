"""TensorRT 엔진 생성·실행 (research R4·R8). FP16(·INT8) 엔진, 배치 프로필 (1/3/4), 작업 공간 상한, 생성 환경 기록.

pip `tensorrt-cu13` 휠은 SBSA용이라 Orin(sm87) 전용 빌더 자원이 없고 PTX 자원만 있다(2026-10-21 확인) —
실제 동작은 `edgeopt probe`로 확인한다.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np


def available() -> tuple[bool, str]:
    try:
        import tensorrt as trt
        return True, trt.__version__
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def env_info() -> dict:
    out = {}
    try:
        import tensorrt as trt
        out["tensorrt"] = trt.__version__
    except Exception as e:
        out["tensorrt"] = f"없음({type(e).__name__})"
    try:
        import torch
        out |= {"torch": torch.__version__, "cuda": torch.version.cuda}
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            out["cc"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    except Exception:
        pass
    try:
        out["l4t"] = Path("/etc/nv_tegra_release").read_text().split(",")[0].strip()
    except Exception:
        pass
    return out


class _Calibrator:
    """INT8 엔트로피 보정기 — 학습 기간 표본만(research R8). TensorRT에 클래스가 없으면 미지원으로 기록."""

    def __new__(cls, samples: dict[str, np.ndarray], batch: int, cache: Path):
        import tensorrt as trt
        import torch
        base = getattr(trt, "IInt8EntropyCalibrator2", None)
        if base is None:
            raise RuntimeError("이 TensorRT 판에 IInt8EntropyCalibrator2가 없음(명시적 양자화만 지원)")

        class Cal(base):
            def __init__(self):
                base.__init__(self)
                self.samples, self.batch, self.cache, self.i = samples, batch, cache, 0
                self.n = len(next(iter(samples.values())))
                self.buf = {}

            def get_batch_size(self):
                return self.batch

            def get_batch(self, names):
                if self.i + self.batch > self.n:
                    return None
                ptrs = []
                for nm in names:
                    t = torch.from_numpy(self.samples[nm][self.i:self.i + self.batch]).cuda().contiguous()
                    self.buf[nm] = t
                    ptrs.append(int(t.data_ptr()))
                self.i += self.batch
                return ptrs

            def read_calibration_cache(self):
                return self.cache.read_bytes() if self.cache.exists() else None

            def write_calibration_cache(self, data):
                self.cache.write_bytes(bytes(data))

        return Cal()


def build_engine(onnx_path: Path, engine_path: Path, input_shapes: dict[str, tuple], profile: dict, workspace_mb: int,
                 fp16: bool = True, int8: bool = False, calib: dict | None = None, fp32_last_layer: bool = False) -> dict:
    """ONNX → 직렬화 엔진. 실패하면 단계와 오류를 담아 돌려준다(예외를 밖으로 던지지 않음)."""
    rec = {"onnx": str(onnx_path), "engine": str(engine_path), "fp16": fp16, "int8": int8, "workspace_mb": workspace_mb,
           "profile": profile, "env": env_info(), "ok": False}
    t0 = time.time()
    stage = "import"
    try:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        stage = "builder"
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        stage = "parse"
        if not parser.parse(Path(onnx_path).read_bytes()):
            raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif hasattr(trt.BuilderFlag, "TF32"):
            config.clear_flag(trt.BuilderFlag.TF32)       # FP32 엔진은 TF32도 끈다(학습·과제 3과 같은 정밀도)
        if int8:
            config.set_flag(trt.BuilderFlag.INT8)
            stage = "calibrator"
            config.int8_calibrator = _Calibrator(calib["samples"], calib["batch"], Path(str(engine_path) + ".calib"))
        if fp32_last_layer:
            stage = "precision"
            last = network.get_layer(network.num_layers - 1)
            last.precision = trt.float32
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        prof = builder.create_optimization_profile()
        for name, shp in input_shapes.items():
            prof.set_shape(name, (profile["min"], *shp), (profile["opt"], *shp), (profile["max"], *shp))
        config.add_optimization_profile(prof)
        stage = "build"
        ser = builder.build_serialized_network(network, config)
        if ser is None:
            raise RuntimeError("build_serialized_network가 None을 돌려줌(로그 확인)")
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(bytes(ser))
        rec |= {"ok": True, "sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest()[:16],
                "size_mb": round(engine_path.stat().st_size / 2**20, 2)}
    except Exception as e:
        rec |= {"stage": stage, "error": f"{type(e).__name__}: {e}"}
    rec["build_seconds"] = round(time.time() - t0, 1)
    if rec["ok"]:
        Path(str(engine_path) + ".json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str))
    return rec


class TrtRunner:
    """직렬화 엔진 실행: 장치 버퍼(torch CUDA 텐서) + execute_async_v3 + 스트림 동기화(research R5)."""

    def __init__(self, engine_path: Path, input_names: list[str], output_names: list[str]):
        import tensorrt as trt
        import torch
        self.torch = torch
        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.rt.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError("엔진 역직렬화 실패(생성 환경과 버전이 다를 수 있음)")
        self.ctx = self.engine.create_execution_context()
        self.inputs, self.outputs = input_names, output_names
        self.stream = torch.cuda.Stream()

    def device_memory_mb(self) -> float:
        try:
            return round(self.engine.device_memory_size_v2 / 2**20, 2)
        except Exception:
            return float("nan")

    def run_device(self, dev_inputs: dict) -> dict:
        """이미 장치에 있는 입력(torch 텐서)으로 실행하고 장치 출력을 돌려준다."""
        torch = self.torch
        for n in self.inputs:
            t = dev_inputs[n]
            self.ctx.set_input_shape(n, tuple(t.shape))
            self.ctx.set_tensor_address(n, int(t.data_ptr()))
        outs = {}
        for n in self.outputs:
            shp = tuple(self.ctx.get_tensor_shape(n))
            o = torch.empty(shp, dtype=torch.float32, device="cuda")
            self.ctx.set_tensor_address(n, int(o.data_ptr()))
            outs[n] = o
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return outs

    def run(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        torch = self.torch
        dev = {n: torch.from_numpy(np.ascontiguousarray(inputs[n])).cuda() for n in self.inputs}
        return self.run_device(dev)[self.outputs[0]].cpu().numpy()

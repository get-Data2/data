"""장치 선택·결정성 설정·실행 환경 기록 (research R1)."""
from __future__ import annotations

import os
import random
import resource
import warnings

import numpy as np
import torch

# Orin(CC 8.7)이 cu130판 공식 목록에 없다는 경고 — CPU와 수치 일치를 확인했다(research R1)
warnings.filterwarnings("ignore", message="Found GPU0")


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "cpu" or (pref == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU를 쓸 수 없음 — `id`로 video·render 그룹을 확인하거나 tools/forecast로 실행")
    return torch.device("cuda")


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # TF32를 끈다: 합성곱이 GPU·CPU 사이 5.5e-4 차이를 냈다(2026-10-21 실측)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def env_info(device: torch.device) -> dict:
    info = {"device": device.type, "torch": torch.__version__, "cuda": torch.version.cuda,
            "deterministic": True, "tf32": False}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def peak_memory(device: torch.device) -> dict:
    m = {"rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    if device.type == "cuda":
        m["gpu_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return m

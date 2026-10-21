"""지연 측정 (research R5): 동기화 포함 측정 루프, 통계, 측정 전 검사(우리 학습 프로세스·GPU 경합·온도).

조합마다 새 프로세스(`python -m edgeopt _bench_one ...`)에서 잰다 — 앞 방식이 올린 라이브러리 메모리가 섞이지 않게.
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# 측정을 거부할 우리 학습 프로세스(명령줄 일부)
TRAINING_PATTERNS = ("-m forecast run", "-m forecast train", "-m anomaly train", "-m anomaly all", "-m state train",
                     "-m state all", "-m pretrained predict", "-m pretrained bench")


def latency_stats(ms: np.ndarray, chunks: int = 3) -> dict:
    ms = np.asarray(ms, float)
    parts = np.array_split(ms, chunks) if chunks > 1 else [ms]
    return {"n": int(len(ms)), "mean": float(ms.mean()), "median": float(np.median(ms)),
            "p95": float(np.percentile(ms, 95)), "p99": float(np.percentile(ms, 99)), "max": float(ms.max()),
            "chunk_p95": [float(np.percentile(p, 95)) for p in parts if len(p)]}


def process_cmdlines() -> list[tuple[int, str, str]]:
    """(pid, 사용자 uid, 명령줄) — /proc에서 읽는다."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore").strip()
            uid = (d / "status").read_text().split("Uid:")[1].split()[0]
        except Exception:
            continue
        if cmd:
            out.append((int(d.name), uid, cmd))
    return out


def our_training(procs=None, uid: str | None = None) -> list[str]:
    uid = uid if uid is not None else str(os.getuid())
    procs = process_cmdlines() if procs is None else procs
    return [c for pid, u, c in procs if u == uid and pid != os.getpid() and any(p in c for p in TRAINING_PATTERNS)]


def other_users_gpu(procs=None, uid: str | None = None) -> list[str]:
    """다른 계정 프로세스 중 GPU를 쓸 법한 것(python·torch 등) — 막을 수 없으므로 기록만."""
    uid = uid if uid is not None else str(os.getuid())
    procs = process_cmdlines() if procs is None else procs
    return [c[:120] for _p, u, c in procs if u != uid and u != "0" and "python" in c]


def contention(idle_gr3d_mean: float, limit_pct: float) -> bool:
    return bool(not np.isnan(idle_gr3d_mean) and idle_gr3d_mean > limit_pct)


def measure(runner, inputs: dict, warmup: int, repeat: int) -> np.ndarray:
    """한 번 = 호스트 입력 → 실행 → (GPU면) 동기화 → 호스트 출력. perf_counter_ns."""
    for _ in range(warmup):
        runner.run(inputs)
    runner.sync()
    t = np.empty(repeat)
    for i in range(repeat):
        s = time.perf_counter_ns()
        runner.run(inputs)
        runner.sync()
        t[i] = (time.perf_counter_ns() - s) / 1e6
    return t


def bench_one(cfg, model_name: str, runner_name: str, batch: int, threads: int | None, root: Path) -> dict:
    """현재 프로세스에서 한 조합을 잰다(_bench_one이 부름). tegrastats 대기·측정 구간을 함께 기록."""
    from .config import resolve_models
    from .models import load_model
    from .runners import artifacts, make_runner
    from .tegra import Tegrastats, summarize
    b = cfg["bench"]
    spec = next(s for s in resolve_models(cfg) if s.name == model_name)
    rec = {"runner": runner_name, "model": model_name, "batch": batch, "threads": threads, "warmup": b["warmup"],
           "repeat": b["repeat"], "chunks": b["chunks"],
           "conditions": {"power_mode": _power_mode(), "clocks_locked": False, "started_at": datetime.now().isoformat(timespec="seconds"),
                          "other_gpu_processes": other_users_gpu()}}
    try:
        ld = load_model(spec)
        runner = make_runner(runner_name, ld, artifacts(root, model_name), threads)
        inputs = ld.example(batch)
        tlog = root / "tegra" / f"{runner_name}__{model_name}__b{batch}__t{threads}.log"
        with Tegrastats(tlog) as tg:
            i0 = datetime.now()
            time.sleep(b["idle_s"])
            i1 = r0 = datetime.now()
            ms = measure(runner, inputs, b["warmup"], b["repeat"])
            r1 = datetime.now()
            time.sleep(1.2)
            frame = tg.frame()
        rec["latency_ms"] = latency_stats(ms, b["chunks"])
        rec["power"] = summarize(frame, (i0, i1), (r0, r1), b["repeat"])
        rec["conditions"]["contention"] = contention(rec["power"].get("idle_gr3d_mean", np.nan), b["contention_gr3d_pct"])
        rec["memory"] = {"rss_peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
        if runner.needs_gpu:
            try:
                import torch
                rec["memory"]["gpu_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            except Exception:
                pass
            if hasattr(runner, "r") and hasattr(runner.r, "device_memory_mb"):
                rec["memory"]["trt_device_mb"] = runner.r.device_memory_mb()
        rec["status"] = "ok"
    except Exception as e:
        rec["status"], rec["error"] = "failed", f"{type(e).__name__}: {e}"
    return rec


def _power_mode() -> str:
    try:
        return subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5).stdout.splitlines()[0].strip()
    except Exception:
        return "unknown"


def run_bench(cfg, root: Path, runners: list[str], models: list[str], batches: list[int], log=print) -> list[dict]:
    """조합마다 새 프로세스. 우리 학습 프로세스가 돌면 거부, 경합이면 max_retry까지 다시 잰다."""
    busy = our_training()
    if busy:
        raise RuntimeError(f"우리 학습 프로세스가 실행 중 — 측정 거부: {busy[:2]}")
    out_dir = root / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for m in models:
        for r in runners:
            thread_opts = cfg["bench"]["cpu_threads"] if r in ("torch_cpu", "ort_cpu") else [None]
            for b in batches:
                for th in thread_opts:
                    for attempt in range(cfg["bench"]["max_retry"]):
                        cmd = [sys.executable, "-m", "edgeopt", "--config", cfg.path, "_bench_one", "--model", m,
                               "--runner", r, "--batch", str(b)] + (["--threads", str(th)] if th else [])
                        p = subprocess.run(cmd, capture_output=True, text=True)
                        try:
                            rec = json.loads(p.stdout.strip().splitlines()[-1])
                        except Exception:
                            rec = {"runner": r, "model": m, "batch": b, "threads": th, "status": "failed",
                                   "error": (p.stderr or "")[-500:]}
                        if not rec.get("conditions", {}).get("contention"):
                            break
                        log(f"  경합(대기 GR3D 높음) — 다시 잼 {attempt + 1}")
                    rec["attempts"] = attempt + 1
                    name = f"{r}__{m}__b{b}" + (f"__t{th}" if th else "")
                    (out_dir / f"{name}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str))
                    lat = rec.get("latency_ms", {})
                    log(f"  {name}: {rec['status']} 평균 {lat.get('mean', float('nan')):.3f}ms p95 {lat.get('p95', float('nan')):.3f}ms")
                    recs.append(rec)
    return recs

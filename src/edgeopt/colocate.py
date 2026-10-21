"""심화 2: 동시 적재 메모리와 로컬 LLM 조합 (research R9).

LLM 크기는 아직 정해지지 않아 가정(1B·3B·7B, 4비트)으로 판단한다(2026-10-21 사용자 결정). 실제 모델이 정해지면 다시 잰다.
"""
from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd


def llm_weights_gb(size_b: float, bits: int) -> float:
    return size_b * bits / 8


def llm_table(cfg, service_mb: float, board_total_mb: float = 7486, other_mb: float = 0.0) -> pd.DataFrame:
    """남는 메모리 = 보드 − OS·수집기 예약 − 서비스(측정) − 다른 계정(관찰). LLM 필요 = 가중치 + 여유."""
    c = cfg["colocate"]
    reserve = sum(c["reserve_gb"].values()) * 1024
    free = board_total_mb - reserve - service_mb - other_mb
    rows = []
    for s in c["llm_sizes_b"]:
        need = (llm_weights_gb(s, c["llm_bits"]) + c["llm_overhead_gb"]) * 1024
        rows.append({"LLM": f"{s}B {c['llm_bits']}비트", "필요(MB, 가정)": round(need), "남는 메모리(MB)": round(free),
                     "여유(MB)": round(free - need), "가능": "✅" if free - need >= 0 else "❌"})
    return pd.DataFrame(rows)


def measure_colocated(cfg, root: Path, runner_name: str, repeat: int = 200, log=print) -> dict:
    """서비스 구성대로 준비된 모델을 한 프로세스에 모두 올리고 배치 1로 한 주기씩 돌린다."""
    from .config import resolve_models
    from .models import load_model
    from .runners import artifacts, make_runner
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    loaded = []
    for spec in resolve_models(cfg):
        if spec.status != "ready" or spec.name in ("forecast_gru", "forecast_lstm", "forecast_tcn"):
            continue
        try:
            ld = load_model(spec)
            loaded.append((spec.name, ld, make_runner(runner_name, ld, artifacts(root, spec.name), None)))
        except Exception as e:
            log(f"  {spec.name}: 적재 실패 — {e}")
    lat = {n: [] for n, _l, _r in loaded}
    cyc = []
    for _ in range(repeat):
        s = time.perf_counter_ns()
        for n, ld, r in loaded:
            a = time.perf_counter_ns()
            r.run(ld.example(1))
            r.sync()
            lat[n].append((time.perf_counter_ns() - a) / 1e6)
        cyc.append((time.perf_counter_ns() - s) / 1e6)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    rec = {"runner": runner_name, "models": [n for n, _l, _r in loaded], "rss_mb": round(rss, 1),
           "rss_delta_mb": round(rss - rss0, 1), "latency_ms_batch1": {n: float(np.mean(v)) for n, v in lat.items()},
           "cycle_ms_mean": float(np.mean(cyc)) if cyc else float("nan"),
           "cycle_ms_p95": float(np.percentile(cyc, 95)) if cyc else float("nan")}
    rec["llm_table"] = llm_table(cfg, rec["rss_mb"]).to_dict("records")
    p = root / "colocate.json"
    allrec = json.loads(p.read_text()) if p.exists() else {}
    allrec[runner_name] = rec
    p.write_text(json.dumps(allrec, ensure_ascii=False, indent=1, default=str))
    return rec

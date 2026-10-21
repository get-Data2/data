"""비용 측정 하네스: 지연(워밍업·반복·동기화), 메모리, tegrastats 전력, 측정 조건 (research R7). 과제 7도 재사용한다."""
from __future__ import annotations

import re
import resource
import subprocess
import threading
import time

import numpy as np

_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
_GR3D = re.compile(r"GR3D_FREQ (\d+)%")
_RAIL = re.compile(r"(VDD_[A-Z_]+) (\d+)mW/(\d+)mW/(\d+)mW")


def parse_tegrastats(line: str) -> dict:
    """tegrastats 한 줄 → RAM(MB)·GPU 사용률(%)·전력 레일 현재값(mW)."""
    d = {}
    m = _RAM.search(line)
    if m:
        d["ram_used_mb"], d["ram_total_mb"] = int(m.group(1)), int(m.group(2))
    m = _GR3D.search(line)
    if m:
        d["gr3d_pct"] = int(m.group(1))
    for name, cur, _avg, _mx in _RAIL.findall(line):
        d[name.lower() + "_mw"] = int(cur)
    return d


class TegraStats:
    """측정 구간 동안 tegrastats를 띄워 줄을 모은다(sudo 불필요, 2026-10-21 확인)."""

    def __init__(self, interval_ms: int = 100):
        self.interval_ms, self.lines, self.proc, self._t = interval_ms, [], None, None

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval_ms)], stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, text=True)
        except FileNotFoundError:
            self.proc = None
            return self
        self._t = threading.Thread(target=self._read, daemon=True)
        self._t.start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            self.lines.append(line.strip())

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=5)
            if self._t:
                self._t.join(timeout=2)

    def summary(self) -> dict:
        rows = [parse_tegrastats(x) for x in self.lines]
        rows = [r for r in rows if r]
        if not rows:
            return {"samples": 0}
        keys = sorted({k for r in rows for k in r if k != "ram_total_mb"})
        out = {"samples": len(rows)}
        for k in keys:
            v = [r[k] for r in rows if k in r]
            out[f"{k}_mean"] = round(float(np.mean(v)), 1)
            out[f"{k}_max"] = float(np.max(v))
        return out


def latency(fn, warmup: int, repeats: int, sync=None) -> dict:
    """fn()을 워밍업 뒤 반복 실행해 ms 통계. GPU면 sync로 매 회 동기화한다."""
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    t = []
    for _ in range(repeats):
        a = time.perf_counter()
        fn()
        if sync:
            sync()
        t.append((time.perf_counter() - a) * 1000)
    t = np.array(t)
    return {"mean": round(float(t.mean()), 3), "p50": round(float(np.percentile(t, 50)), 3),
            "p95": round(float(np.percentile(t, 95)), 3), "max": round(float(t.max()), 3), "repeats": repeats, "warmup": warmup}


def peak_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def conditions() -> dict:
    """측정 조건: 전력 모드, 클럭 고정 여부(sudo 없어 고정 안 함), 동시 실행 고부하 프로세스."""
    out = {"clocks_locked": False, "clocks_note": "jetson_clocks는 sudo 필요 — 고정하지 않음"}
    try:
        q = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"NV Power Mode: (\S+)", q)
        out["power_mode"] = m.group(1) if m else q.strip()[:40]
    except Exception as e:
        out["power_mode"] = f"확인 불가({type(e).__name__})"
    try:
        ps = subprocess.run(["ps", "-eo", "user,pid,rss,pcpu,comm", "--sort=-rss"], capture_output=True, text=True,
                            timeout=5).stdout.splitlines()[1:9]
        out["top_processes"] = [" ".join(x.split()) for x in ps]
    except Exception:
        out["top_processes"] = []
    out["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return out

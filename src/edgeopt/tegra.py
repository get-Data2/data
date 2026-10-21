"""tegrastats 기록·해석 (research R6). sudo 없이 동작한다(2026-10-21 확인).

줄 형식 예: `10-21-2026 14:53:52 RAM 4381/7486MB ... GR3D_FREQ 81% cpu@61.218C ... VDD_IN 15231mW/15231mW/15231mW ...`
전력 값은 `현재/평균/최대` mW다. 구간 계산에는 현재값을 쓴다.
"""
from __future__ import annotations

import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RAILS = ("VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC")
_TS = re.compile(r"^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})")
_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
_GR3D = re.compile(r"GR3D_FREQ (\d+)%")
_TEMP = re.compile(r"\b(cpu|gpu|tj)@([-\d.]+)C")
_RAIL = re.compile(r"\b(VDD_[A-Z_]+) (\d+)mW/(\d+)mW/(\d+)mW")
_CPU = re.compile(r"CPU \[([^\]]*)\]")


def parse_line(line: str) -> dict | None:
    m = _TS.match(line.strip())
    if not m:
        return None
    r = {"t": datetime.strptime(m.group(1), "%m-%d-%Y %H:%M:%S")}
    if (x := _RAM.search(line)):
        r["ram_used_mb"], r["ram_total_mb"] = int(x.group(1)), int(x.group(2))
    if (x := _GR3D.search(line)):
        r["gr3d_pct"] = int(x.group(1))
    for name, v in _TEMP.findall(line):
        r[f"temp_{name}_c"] = float(v)
    for name, cur, _avg, _mx in _RAIL.findall(line):
        r[f"{name.lower()}_mw"] = int(cur)
    if (x := _CPU.search(line)):
        loads = [int(p.split("%")[0]) for p in x.group(1).split(",") if "%" in p]
        r["cpu_mean_pct"] = float(np.mean(loads)) if loads else np.nan
    return r


def parse_log(text: str) -> pd.DataFrame:
    rows = [r for r in (parse_line(l) for l in text.splitlines()) if r]
    return pd.DataFrame(rows)


def window(df: pd.DataFrame, a: datetime, b: datetime) -> pd.DataFrame:
    return df[(df["t"] >= a) & (df["t"] <= b)]


def summarize(df: pd.DataFrame, idle: tuple, run: tuple, n_calls: int) -> dict:
    """대기 구간 평균 대비 측정 구간 평균의 증가분, 1회당 에너지(mJ) = 증가분(mW) × 측정 시간(s) / 호출 수."""
    i, r = window(df, *idle), window(df, *run)
    out = {"idle_samples": len(i), "run_samples": len(r)}
    for rail in RAILS:
        c = f"{rail.lower()}_mw"
        if c in df:
            out[f"idle_{c}"] = float(i[c].mean()) if len(i) else np.nan
            out[f"run_{c}"] = float(r[c].mean()) if len(r) else np.nan
    dur = (run[1] - run[0]).total_seconds()
    delta = out.get("run_vdd_in_mw", np.nan) - out.get("idle_vdd_in_mw", np.nan)
    out |= {"delta_vdd_in_mw": delta, "run_seconds": dur,
            "energy_mj_per_call": delta * dur / n_calls if n_calls else np.nan,
            "idle_gr3d_mean": float(i["gr3d_pct"].mean()) if len(i) and "gr3d_pct" in i else np.nan,
            "run_gr3d_mean": float(r["gr3d_pct"].mean()) if len(r) and "gr3d_pct" in r else np.nan,
            "ram_idle_mb": float(i["ram_used_mb"].mean()) if len(i) else np.nan,
            "ram_run_max_mb": float(r["ram_used_mb"].max()) if len(r) else np.nan}
    for k in ("cpu", "gpu", "tj"):
        c = f"temp_{k}_c"
        if c in df and len(r):
            out[f"temp_{k}_start"], out[f"temp_{k}_end"] = float(r[c].iloc[0]), float(r[c].iloc[-1])
    return out


class Tegrastats:
    """측정 동안 백그라운드로 기록하고 끝나면 멈춘다(우리가 띄운 프로세스만 종료)."""

    def __init__(self, logfile: Path, interval_ms: int = 1000):
        self.logfile, self.interval = Path(logfile), interval_ms
        self.p = None

    def __enter__(self):
        self.logfile.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.logfile, "w")
        self.p = subprocess.Popen(["tegrastats", "--interval", str(self.interval)], stdout=self.fh, stderr=subprocess.DEVNULL)
        time.sleep(self.interval / 1000 * 1.5)
        return self

    def __exit__(self, *exc):
        if self.p:
            self.p.terminate()
            try:
                self.p.wait(5)
            except subprocess.TimeoutExpired:
                self.p.kill()
        self.fh.close()

    def frame(self) -> pd.DataFrame:
        self.fh.flush()
        return parse_log(self.logfile.read_text())

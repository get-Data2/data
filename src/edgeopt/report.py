"""결과 표·그래프(edgeopt report). 결론·해석이 들어가는 본문은 사람이 쓰고, 여기서는 숫자 원본을 만든다.

- docs/deliverables/task7/tables/  지연·자원 표, 결과 차이 판정표, 변환 기록
- docs/deliverables/task7/figures/ 방식별 평균·p95 지연, 1회당 에너지
- docs/advanced/task7/tables/      INT8, 동시 적재·LLM 조합
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pipeline import _table

from .diff import verdict_table

DELIV = Path("docs/deliverables/task7")
ADV = Path("docs/advanced/task7")


def bench_frame(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((root / "bench").glob("*.json")):
        r = json.loads(p.read_text())
        lat, pw, mem = r.get("latency_ms", {}), r.get("power", {}), r.get("memory", {})
        rows.append({"model": r.get("model"), "runner": r.get("runner"), "batch": r.get("batch"), "threads": r.get("threads"),
                     "status": r.get("status"), "mean_ms": lat.get("mean"), "p95_ms": lat.get("p95"), "p99_ms": lat.get("p99"),
                     "rss_mb": mem.get("rss_peak_mb"), "gpu_mb": mem.get("gpu_alloc_mb", mem.get("trt_device_mb")),
                     "delta_mw": pw.get("delta_vdd_in_mw"), "mj_per_call": pw.get("energy_mj_per_call"),
                     "gr3d": pw.get("run_gr3d_mean"), "contention": r.get("conditions", {}).get("contention"),
                     "error": r.get("error", "")[:80]})
    return pd.DataFrame(rows)


def conditions(root: Path) -> list[str]:
    conv = json.loads((root / "convert.json").read_text()) if (root / "convert.json").exists() else {}
    env = conv.get("_env", {})
    return ["> **측정 조건**",
            f"> - 보드: Jetson Orin Nano Super, L4T {env.get('l4t', '?')}, 전력 모드 25W, **클럭 고정 안 함**(sudo 없음 — jetson_clocks 불가)",
            f"> - torch {env.get('torch', '?')}(CUDA {env.get('cuda', '?')}), TensorRT {env.get('tensorrt', '?')}, ONNX Runtime 1.31.0",
            "> - 지연: 호스트 입력 → 실행 → 동기화 → 호스트 출력, 워밍업 50·반복 1,000·3구간. 조합마다 새 프로세스. 전력은 tegrastats 1초 간격(대기 10초 대비 증가분)",
            "> - 결과 차이: 과제 3 시험 표본 전체, 기준 PyTorch CPU FP32, 원 단위 복원은 엔진 밖 FP32",
            "> - 다시 만들기: `edgeopt report`", ""]


def plot_latency(df: pd.DataFrame, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Noto Sans CJK KR"
    files = []
    ok = df[(df["status"] == "ok") & ((df["threads"].isna()) | (df["threads"] == 6))]
    if ok.empty:
        return files
    out.mkdir(parents=True, exist_ok=True)
    for model, g in ok.groupby("model"):
        fig, ax = plt.subplots(figsize=(8, 3.6))
        piv = g.pivot_table(index="runner", columns="batch", values="p95_ms")
        piv.plot.bar(ax=ax)
        ax.set_ylabel("p95 지연(ms)")
        ax.set_title(f"{model} — 실행 방식 × 배치 p95 지연")
        ax.set_yscale("log")
        fig.tight_layout()
        name = f"latency_{model}.png"
        fig.savefig(out / name, dpi=110)
        plt.close(fig)
        files.append(name)
    return files


def write_reports(root: Path = Path("var/edgeopt"), deliv: Path = DELIV, adv: Path = ADV) -> list[Path]:
    tdir, fdir, adir = deliv / "tables", deliv / "figures", adv / "tables"
    for d in (tdir, fdir, adir):
        d.mkdir(parents=True, exist_ok=True)
    cond = conditions(root)
    written = []
    df = bench_frame(root)
    res = json.loads((root / "results.json").read_text()) if (root / "results.json").exists() else {}
    figs = plot_latency(df, fdir) if len(df) else []
    show = df.round({"mean_ms": 3, "p95_ms": 3, "p99_ms": 3, "delta_mw": 0, "mj_per_call": 3, "gr3d": 0}) if len(df) else df
    probe = json.loads((root / "probe.json").read_text()) if (root / "probe.json").exists() else {}
    md = ["# 과제 7 결과 표 ① 실행 방식 × 모델 (자동 생성)", "", *cond,
          "## 동작 확인(probe)", "", _table(pd.DataFrame(probe.get("rows", []))) if probe.get("rows") else "없음", "",
          "## 결과 차이 판정(확인 기준 1)", "", _table(verdict_table(res)) if res else "없음", "",
          "## 지연·메모리·전력", "", _table(show) if len(show) else "측정 없음", "",
          *[f"![{f}](../figures/{f})" for f in figs], ""]
    p = tdir / "01-runners.md"
    p.write_text("\n".join(md), encoding="utf-8")
    written.append(p)
    int8 = df[df["runner"] == "trt_int8"] if len(df) else df
    md = ["# 과제 7 심화 결과 표 ① INT8 (자동 생성)", "", *cond, _table(int8) if len(int8) else "측정 없음", ""]
    p = adir / "01-int8.md"
    p.write_text("\n".join(md), encoding="utf-8")
    written.append(p)
    col = json.loads((root / "colocate.json").read_text()) if (root / "colocate.json").exists() else {}
    md = ["# 과제 7 심화 결과 표 ② 동시 적재·로컬 LLM 조합 (자동 생성)", "", *cond,
          "> LLM 크기는 가정(1B·3B·7B, 4비트, 여유 1GB)이다(2026-10-21 사용자 결정).", ""]
    for rn, r in col.items():
        md += [f"## {rn}", "", f"- 모델: {', '.join(r['models'])}", f"- 프로세스 최대 RSS {r['rss_mb']}MB, 한 주기 평균 {r['cycle_ms_mean']:.2f}ms · p95 {r['cycle_ms_p95']:.2f}ms",
               "", _table(pd.DataFrame(r["llm_table"])), ""]
    p = adir / "02-colocation.md"
    p.write_text("\n".join(md), encoding="utf-8")
    written.append(p)
    return written

"""결과 표·그래프(pretrained report). 채택 의견 등 해석은 사람이 쓰고, 여기서는 숫자 원본을 만든다.

- docs/deliverables/task6/tables/ 비교표·적중률·비용·판단표·예산표, figures/
- docs/advanced/task6/tables/     공변량·섞은 예측
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.baseline import METHODS, SPACE_NAME
from analysis.pipeline import _table

from .budget import budget_table
from .config import ROOT
from .metrics import calibration, comparison_table, coverage_table
from .pipeline import eval_frames
from .selection import require

DELIV = Path("docs/deliverables/task6")
ADV = Path("docs/advanced/task6")


def _f(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{nd}f}"


def conditions(cfg, sel, root: Path = ROOT) -> list[str]:
    sp = cfg.split
    rec = json.loads((root / "models.json").read_text()) if (root / "models.json").exists() else {}
    revs = ", ".join(f"{k}@{v['revision'][:10]}" for k, v in rec.items())
    return ["> **측정 조건**",
            f"> - 실데이터(과제 2 공간 5분 시계열, CO₂). 시험 {sp.test[0]} ~ {sp.test[1]}(KST), 평가 시점 파일 sha256 "
            f"`{cfg['inputs']['eval_points_sha256'][:16]}…`(과제 2·3과 같은 행)",
            f"> - 추가 학습 없음. 크기·컨텍스트·앙상블 가중치는 검증 {sp.val[0]} ~ {sp.val[1]}로 고정(selection sha256 `{sel['sha256'][:16]}…`)",
            f"> - 모델 리비전: {revs or '—'}. 결측 입력: {cfg['missing']}",
            "> - 비용: Jetson Orin Nano Super, 전력 모드 25W, **클럭 고정 안 함**(sudo 없음), 워밍업·반복은 표에 적음. 다시 만들기: `pretrained report`", ""]


def _cmp_md(t: pd.DataFrame, labels: list[str]) -> str:
    rows = []
    for r in t.to_dict("records"):
        row = {"공간": SPACE_NAME.get(r["space_id"], r["space_id"]), "지평(분)": r["horizon_min"]}
        for m, name in METHODS.items():
            row[name] = _f(r.get(m))
        for lb in labels:
            row[lb] = _f(r.get(lb))
        row["평가 시점 수"] = f"{r['n']:,}"
        rows.append(row)
    return _table(pd.DataFrame(rows))


def bench_table(root: Path = ROOT) -> pd.DataFrame:
    rows = []
    for p in sorted((root / "bench").glob("*.json")) if (root / "bench").exists() else []:
        b = json.loads(p.read_text())
        for bs, v in b["batches"].items():
            pw = v["power"]
            rows.append({"모델": b["model"], "장치": b["device"], "배치": bs, "평균(ms)": v["latency_ms"]["mean"],
                         "p95(ms)": v["latency_ms"]["p95"], "최대 RSS(MB)": b["peak_rss_mb"], "GPU 할당(MB)": b.get("gpu_alloc_mb", "—"),
                         "VDD_IN 평균(mW)": pw.get("vdd_in_mw_mean", "—"), "유휴 VDD_IN(mW)": b["idle"].get("vdd_in_mw_mean", "—"),
                         "GPU 사용률(%)": pw.get("gr3d_pct_mean", "—"), "반복/워밍업": f"{v['latency_ms']['repeats']}/{v['latency_ms']['warmup']}"})
    return pd.DataFrame(rows)


def write_reports(cfg, deliv: Path = DELIV, adv: Path = ADV, root: Path = ROOT) -> list[Path]:
    sel = require(root)
    frames = eval_frames(cfg, root)
    tdir, fdir, adir = deliv / "tables", deliv / "figures", adv / "tables"
    for d in (tdir, fdir, adir):
        d.mkdir(parents=True, exist_ok=True)
    cond = conditions(cfg, sel, root)
    main_labels = [k for k in frames if k in (sel["chosen"]["model"], "직접 학습(과제 3)")]
    written = []
    t_all, t_use = comparison_table(cfg, frames, "all"), comparison_table(cfg, frames, "usage")
    grid = pd.DataFrame(sel["grid"]).sort_values("val_norm_mae")
    md = ["# 과제 6 결과 표 ① 비교표·예측 구간·비용 (자동 생성)", "", *cond,
          f"선택(검증): **{sel['chosen']['model']}, 컨텍스트 {sel['chosen']['ctx_min']}분** — {sel['chosen']['reason']}", "",
          "## 시험 기간 CO₂ MAE(ppm) — 전체", "", _cmp_md(t_all, list(frames)), "",
          "## 시험 기간 CO₂ MAE(ppm) — 사용 시간대", "", _cmp_md(t_use, list(frames)), "",
          "다른 크기 열은 비용 비교용이며 선택에 쓰지 않았다.", ""]
    ch = frames.get(sel["chosen"]["model"])
    if ch is not None:
        cv = coverage_table(ch, "all")
        md += ["## 예측 구간(10~90%) 적중률 — 이상적 80%", "",
               _table(cv.assign(**{c: cv[c].map(lambda x: _f(x * 100 if c != "평균 폭" else x)) for c in ("적중률", "아래 이탈", "위 이탈", "평균 폭")})
                      .drop(columns=["space_id"])), ""]
        cal = calibration(ch)
        md += ["## 보정(명목 분위수 대비 실제 아래 비율)", "", _table(cal.round(3)), ""]
    bt = bench_table(root)
    md += ["## 비용(보드 실측)", "", _table(bt) if len(bt) else "측정 전(`pretrained bench`)", "",
           "## 검증 선택 격자(크기 × 컨텍스트)", "", _table(grid.round(4)), ""]
    if len(bt):
        meas = bt.groupby("모델")["최대 RSS(MB)"].max().to_dict()
        md += ["## 메모리 예산(가정 LLM 1B·3B·7B 4비트)", "", _table(budget_table(cfg, meas)), ""]
    p = tdir / "01-comparison.md"
    p.write_text("\n".join(md), encoding="utf-8")
    written.append(p)
    for name, keys, title in (("01-covariates.md", [k for k in frames if "공변량" in k] + main_labels[:1], "공변량 입력"),
                              ("02-ensemble.md", ["섞은 예측"] + main_labels, "직접 학습 + 사전학습 섞은 예측")):
        sub = {k: frames[k] for k in keys if k in frames}
        if len(sub) < 2:
            continue
        md = [f"# 과제 6 심화 결과 표 — {title} (자동 생성)", "", *cond,
              _cmp_md(comparison_table(cfg, sub, "all"), list(sub)), ""]
        if title.startswith("직접") and sel.get("ensemble"):
            e = sel["ensemble"]
            md += [f"검증으로 고른 방식: {e['mode']}, 하나의 w = {e['w_global']}(검증 MAE {e['val_mae_global']:.2f}), "
                   f"공간 × 지평별 검증 MAE {e['val_mae_per_cell']:.2f}", ""]
        p = adir / name
        p.write_text("\n".join(md), encoding="utf-8")
        written.append(p)
    return written

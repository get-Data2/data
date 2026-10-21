"""결과 표·그래프(anomaly report). 해석이 들어가는 산출물 본문은 사람이 쓰고, 여기서는 숫자 원본을 만든다.

- docs/deliverables/task4/tables/  성적표·판정·미탐 목록·임계값 곡선·운영 제안 표
- docs/deliverables/task4/figures/ 임계값 곡선, 종류별 지연
- docs/advanced/task4/tables/      예측 오차 탐지기, 영점 틀어짐 전용
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.baseline import SPACE_NAME
from analysis.pipeline import _table

from .config import KIND_NAME

DELIV = Path("docs/deliverables/task4")
ADV = Path("docs/advanced/task4")
GROUP_NAME = {"model": "모델(오토인코더)", "rules": "규칙(급변·값 고정)", "combined": "결합(모델+규칙)",
              "forecast_err": "예측 오차(심화 1)", "drift": "영점 전용(심화 2)", "combined_plus": "결합+영점 전용"}


def _pct(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.0f}%"


def _min(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.0f}"


def conditions(work, res) -> list[str]:
    sp = work.cfg.split
    inj = json.loads((work.root / "inject.json").read_text())
    tr = json.loads((work.root / "models" / "train.json").read_text()) if (work.root / "models" / "train.json").exists() else {}
    dev = next(iter(tr.values()), {}).get("device", "—")
    return ["> **측정 조건**",
            f"> - **실데이터 + 주입 고장**: 시험 기간 {sp.test[0]} ~ {sp.test[1]}({res['test_days']}일) 센서 5분 시계열 사본에 고장 5종 × "
            f"{work.cfg['inject']['per_kind']}건 주입(씨앗 {inj['seed']}, 정답 표 sha256 `{inj['faults_sha256'][:16]}…`)",
            f"> - 학습 {sp.train[0]} ~ {sp.train[1]}(정상만, 사건 후보 제외), 임계값은 검증 {sp.val[0]} ~ {sp.val[1]}(주입 없음) "
            f"하루당 경보 ≤ {work.cfg['alarms']['target_per_day']}로 고정(thresholds sha256 `{res['thresholds_sha256'][:16]}…`)",
            f"> - 경보: 30분 안 병합, 매칭 여유 {work.cfg['alarms']['match_slack_min']}분(영점 틀어짐 {work.cfg['alarms']['F3_match_h']}시간). "
            f"장치 {dev}, Jetson Orin Nano Super 25W. 다시 만들기: `anomaly report`", ""]


def scorecard_table(res) -> pd.DataFrame:
    rows = []
    for g, v in res["groups"].items():
        card = {r["kind"]: r for r in v["scorecard"]}
        row = {"탐지기": GROUP_NAME.get(g, g)}
        for k in ("F1", "F2", "F3", "F4", "F5"):
            r = card[k]
            row[f"{KIND_NAME[k]}"] = f"{r['detected']}/{r['n']}"
        t = card["전체"]
        row |= {"전체 탐지율": _pct(t["rate"]), "지연 중앙(분)": _min(t["delay_median_min"]), "원인 정확도": _pct(t["cause_acc"]),
                "헛경보/일": f"{v['false']['per_day']:.2f}", "실제 이상 후보 제외": f"{v['false']['per_day_excl_real']:.2f}"}
        rows.append(row)
    return pd.DataFrame(rows)


def delay_table(res) -> pd.DataFrame:
    rows = []
    for g, v in res["groups"].items():
        for r in v["scorecard"]:
            if r["kind"] == "전체":
                continue
            rows.append({"탐지기": GROUP_NAME.get(g, g), "고장": KIND_NAME[r["kind"]], "잡음": f"{r['detected']}/{r['n']}",
                         "지연 중앙(분)": _min(r["delay_median_min"]), "지연 최대(분)": _min(r["delay_max_min"]),
                         "원인 정확도": _pct(r["cause_acc"])})
    return pd.DataFrame(rows)


def missed(work, group) -> pd.DataFrame:
    faults = pd.read_parquet(work.root / "faults.parquet")
    al = pd.read_parquet(work.root / "alarms" / f"{group}.parquet")
    hit = set(al["fault_id"])
    m = faults[~faults["fault_id"].isin(hit)].copy()
    m["시작(KST)"] = m["start"].dt.tz_convert("Asia/Seoul").dt.strftime("%m-%d %H:%M")
    m["길이(시간)"] = ((m["end"] - m["start"]).dt.total_seconds() / 3600).round(1)
    m["고장"] = m["kind"].map(KIND_NAME)
    m["공간"] = m["space_id"].map(SPACE_NAME)
    return m[["fault_id", "고장", "공간", "device_id", "items", "시작(KST)", "길이(시간)", "magnitude"]]


def plot_curve(res, out: Path) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Noto Sans CJK KR"
    c = pd.DataFrame(res.get("curve", []))
    if c.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, g in c.groupby("detector"):
        g = g.sort_values("test_false_per_day")
        ax.plot(g["test_false_per_day"], g["test_rate"] * 100, marker=".", label=name)
        ch = g[g["chosen"]]
        if len(ch):
            ax.scatter(ch["test_false_per_day"], ch["test_rate"] * 100, s=80, facecolors="none", edgecolors="red")
    ax.axvline(1.0, color="gray", ls="--", lw=0.8)
    ax.axhline(80, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("시험 기간 헛경보(하루당, 전체 공간)")
    ax.set_ylabel("이벤트 탐지율(%)")
    ax.set_title("임계값을 바꿀 때 탐지율과 헛경보(빨간 원 = 검증으로 고른 값)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "threshold_curve.png", dpi=110)
    plt.close(fig)
    return "threshold_curve.png"


def operations_table(work, res) -> pd.DataFrame:
    """공간별 결합 경보 수(시험)와 임계값, 경보 원인 종류별 알림 대상."""
    al = pd.read_parquet(work.root / "alarms" / "combined.parquet")
    thr = json.loads((work.root / "thresholds.json").read_text())
    ops = work.cfg["operations"]
    rows = []
    for space in work.cfg.spaces:
        a = al[al["space_id"] == space]
        rows.append({"공간": SPACE_NAME.get(space, space), "센서 수(열)": len(work.cols[space]),
                     "모델 임계 분위수": f"{thr['detectors'].get('ae', {}).get('threshold_q', float('nan')):.5f}",
                     "시험 경보/일": f"{len(a) / res['test_days']:.2f}",
                     "그중 헛경보/일": f"{(a['fault_id'] == 0).sum() / res['test_days']:.2f}",
                     "센서 고장 알림": ops["sensor_fault"], "설비 고장 알림": ops["equipment_fault"], "공기질 알림": ops["air_quality"]})
    return pd.DataFrame(rows)


def write_reports(work, deliv: Path = DELIV, adv: Path = ADV) -> list[Path]:
    res = json.loads((work.root / "results.json").read_text())
    tdir, fdir, adir = deliv / "tables", deliv / "figures", adv / "tables"
    for d in (tdir, fdir, adir):
        d.mkdir(parents=True, exist_ok=True)
    cond = conditions(work, res)
    written = []
    crit = pd.DataFrame(res.get("criteria", []))
    md = ["# 과제 4 결과 표 ① 탐지 성적표 (자동 생성)", "", *cond, "## 확인 기준 판정", "",
          _table(crit) if len(crit) else "결합 결과 없음", "", "## 성적표(탐지기 × 고장 종류)", "", _table(scorecard_table(res)), "",
          "## 종류별 지연·원인 정확도", "", _table(delay_table(res)), ""]
    for g in ("model", "rules", "combined"):
        if (work.root / "alarms" / f"{g}.parquet").exists():
            m = missed(work, g)
            md += [f"## 놓친 고장 — {GROUP_NAME[g]} ({len(m)}건)", "", _table(m) if len(m) else "없음", ""]
    p = tdir / "01-scorecard.md"
    p.write_text("\n".join(md), encoding="utf-8")
    written.append(p)

    fig = plot_curve(res, fdir)
    c = pd.DataFrame(res.get("curve", []))
    show = c.assign(threshold_q=c["threshold_q"].round(5), test_rate=(c["test_rate"] * 100).round(1),
                    val_per_day=c["val_per_day"].round(2), test_false_per_day=c["test_false_per_day"].round(2),
                    chosen=c["chosen"].map({True: "◀", False: ""})) if len(c) else c
    md = ["# 과제 4 결과 표 ② 임계값–탐지율·헛경보 (자동 생성)", "", *cond,
          "선택은 검증 기간(주입 없음) 하루당 경보로만 했다. 아래 시험 값은 선택 뒤 보고용이다.", "",
          (f"![임계값 곡선](../figures/{fig})" if fig else ""), "",
          _table(show.rename(columns={"detector": "탐지기", "threshold_q": "임계 분위수", "val_per_day": "검증 경보/일",
                                      "test_rate": "시험 탐지율(%)", "test_false_per_day": "시험 헛경보/일",
                                      "delay_median": "지연 중앙(분)", "chosen": "선택"})) if len(c) else "없음", ""]
    p = tdir / "02-threshold-curve.md"
    p.write_text("\n".join(md), encoding="utf-8")
    written.append(p)

    md = ["# 과제 4 결과 표 ③ 운영 제안 근거 (자동 생성)", "", *cond, _table(operations_table(work, res)), ""]
    p = tdir / "03-operations.md"
    p.write_text("\n".join(md), encoding="utf-8")
    written.append(p)

    for name, groups, title in (("01-forecast-error.md", ["model", "forecast_err", "combined"], "예측 오차 탐지기 vs 오토인코더"),
                                ("02-drift.md", ["model", "rules", "drift", "combined", "combined_plus"], "영점 틀어짐 전용 탐지기")):
        sub = {"groups": {g: res["groups"][g] for g in groups if g in res["groups"]}}
        if len(sub["groups"]) <= 1:
            continue
        md = [f"# 과제 4 심화 결과 표 — {title} (자동 생성)", "", *cond, _table(scorecard_table(sub)), "",
              _table(delay_table(sub)), ""]
        p = adir / name
        p.write_text("\n".join(md), encoding="utf-8")
        written.append(p)
    return written

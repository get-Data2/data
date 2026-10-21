"""결과 표·그래프(state report). 해석이 들어가는 산출물 본문은 사람이 쓰고, 여기서는 숫자 원본을 만든다.

- docs/deliverables/task5/tables/  성적표·혼동 행렬·판정, 정답 비교, 오류 사례
- docs/deliverables/task5/figures/ 혼동 행렬, PR 곡선
- docs/advanced/task5/tables/      처음 보는 공간, 노쇼·예약 없는 사용
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pipeline import _table

DELIV = Path("docs/deliverables/task5")
ADV = Path("docs/advanced/task5")


def _cm_md(m: dict) -> str:
    if not m:
        return "(자료 없음)"
    return ("| 정답 \\ 예측 | 양성(1) | 음성(0) |\n|---|---|---|\n"
            f"| 양성(1) | {m['tp']:,} | {m['fn']:,} |\n| 음성(0) | {m['fp']:,} | {m['tn']:,} |")


def _mrow(name, m):
    if not m:
        return {"방법": name, "칸 수": 0}
    f = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
    return {"방법": name, "칸 수": f"{m['n']:,}", "정확도": f(m["accuracy"]), "재현율": f(m["recall"]), "정밀도": f(m["precision"]),
            "F1": f(m["f1"]), "macro-F1": f(m["macro_f1"]), "비어있음→사용중(헛 사용중)": f(m["false_occupied_rate"]),
            "항상 음성 정확도": f(m["always_negative_accuracy"])}


def _plots(root: Path, figs: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Noto Sans CJK KR"
    figs.mkdir(parents=True, exist_ok=True)
    te = pd.read_parquet(root / "eval" / "pred_vent.parquet")
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(figsize=(5, 4))
    if te["y"].nunique() == 2:
        pr, rc, _ = precision_recall_curve(te["y"], te["p"])
        ax.plot(rc, pr, label="분류기")
    r = te["rule"]
    tp = ((r == 1) & (te["y"] == 1)).sum()
    ax.scatter([tp / max(te["y"].sum(), 1)], [tp / max(r.sum(), 1)], color="red", label="규칙(CO2 ≥ 850)")
    ax.axvline(0.85, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("재현율")
    ax.set_ylabel("정밀도")
    ax.set_title("환기 선제 경보 PR 곡선(시험 기간)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "vent_pr.png", dpi=110)
    plt.close(fig)
    return ["vent_pr.png"]


def write_reports(cfg, root: Path = Path("var/state"), deliv: Path = DELIV, adv: Path = ADV) -> list[Path]:
    res = json.loads((root / "eval" / "results.json").read_text())
    sel = json.loads((root / "selection.json").read_text())
    ana = json.loads((root / "eval" / "analysis.json").read_text()) if (root / "eval" / "analysis.json").exists() else {}
    tdir, adir = deliv / "tables", adv / "tables"
    for d in (tdir, adir):
        d.mkdir(parents=True, exist_ok=True)
    sp = cfg.split
    cond = ["> **측정 조건**",
            f"> - 실데이터(과제 2 공간 5분 시계열). 분할 `{cfg['split']}`: 학습 {sp.train[0]} ~ {sp.train[1]}, 검증 {sp.val[0]} ~ {sp.val[1]}, "
            f"시험 {sp.test[0]} ~ {sp.test[1]}(KST). 관찰 기준은 실시간 수집 기간(2026-10-20~)의 직접 관찰",
            f"> - 선택 고정 sha256 `{sel['sha256'][:16]}…`({sel['frozen_at']}), 씨앗 {len(cfg['models']['seeds'])}개 평균 확률",
            f"> - 관찰 진행: {res['observation']['status']}", ""]
    figs = _plots(root, deliv / "figures")
    out = []
    v = res["vent"]
    md = ["# 과제 5 결과 표 ① 성적표·혼동 행렬 (자동 생성)", "", *cond, "## 확인 기준 판정", "", _table(pd.DataFrame(res["criteria"])), "",
          "## 환기 선제 경보(시험 기간, 양성 = 지금 < 1,000ppm이고 30분 안에 ≥ 1,000)", "",
          f"선택: {sel['vent']['candidate']}, 임계값 {sel['vent']['threshold']:.4f}(검증 재현율 ≥ 85% 중 정밀도 최대"
          f"{'' if sel['vent'].get('recall_ok', True) else ' — 검증에서도 85% 미달'})", "",
          _table(pd.DataFrame([_mrow("분류기", v["classifier"]["all"]), _mrow("규칙 CO₂ ≥ 850", v["rule_850"]["all"])])), "",
          f"하루당 헛경보(공간 합): 분류기 {v['classifier']['all']['false_alarms_per_day']:.1f}, 규칙 {v['rule_850']['all']['false_alarms_per_day']:.1f}. "
          f"선행 시간 중앙 {v['lead_min']['median']}분", "", "분류기 혼동 행렬:", "", _cm_md(v["classifier"]["all"]), "",
          f"과제 3 예측 기반 판정: {json.dumps(v.get('forecast_model', {}), ensure_ascii=False, default=str)[:400]}", "",
          *[f"![{f}](../figures/{f})" for f in figs], "",
          "## 사용 여부", "", f"선택: {sel['occupancy']['candidate']}, 임계값 {sel['occupancy']['threshold']:.4f}(검증 macro-F1 최대)", ""]
    for basis, title in (("observation", "직접 관찰 기준(확인 기준)"), ("schedule_test", "일정 기반 정답(시험 기간, 실습실·사무실)")):
        b = res["occupancy"][basis]
        rows = [_mrow({"classifier": "분류기", "rule_15min_30ppm": "규칙 15분 +30ppm", "leak_with_schedule": "누수 실험(일정·시각 입력)"}[k], x["all"])
                for k, x in b.items()]
        md += [f"### {title}", "", _table(pd.DataFrame(rows)), "", "분류기 혼동 행렬:", "", _cm_md(b["classifier"]["all"]), ""]
    p = tdir / "01-scorecards.md"
    p.write_text("\n".join(md), encoding="utf-8")
    out.append(p)
    md = ["# 과제 5 결과 표 ② 오류 분석 (자동 생성)", "", *cond]
    for k, title in (("transition_schedule", "전환 뒤 경과별 오류율(일정 정답)"), ("transition_observation", "전환 뒤 경과별 오류율(관찰)"),
                     ("people_recall", "인원별 재현율(관찰)"), ("top_confusions_schedule", "가장 많이 헷갈리는 경우(일정)"),
                     ("top_confusions_observation", "가장 많이 헷갈리는 경우(관찰)"), ("vent_false_alarms", "환기 헛경보 시각대")):
        rows = ana.get(k) or []
        md += [f"## {title}", "", _table(pd.DataFrame(rows)) if rows else "(자료 없음)", ""]
    p = tdir / "02-errors.md"
    p.write_text("\n".join(md), encoding="utf-8")
    out.append(p)
    md = ["# 과제 5 심화 결과 표 (자동 생성)", "", *cond, "## 처음 보는 공간", "",
          _table(pd.DataFrame(ana.get("unseen_space") or [])) if ana.get("unseen_space") else "(자료 없음)", "",
          "## 노쇼·예약 없는 사용", "", f"예약 출처: {ana.get('noshow_unbooked', {}).get('source')}", "",
          f"건수: {ana.get('noshow_unbooked', {}).get('counts')}", "",
          f"관찰 대조: {ana.get('noshow_unbooked', {}).get('observation_check', '관찰 자료 없음')}", ""]
    p = adir / "01-advanced.md"
    p.write_text("\n".join(md), encoding="utf-8")
    out.append(p)
    return out

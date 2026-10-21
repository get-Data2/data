"""결과 표·그래프 생성(forecast report). 해석이 들어가는 산출물 본문은 사람이 쓰고, 여기서는 숫자 원본을 만든다.

- docs/deliverables/task3/tables/  비교표·판정표·공간별 분석·일정 가치
- docs/deliverables/task3/figures/ 지평 곡선·시각별 오차·대표 하루
- docs/advanced/task3/tables/      과거 길이 × 모델, 미래 예약, 처음 보는 공간
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.baseline import SPACE_NAME
from analysis.pipeline import _table

from . import analysis as A
from .evaluate import comparison_table, criteria, load_test_preds
from .experiments import load_stages, require_selection

DELIV = Path("docs/deliverables/task3")
ADV = Path("docs/advanced/task3")


def _f(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{nd}f}"


def conditions(ctx, sel) -> list[str]:
    sp = ctx.cfg.split
    run = json.loads((ctx.runs / next(iter(sel["runs"])) / "run.json").read_text())
    return ["> **측정 조건**",
            f"> - 실데이터(과제 2 공간 5분 시계열). 분할 `{ctx.cfg['split']}`: 학습 {sp.train[0]} ~ {sp.train[1]}, 검증 {sp.val[0]} ~ {sp.val[1]}, "
            f"**시험 {sp.test[0]} ~ {sp.test[1]}**(KST)",
            f"> - 평가 시점 파일 sha256 `{ctx.inputs['eval_sha256'][:16]}…`(과제 2와 같은 행·같은 정답), 선택 고정 sha256 `{sel['sha256'][:16]}…`({sel['frozen_at']})",
            f"> - 장치 {run.get('gpu', run.get('device'))}(Jetson Orin Nano Super, 전력 모드 25W), torch {run.get('torch')}, TF32 끔·결정적 설정, "
            f"씨앗 {len([r for r in sel['runs'].values() if r['group'] == 'final'])}개(평균 ± 표준편차)",
            "> - 다시 만들기: `forecast report`(표·그래프), 해석은 사람이 쓴다", ""]


def _comparison_md(t: pd.DataFrame) -> str:
    rows = []
    for r in t.itertuples():
        unit = 2 if r.item == "temperature" else 1
        rows.append({"공간": SPACE_NAME.get(r.space_id, r.space_id), "항목": "CO₂" if r.item == "co2" else "온도",
                     "지평(분)": r.horizon_min, "현재값 유지": _f(r.persistence, unit), "최근 1시간 평균": _f(r.mean_1h, unit),
                     "어제 같은 시각": _f(r.yesterday, unit),
                     "학습 모델": "평가 불가" if np.isnan(r.mae) else f"**{_f(r.mae, unit)}** ± {_f(0.0 if np.isnan(r.sd) else r.sd, unit)}",
                     "개선율(%)": "—" if np.isnan(r.mae) else f"{(1 - r.mae / r.persistence) * 100:.1f}", "평가 시점 수": f"{r.n:,}"})
    return _table(pd.DataFrame(rows))


def stage_table(ctx, stage: str) -> pd.DataFrame:
    st = load_stages(ctx)[stage]
    rows = []
    for c in st["candidates"]:
        recs = [json.loads(p.read_text()) for p in ctx.runs.glob(f"{c['hash']}-s*/run.json")]
        recs = [r for r in recs if r["status"] == "done"]
        lab = [r["val"]["cells"].get("lab/co2/60", {}).get("mae") for r in recs]
        rows.append({"후보": c["label"], "검증 정규화 MAE": f"{c['mean']:.4f} ± {c['sd']:.4f}",
                     "검증 실습실 CO₂ 60분(ppm)": _f(np.mean([x for x in lab if x is not None])) if recs else "—",
                     "파라미터": f"{recs[0]['params']:,}" if recs else "—",
                     "학습 시간(초)": _f(np.mean([r["train_seconds"] for r in recs]), 0) if recs else "—",
                     "에폭(최고)": f"{np.mean([r['epochs'] for r in recs]):.0f}({np.mean([r['best_epoch'] for r in recs]):.0f})" if recs else "—",
                     "선택": "◀" if c["hash"] == st.get("picked_hash") else ""})
    return pd.DataFrame(rows)


def write_reports(ctx, deliv: Path = DELIV, adv: Path = ADV) -> list[Path]:
    sel = require_selection(ctx)
    bl = pd.read_parquet(ctx.cfg["inputs"]["baseline"])
    acfg = ctx.cfg.acfg
    figs = deliv / "figures"
    tdir = deliv / "tables"
    adir = adv / "tables"
    for d in (figs, tdir, adir):
        d.mkdir(parents=True, exist_ok=True)
    cond = conditions(ctx, sel)
    final = load_test_preds(ctx, sel, "final")[sel["final_label"]]
    written = []

    # 1) 모델 비교표·판정표
    t_all, t_use = comparison_table(ctx, final, "all"), comparison_table(ctx, final, "usage")
    fig = A.plot_horizon_curves(t_all, figs)
    md = ["# 과제 3 결과 표 ① 모델 비교표·확인 기준 판정 (자동 생성)", "", *cond,
          f"최종 모델: **{sel['final_label']}** (검증 기준 선택, `selection.json`)", "",
          "## 확인 기준 판정", "", _table(criteria(ctx, final)), "",
          "## 시험 기간 MAE — 전체 시간", "", _comparison_md(t_all), "",
          "## 시험 기간 MAE — 사용 시간대(평일·공휴일 제외·08~19시)", "", _comparison_md(t_use), "",
          f"![지평별 MAE](../figures/{fig})", "",
          "## 설계 선택 경로(검증 기준)", "",
          _table(pd.DataFrame([{"단계": s, "선택": v["picked_label"], "이유": v["reason"]} for s, v in sel["stages"].items()])), ""]
    for s in ("D1", "D2", "D3", "D4"):
        md += [f"### {s}", "", _table(stage_table(ctx, s)), ""]
    p = tdir / "01-comparison.md"
    p.write_text("\n".join(md), encoding="utf-8")
    written.append(p)

    # 2) 공간별 오차 분석
    pe = A.phase_errors(final, bl, acfg)
    te = A.top_errors(final, bl, acfg)
    segs = A.window_open_segments(ctx.arrays, ctx.cfg["window_open"], acfg)
    a, b = ctx.cfg.split.utc("test")
    segs_t = segs[(segs["t"] >= pd.Timestamp(a)) & (segs["t"] < pd.Timestamp(b))] if len(segs) else segs
    wo = A.subset_mae(final, bl, A.rows_near(final, segs_t, 0, 30)) if len(segs_t) else pd.DataFrame()
    ev = A.event_spans(ctx.cfg["inputs"]["events"], ctx.cfg.split)
    evm = A.subset_mae(final, bl, A.rows_near(final, ev.rename(columns={"start": "t"}), 0, 120)) if len(ev) else pd.DataFrame()
    f2 = A.plot_hourly_error(final, bl, figs)
    md = ["# 과제 3 결과 표 ② 공간별 오차 분석 (자동 생성)", "", *cond,
          "## 일과 단계별 CO₂ 60분 MAE(정답 시각 기준, 씨앗 평균 예측)", "",
          _table(pe.assign(space_id=pe["space_id"].map(SPACE_NAME), model=pe["model"].round(1),
                           persistence=pe["persistence"].round(1), **{"개선율(%)": pe["개선율(%)"].round(1)})
                 .rename(columns={"space_id": "공간", "phase": "단계", "model": "학습 모델", "persistence": "현재값 유지"})), "",
          f"![시각별 오차](../figures/{f2})", "",
          "## 오차 상위 20개(CO₂ 60분)", "",
          _table(te.assign(공간=te["space_id"].map(SPACE_NAME), 정답=te["y_true"].round(0), 예측=te["y_pred"].round(0),
                           현재값=te["persistence"].round(0), 오차=te["err"].round(0))[["공간", "정답 시각(KST)", "단계", "정답", "예측", "현재값", "오차"]]), "",
          f"## 창문 개방 추정 구간(사용 시간대 {ctx.cfg['window_open']['within_min']}분 안 −{ctx.cfg['window_open']['drop_ppm']}ppm, 일과 경계 제외) — 시험 기간 {len(segs_t)}회", "",
          (_table(wo.round(1)) if len(wo) else "없음"), "",
          f"## 시험 기간 밀집 사건(B3) 구간 — {len(ev)}건, 사건 시작 뒤 2시간", "",
          (_table(evm.round(1)) if len(evm) else "없음"), ""]
    for sp, day in _example_days(final):
        fn = A.plot_day(final, bl, sp, day, figs, f"day_{sp}_{day}.png")
        md += [f"![{SPACE_NAME[sp]} {day}](../figures/{fn})", ""]
    p = tdir / "02-space-analysis.md"
    p.write_text("\n".join(md), encoding="utf-8")
    written.append(p)

    # 3) 일정 정보의 가치
    wo_s = load_test_preds(ctx, sel, "S")
    if wo_s:
        sv = A.schedule_value(final, next(iter(wo_s.values())), bl, acfg)
        show = sv[sv["horizon_min"].isin([15, 30, 60])].copy()
        show["공간"] = show["space_id"].map(SPACE_NAME)
        show = show.assign(**{"일정 포함": show["with_"].round(2), "일정 제외": show["without"].round(2),
                              "감소(제외−포함)": show["reduce"].round(2), "씨앗 변동(sd)": show["reduce_sd"].round(2)})
        lines = [A.conclusion(sv, s, it, 60, part) for s in ("lab", "office") for it in ("co2", "temperature")
                 for part in ("전체", "사용 시간대")]
        md = ["# 과제 3 결과 표 ③ 일정 정보 포함/제외 (자동 생성)", "", *cond,
              f"비교: 최종 모델 vs 같은 설정에서 `schedule`만 뺀 모델({next(iter(wo_s))}). 같은 씨앗끼리 차이를 낸다.", "",
              "## 결론 문장(자동)", "", *[f"- {x}" for x in lines], "",
              "## 공간 × 항목 × 지평 × 구간", "",
              _table(show[["구간", "공간", "item", "horizon_min", "일정 포함", "일정 제외", "감소(제외−포함)", "씨앗 변동(sd)", "n"]]), ""]
        p = tdir / "03-schedule-value.md"
        p.write_text("\n".join(md), encoding="utf-8")
        written.append(p)

    # 심화 1: 과거 길이 × 모델
    md = ["# 과제 3 심화 결과 표 ① 과거 길이 × 모델 (자동 생성)", "", *cond, "## 검증 기간(선택 기준)", "",
          _table(stage_table(ctx, "B1")), "",
          f"선택: **{sel['stages']['B1']['picked_label']}** — {sel['stages']['B1']['reason']}. 이 조합의 시험 결과는 결과 표 ①이다.", ""]
    p = adir / "01-lookback-model.md"
    p.write_text("\n".join(md), encoding="utf-8")
    written.append(p)

    # 심화 2: 미래 예약
    a2 = load_test_preds(ctx, sel, "A2")
    if a2:
        lab2, d2 = next(iter(a2.items()))
        rows = []
        for sp in ctx.cfg["spaces"]:
            for it in ("co2", "temperature"):
                for h in (30, 60):
                    for subset in ("all", "usage"):
                        ta = comparison_table(ctx, final, subset)
                        tb = comparison_table(ctx, d2, subset)
                        ra = ta[(ta.space_id == sp) & (ta["item"] == it) & (ta.horizon_min == h)].iloc[0]
                        rb = tb[(tb.space_id == sp) & (tb["item"] == it) & (tb.horizon_min == h)].iloc[0]
                        rows.append({"공간": SPACE_NAME[sp], "항목": it, "지평": h, "구간": "전체" if subset == "all" else "사용 시간대",
                                     "현재값 유지": _f(ra.persistence), "예약 없음(최종)": _f(ra.mae), "예약 있음": _f(rb.mae),
                                     "변화(%)": "—" if np.isnan(ra.mae) else f"{(rb.mae / ra.mae - 1) * 100:+.1f}"})
        bk = ctx.arrays.bookings
        md = ["# 과제 3 심화 결과 표 ② 미래 예약 입력 (자동 생성)", "", *cond,
              f"> ⚠️ 회의실 예약은 **추정 예약**(실제 CO₂ 급상승에서 거꾸로 만든 회의 구간, {len(bk)}건)이다. 시험 기간에는 미래 CO₂를 본 셈이므로 "
              "**\"예약을 완벽히 알았다면\"의 상한**으로만 해석한다.", "",
              f"비교: 최종 모델 vs `booking` 추가({lab2})", "", _table(pd.DataFrame(rows)), ""]
        p = adir / "02-future-info.md"
        p.write_text("\n".join(md), encoding="utf-8")
        written.append(p)

    # 심화 3: 처음 보는 공간
    a3 = load_test_preds(ctx, sel, "A3")
    if a3:
        ref_label = next(k for k in a3 if "학습=" not in k)
        ref = a3[ref_label]
        rows = []
        for lab3, d3 in a3.items():
            if "학습=" not in lab3:
                continue
            held = [s for s in ctx.cfg["spaces"] if s not in lab3.split("학습=")[1].split(",")][0]
            for it in ("co2", "temperature"):
                for h in (30, 60):
                    ta = comparison_table(ctx, ref, "all")
                    tb = comparison_table(ctx, d3, "all")
                    tf = comparison_table(ctx, final, "all")
                    q = lambda t: t[(t.space_id == held) & (t["item"] == it) & (t.horizon_min == h)].iloc[0]
                    rows.append({"뺀 공간": SPACE_NAME[held], "항목": it, "지평": h, "현재값 유지": _f(q(ta).persistence),
                                 "포함 학습(space 끔)": _f(q(ta).mae), "빼고 학습(처음 봄)": _f(q(tb).mae),
                                 "최종 모델(space 켬)": _f(q(tf).mae),
                                 "처음 봄 vs 포함(%)": "—" if np.isnan(q(ta).mae) else f"{(q(tb).mae / q(ta).mae - 1) * 100:+.1f}"})
        md = ["# 과제 3 심화 결과 표 ③ 처음 보는 공간 (자동 생성)", "", *cond,
              f"비교 기준: `space` 묶음을 끄고 합친 정규화 통계를 쓰는 같은 설정({ref_label}). 뺀 공간은 학습·검증·정규화 어디에도 쓰지 않았다.", "",
              _table(pd.DataFrame(rows)), ""]
        p = adir / "03-unseen-space.md"
        p.write_text("\n".join(md), encoding="utf-8")
        written.append(p)
    return written


def _example_days(final) -> list[tuple[str, str]]:
    """대표 하루: 실습실은 시험 기간 수업일 중 CO₂ 최고가 중앙값인 날, 회의실은 CO₂ 최고가 가장 높은 날."""
    d = final[(final["item"] == "co2") & (final["horizon_min"] == 60)].copy()
    d["day"] = (d["t0"] + pd.Timedelta(minutes=60)).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    out = []
    lab = d[(d["space_id"] == "lab") & d["usage_hours"]].groupby("day")["y_true"].max()
    if len(lab):
        out.append(("lab", (lab - lab.median()).abs().idxmin()))
    mt = d[d["space_id"] == "meeting"].groupby("day")["y_true"].max()
    if len(mt):
        out.append(("meeting", mt.idxmax()))
    return out

"""헛경보 원인 대응: 최근 N일 점수로 매일 기준을 다시 잡는 적응 임계값 (2026-10-21, 시험 헛경보 4.65/일 분석 뒤 추가).

고정 임계값은 검증 기간(4~5월) 점수 분포에 묶여 있어, 계절이 바뀐 시험 기간(5~7월)에 점수 수준 자체가 2~4배로 오르면
정상 칸도 경보가 된다. 여기서는 하루마다 **그 날 이전 ref_days일**의 점수(같은 공간)를 기준 분포로 써서 분위수를 다시 낸다.
- 인과적: 그 날 이후 점수는 쓰지 않는다. 운영에서 그대로 쓸 수 있다.
- 분위수 기준값(threshold_q)은 검증 기간에서만 고른다(목표 하루 경보 이하인 최저 값). 시험 정답은 쓰지 않는다.
- 검증 기간 첫 min_ref_days일은 기준 분포가 짧아 선택 계산에서 뺀다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .alarms import alarms_from, candidates, choose_threshold, combine
from .evaluate import false_alarms, match, scorecard


def rolling_points(scores: pd.DataFrame, ref_days: int = 14, min_ref_days: int = 7, tz: str = "Asia/Seoul") -> pd.DataFrame:
    """scores: space_id, t, score, cause(시간순으로 이어 붙인 여러 기간). 하루마다 직전 ref_days일(관측된 칸 기준) 분포 대비 분위수 q."""
    out = []
    per_day = 288
    for space, g in scores.sort_values("t").groupby("space_id"):
        g = g.reset_index(drop=True)
        day = g["t"].dt.tz_convert(tz).dt.floor("D")
        vals = g["score"].to_numpy()
        for d, idx in g.groupby(day).groups.items():
            i0 = idx.min()
            ref = vals[max(0, i0 - ref_days * per_day):i0]
            ref = np.sort(ref[~np.isnan(ref)])
            if len(ref) < min_ref_days * per_day * 0.5:
                continue
            q = np.searchsorted(ref, vals[idx], side="right") / len(ref)
            out.append(g.loc[idx].assign(q=q))
    return pd.concat(out, ignore_index=True) if out else scores.assign(q=np.nan).iloc[0:0]


def median_scaled(scores: pd.DataFrame, ref_days: int = 14, min_ref_days: int = 7, tz: str = "Asia/Seoul") -> pd.DataFrame:
    """점수를 그 날 이전 ref_days일 점수의 중앙값으로 나눈다(계절에 따른 수준 이동 제거). 중앙값이라 지난 고장 몇 건에 끌리지 않는다."""
    out = []
    per_day = 288
    for space, g in scores.sort_values("t").groupby("space_id"):
        g = g.reset_index(drop=True)
        day = g["t"].dt.tz_convert(tz).dt.floor("D")
        vals = g["score"].to_numpy()
        for d, idx in g.groupby(day).groups.items():
            i0 = idx.min()
            ref = vals[max(0, i0 - ref_days * per_day):i0]
            ref = ref[~np.isnan(ref)]
            if len(ref) < min_ref_days * per_day * 0.5:
                continue
            out.append(g.loc[idx].assign(score=vals[idx] / max(float(np.median(ref)), 1e-9)))
    return pd.concat(out, ignore_index=True) if out else scores.iloc[0:0]


def run_scaled(work, det: str = "ae", ref_days: int = 14, log=print) -> dict:
    """중앙값 나눔 점수 → 검증 기간(나눔 점수) 분포 대비 분위수 → 검증으로 고른 임계값."""
    from .alarms import to_quantile
    root = work.root
    merge = work.cfg["alarms"]["merge_min"]
    val = pd.read_parquet(root / "scores" / f"{det}__val.parquet").assign(period="val")
    test = pd.read_parquet(root / "scores" / f"{det}__test.parquet").assign(period="test")
    sc = median_scaled(pd.concat([val, test], ignore_index=True), ref_days)
    sv, st = sc[sc["period"] == "val"], sc[sc["period"] == "test"]
    parts_v, parts_t = [], []
    for space in sc["space_id"].unique():
        ref = sv[sv["space_id"] == space]["score"]
        for src, dst in ((sv, parts_v), (st, parts_t)):
            g = src[src["space_id"] == space]
            dst.append(g.assign(q=to_quantile(g["score"], ref).to_numpy()))
    return _evaluate(work, pd.concat(parts_v), pd.concat(parts_t), f"중앙값 나눔 {det}, 기준 {ref_days}일", ref_days, log)


def run(work, det: str = "ae", ref_days: int = 14, log=print) -> dict:
    root = work.root
    val = pd.read_parquet(root / "scores" / f"{det}__val.parquet").assign(period="val")
    test = pd.read_parquet(root / "scores" / f"{det}__test.parquet").assign(period="test")
    pts = rolling_points(pd.concat([val, test], ignore_index=True), ref_days)
    return _evaluate(work, pts[pts["period"] == "val"], pts[pts["period"] == "test"], f"적응 분위수 {det}, 기준 {ref_days}일",
                     ref_days, log)


def _evaluate(work, vp: pd.DataFrame, tp: pd.DataFrame, label: str, ref_days: int, log=print) -> dict:
    root = work.root
    al = work.cfg["alarms"]
    merge = al["merge_min"]
    vdays = vp["t"].dt.tz_convert("Asia/Seoul").dt.floor("D").nunique()
    pick, rows, why = choose_threshold(vp, candidates(work.cfg), al["target_per_day"], vdays, merge)
    log(f"[{label}] 검증 {vdays}일로 고른 분위수 {pick['threshold_q']:.5f} — {why}")
    faults = pd.read_parquet(root / "faults.parquet")
    ev = pd.read_parquet(work.cfg["inputs"]["events"])
    ev = ev[ev["kind"].isin(["B0", "B3"])]
    days = work.cfg.test_days()
    rules = []
    for d in ("rule_spike", "rule_stuck"):
        p = root / "scores" / f"{d}__test.parquet"
        if p.exists():
            rules.append(alarms_from(pd.read_parquet(p).assign(q=1.0), 0.0, merge))
    res = {"ref_days": ref_days, "threshold_q": pick["threshold_q"], "val_days": vdays, "val_per_day": pick["val_per_day"],
           "groups": {}, "curve": []}
    a_model = alarms_from(tp, pick["threshold_q"], merge)
    for name, a in (("model", a_model), ("combined", combine([a_model] + rules, merge))):
        m = match(a, faults)
        card = scorecard(m, faults)
        fa = false_alarms(m, ev, days)
        res["groups"][name] = {"scorecard": card.to_dict("records"), "false": {k: v for k, v in fa.items() if k != "_rows"}}
        log(f"  {name}: 탐지율 {card.iloc[-1]['rate'] * 100:.1f}%, 헛경보 {fa['per_day']:.2f}/일")
    for r in rows:                                   # 보고용 곡선(선택은 이미 끝남)
        a = combine([alarms_from(tp, r["threshold_q"], merge)] + rules, merge)
        m = match(a, faults)
        card = scorecard(m, faults)
        res["curve"].append({"threshold_q": r["threshold_q"], "val_per_day": r["val_per_day"],
                             "test_rate": float(card.iloc[-1]["rate"]),
                             "test_false_per_day": false_alarms(m, None, days)["per_day"]})
    return res

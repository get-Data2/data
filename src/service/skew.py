"""전처리 불일치 실험(README 과제 8 해결할 것 3 "다르면 어떤 일이 생기나", 생각해 볼 점 1 "부분 칸").

같은 모델에 전처리만 바꾼 입력을 넣어 60분 예측 오차를 비교한다. 정답은 학습과 같은 규칙의 닫힌 칸 공간 대표값이다.
- `same`     : 서비스 경로 그대로(닫힌 칸만, 앞으로 채움)
- `partial`  : 진행 중인 5분 칸(시작 뒤 2분치)을 마지막 칸으로 씀
- `zerofill` : 빈 칸을 앞으로 채우지 않고 0(정규화 평균)으로 둠
- `nonorm_space` : 공간별 정규화 통계 대신 실습실 통계를 씀(입력·출력 모두)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import preprocess as pp

from .control_sim import fetch_raw
from .prep import STEP, Prepared, forecast_input, prepare

VARIANTS = ("same", "partial", "zerofill", "nonorm_space")


def _input(variant, raw, cfg, runner, space, t0_bin, full_space):
    acfg = cfg.acfg
    look = runner.L * STEP
    if variant == "partial":
        now = t0_bin + STEP + pd.Timedelta(minutes=2)
        w = raw[(raw["ts"] >= now - look - STEP) & (raw["ts"] < now)]
        p = prepare(w, acfg, now + pd.Timedelta(minutes=3), cfg.spaces)       # 진행 중인 칸을 닫힌 칸처럼
        return forecast_input(p, runner, space, acfg), p.t0_bin
    sub = full_space[(full_space["bin"] <= t0_bin) & (full_space["bin"] > t0_bin - look)]
    p = Prepared(t0_bin + STEP, t0_bin + STEP, t0_bin, None, None, None, sub)
    if variant == "zerofill":
        orig = pp.causal

        def nofill(v, cap):
            f, o, a = orig(v, cap)
            return np.where(o.astype(bool), f, np.nan), o, a        # 관측 없는 칸은 NaN → forecast_input에서 0
        pp.causal = nofill
        try:
            return forecast_input(p, runner, space, acfg), t0_bin
        finally:
            pp.causal = orig
    return forecast_input(p, runner, space, acfg), t0_bin


def run(cfg, device, days: int = 3, every_min: int = 15, end: pd.Timestamp | None = None, log=print) -> pd.DataFrame:
    from .bundles import load_forecast
    from .window import DbSource
    runner = load_forecast(cfg, device).obj
    end = (end or pd.Timestamp.now(tz="UTC")).floor("5min") - pd.Timedelta(hours=1)       # 정답 60분이 닫혀 있게
    a = end - pd.Timedelta(days=days)
    raw = fetch_raw(DbSource(cfg), a - pd.Timedelta(hours=cfg["cycle"]["window_h"]), end + pd.Timedelta(hours=1, minutes=10))
    full = prepare(raw, cfg.acfg, end + pd.Timedelta(hours=1, minutes=10), cfg.spaces)
    truth = full.space[~full.space["filled"].astype(bool)]
    rows = []
    for space in cfg.spaces:
        tr = truth[truth["space_id"].astype(str) == space]
        tv = {it: g.set_index("bin")["value"] for it, g in tr.groupby(tr["item"].astype(str))}
        for t0 in pd.date_range(a, end, freq=f"{every_min}min"):
            for v in VARIANTS:
                if v == "nonorm_space" and space == "lab":
                    continue
                if v == "nonorm_space":
                    orig = runner.mu_sd
                    runner.mu_sd = lambda s, o=orig: o("lab")
                try:
                    inp, tb = _input(v, raw, cfg, runner, space, t0, full.space)
                    if inp is None:
                        continue
                    y = runner.predict(*inp[:2], space, inp[2])
                finally:
                    if v == "nonorm_space":
                        runner.mu_sd = orig
                target = tb + 12 * STEP
                for j, item in enumerate(("co2", "temperature")):
                    s = tv.get(item)
                    if s is not None and target in s.index:
                        rows.append({"space_id": space, "variant": v, "item": item, "t0_bin": tb,
                                     "err": abs(float(y[11, j]) - float(s[target]))})
        log(f"{space}: {sum(r['space_id'] == space for r in rows)}건")
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, days: int) -> str:
    from analysis.pipeline import _table
    m = df.groupby(["space_id", "item", "variant"])["err"].agg(["mean", "count"]).reset_index()
    rows = []
    for (sp, it), g in m.groupby(["space_id", "item"]):
        base = g[g["variant"] == "same"]["mean"]
        base = float(base.iloc[0]) if len(base) else np.nan
        for _, r in g.iterrows():
            rows.append({"공간": sp, "항목": it, "전처리": r["variant"], "60분 MAE": round(r["mean"], 2 if it == "temperature" else 1),
                         "같은 전처리 대비": "" if r["variant"] == "same" else f"{(r['mean'] / base - 1) * 100:+.0f}%",
                         "n": int(r["count"])})
    return "\n".join([f"최근 {days}일, 15분마다 기준 시각, 정답 = 닫힌 칸 공간 대표값(학습과 같은 규칙)", "", _table(pd.DataFrame(rows))])

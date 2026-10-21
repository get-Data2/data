"""비교표·예측 구간 적중률·보정 곡선 (research R6, FR 비교·구간).

비교표는 과제 3 `forecast.evaluate`의 기준선·모델 MAE 함수를 그대로 쓴다(같은 시점·같은 규칙).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.baseline import SPACE_NAME
from forecast.evaluate import baseline_by_cell, mae_by_cell

HORIZONS = (5, 15, 30, 60)


def comparison_table(cfg, frames: dict[str, pd.DataFrame], subset: str = "all") -> pd.DataFrame:
    """기준선 3종 + 모델들(라벨 → 예측). 모델 예측에 seed 열이 없으면 0으로 둔다(사전학습은 결정적)."""
    mp = cfg.acfg["baseline"].get("min_points", 1000)
    bl = pd.read_parquet(cfg["inputs"]["baseline"])
    bl = bl[(bl["item"] == cfg["item"]) & bl["space_id"].isin(cfg["spaces"])]
    t = baseline_by_cell(bl, subset, mp)
    for label, df in frames.items():
        d = df if "seed" in df.columns else df.assign(seed=0)
        m = mae_by_cell(d, subset, mp)[["space_id", "item", "horizon_min", "mae", "sd"]]
        t = t.merge(m.rename(columns={"mae": label, "sd": f"{label}_sd"}), on=["space_id", "item", "horizon_min"], how="left")
    t = t[t["horizon_min"].isin(HORIZONS)]
    order = {s: i for i, s in enumerate(cfg["spaces"])}
    return t.sort_values(["space_id", "horizon_min"], key=lambda c: c.map(order) if c.name == "space_id" else c).reset_index(drop=True)


def coverage_table(df: pd.DataFrame, subset: str = "all", min_points: int = 1000) -> pd.DataFrame:
    """10~90% 구간 적중률(이상 80%), 아래·위 이탈, 평균 폭. 예측 불가(현재값 대체) 행은 빼고 수를 따로 낸다."""
    d = df if subset == "all" else df[df["usage_hours"]]
    rows = []
    for (s, h), g in d.groupby(["space_id", "horizon_min"]):
        u = g[~g["unpredictable"]]
        n = len(u)
        r = {"space_id": s, "공간": SPACE_NAME.get(s, s), "horizon_min": h, "n": n, "예측 불가": int(g["unpredictable"].sum())}
        if n < min_points:
            r |= {"적중률": np.nan, "아래 이탈": np.nan, "위 이탈": np.nan, "평균 폭": np.nan}
        else:
            y = u["y_true"].to_numpy()
            lo, hi = u["p10"].to_numpy(), u["p90"].to_numpy()
            r |= {"적중률": float(((y >= lo) & (y <= hi)).mean()), "아래 이탈": float((y < lo).mean()),
                  "위 이탈": float((y > hi).mean()), "평균 폭": float((hi - lo).mean())}
        rows.append(r)
    return pd.DataFrame(rows)


def calibration(df: pd.DataFrame) -> pd.DataFrame:
    """명목 분위수(10·50·90%) 대비 실제로 정답이 그 아래에 있는 비율."""
    u = df[~df["unpredictable"]]
    rows = []
    for (s, h), g in u.groupby(["space_id", "horizon_min"]):
        for q, c in ((0.1, "p10"), (0.5, "p50"), (0.9, "p90")):
            rows.append({"space_id": s, "horizon_min": h, "nominal": q, "observed": float((g["y_true"] <= g[c]).mean())})
    return pd.DataFrame(rows)


def norm_mae(cfg, df: pd.DataFrame, norm: dict) -> float:
    """선택 지표: 공간 × 지평 MAE를 그 공간 CO₂ 표준편차(과제 2 학습 기간)로 나눈 평균."""
    vals = []
    for (s, h), g in df.groupby(["space_id", "horizon_min"]):
        if len(g) >= 100:
            vals.append(float((g["y_pred"] - g["y_true"]).abs().mean()) / norm["spaces"][s][cfg["item"]]["std"])
    return float(np.mean(vals)) if vals else float("nan")

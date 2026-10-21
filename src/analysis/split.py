"""분할과 학습 기간 통계(research R7). 학습 기간 칸만 읽는다 — 시험 기간 값은 어떤 통계에도 들어가지 않는다(FR-007)."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def _window(df: pd.DataFrame, rng, col="bin") -> pd.DataFrame:
    a, b = rng
    return df[(df[col] >= pd.Timestamp(a)) & (df[col] < pd.Timestamp(b))]


def fit_train_stats(ss: pd.DataFrame, sb: pd.DataFrame, sp, cfg) -> dict:
    """공간×항목 평균·표준편차·p99.9·칸 수, 센서-공간 편차의 MAD, 학습 구간 데이터 지문."""
    tr = sp.utc("train")
    s = _window(ss[~ss["filled"]], tr)
    out = {"split": sp.name, "train_utc": [str(tr[0]), str(tr[1])], "config": cfg.digest, "spaces": {}}
    q = cfg["axis_b"]["crowding"]["train_quantile"]
    for (space, item), g in s.groupby(["space_id", "item"], observed=True):
        v = g["value"].to_numpy()
        if len(v) == 0:
            continue
        out["spaces"].setdefault(str(space), {})[str(item)] = {
            "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            f"p{q * 100:g}": float(np.quantile(v, q)), "bins": int(len(v))}
    # 센서 값 − 같은 칸 공간 중앙값 → MAD(축 B B0 의심 판정용)
    b = _window(sb[~sb["filled"]], tr)
    j = b.merge(ss[["space_id", "item", "bin", "value"]].rename(columns={"value": "med"}),
                on=["space_id", "item", "bin"], how="inner")
    j["dev"] = j["value"] - j["med"]
    for (space, item), g in j.groupby(["space_id", "item"], observed=True):
        d = g["dev"].to_numpy()
        if len(d) and str(space) in out["spaces"] and str(item) in out["spaces"][str(space)]:
            out["spaces"][str(space)][str(item)]["mad"] = float(1.4826 * np.median(np.abs(d - np.median(d))))
    # 센서마다 자기 편향(편차 중앙값)과 자기 MAD — B0는 "평소의 그 센서"와 비교한다
    out["sensors"] = {}
    for (space, item, dev), g in j.groupby(["space_id", "item", "device_id"], observed=True):
        d = g["dev"].to_numpy()
        if len(d) >= 100:
            med = float(np.median(d))
            out["sensors"].setdefault(str(dev), {})[str(item)] = {
                "bias": med, "mad": float(1.4826 * np.median(np.abs(d - med))), "bins": int(len(d))}
    h = hashlib.sha256(pd.util.hash_pandas_object(s[["bin", "value"]], index=False).values.tobytes()).hexdigest()
    out["train_fingerprint"] = h[:16]
    return out


def autocorr(series: pd.Series, lags=(1, 12, 288)) -> dict[int, float]:
    """연속 칸 시계열(빈 칸 NaN)의 자기상관. 무작위 분할이 안 되는 근거(FR-011)."""
    x = series.astype(float)
    res = {}
    for k in lags:
        a, b = x.iloc[:-k].to_numpy(), x.iloc[k:].to_numpy()
        m = ~np.isnan(a) & ~np.isnan(b)
        res[k] = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else float("nan")
    return res


def split_table(ss: pd.DataFrame, sp, cfg) -> str:
    """분할 기간별 대상 공간·항목의 관측 칸 비율."""
    targets = cfg["space"]["targets"]
    interval = cfg["interval"]["minutes"]
    lines = [f"분할 {sp.name}",
             "| 기간 | KST | " + " | ".join(f"{t} co2" for t in targets) + " |",
             "|---|---|" + "---|" * len(targets)]
    for part in ("train", "val", "test"):
        a, b = sp.utc(part)
        expected = int((b - a).total_seconds() // 60 // interval)
        w = _window(ss[~ss["filled"] & (ss["item"].astype(str) == "co2")], (a, b))
        cells = []
        for t in targets:
            n = int((w["space_id"].astype(str) == t).sum())
            cells.append(f"{n / expected * 100:.0f}%")
        lines.append(f"| {part} | {getattr(sp, part)[0]} ~ {getattr(sp, part)[1]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)

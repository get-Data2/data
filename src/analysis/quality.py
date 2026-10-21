"""축 A — 센서 오류 판정(값 하나 + 기종). 원본은 바꾸지 않고 품질 표시만 돌려준다(spec 값 분류 체계).

ok: 정상 / saturated: 측정 상한·하한에 붙음(오류 아님, 축 B 재료) / error: 센서가 낼 수 없는 값(분석 입력에서 제외)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_axis_a(raw: pd.DataFrame, cfg, models: dict[str, str] | None = None) -> pd.DataFrame:
    """raw와 같은 순서·길이의 [quality, rule] 표."""
    a = cfg["axis_a"]
    models = models or {}
    n = len(raw)
    quality = np.full(n, "ok", dtype=object)
    rule = np.full(n, "", dtype=object)
    item = raw["item"].astype(str).to_numpy()
    value = raw["value"].to_numpy()

    # A1 오류 코드
    for r in a.get("error_codes") or []:
        m = (item == r["item"]) & np.isin(value, r["values"])
        quality[m], rule[m] = "error", "A1"

    # A2 오류 코드 쌍 — 같은 장치에서 pair_tolerance_s(기본 1초) 안에 쌍의 모든 항목이 그 값이면 모두 error.
    #   InfluxDB 과거분은 온도·습도가 1ms 차이 나는 별도 행으로 저장된 경우가 있다(2026-10-21 실측 83쌍)
    tol = pd.Timedelta(seconds=float(a.get("pair_tolerance_s", 1)))
    for pair in a.get("error_pairs") or []:
        its = list(pair.items())
        base_it, base_v = its[0]
        base = raw.loc[(item == base_it) & np.isclose(value, float(base_v)), ["device_id", "ts"]]
        base = base.assign(_i=base.index).sort_values("ts")
        base["device_id"] = base["device_id"].astype(str)
        matched = base
        hits = {base_it: None}
        for k, (it, v) in enumerate(its[1:]):
            col = f"_j{k}"
            other = raw.loc[(item == it) & np.isclose(value, float(v)), ["device_id", "ts"]]
            other = other.assign(**{col: other.index}).sort_values("ts")
            other["device_id"] = other["device_id"].astype(str)
            if matched.empty or other.empty:
                matched = matched.iloc[0:0]
                break
            matched = pd.merge_asof(matched, other, on="ts", by="device_id", tolerance=tol,
                                    direction="nearest").dropna(subset=[col])
            hits[it] = col
        if len(matched):
            idx = list(matched["_i"].astype(int))
            for col in (c for c in hits.values() if c is not None):
                idx += list(matched[col].astype(int))
            pos = raw.index.get_indexer(idx)
            quality[pos], rule[pos] = "error", "A2"

    # A3 제원 범위 밖(기종별 덮어쓰기)
    ranges = a.get("ranges") or {}
    model_over = a.get("models") or {}
    dev = raw["device_id"].astype(str).to_numpy()
    for it, (lo, hi) in ranges.items():
        m_item = item == it
        lo_arr = np.full(n, float(lo))
        hi_arr = np.full(n, float(hi))
        for model, over in model_over.items():
            if it in over:
                devs = [d for d, mo in models.items() if mo == model]
                mm = m_item & np.isin(dev, devs)
                lo_arr[mm], hi_arr[mm] = over[it]
        m = m_item & ((value < lo_arr) | (value > hi_arr)) & (quality == "ok")
        quality[m], rule[m] = "error", "A3"

    # A4 측정 상한·하한에 붙음 — 오류가 아니다
    for it, s in (a.get("saturation") or {}).items():
        m = (item == it) & (quality == "ok")
        if "upper" in s:
            mu = m & (value >= float(s["upper"]))
            quality[mu], rule[mu] = "saturated", "A4"
        if "lower" in s:
            ml = m & (value <= float(s["lower"]))
            quality[ml], rule[ml] = "saturated", "A4"

    return pd.DataFrame({"quality": pd.Categorical(quality, ["ok", "saturated", "error"]),
                         "rule": pd.Categorical(rule)}, index=raw.index)


def quality_counts(raw: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """공간·항목별 품질·규칙 건수(진단서·검산용)."""
    df = pd.DataFrame({"space_id": raw["space_id"].astype(str), "item": raw["item"].astype(str),
                       "quality": q["quality"].astype(str), "rule": q["rule"].astype(str)})
    return df.groupby(["space_id", "item", "quality", "rule"], observed=True).size().rename("rows").reset_index()

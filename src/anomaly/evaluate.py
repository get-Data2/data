"""경보 ↔ 고장 매칭과 지표: 종류별 탐지율·지연·하루당 헛경보·원인 정확도, 확인 기준 판정 (research R6)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import KIND_NAME, KINDS
from .data import split_col

KIND_ITEM = {"F4": "temperature", "F5": "co2"}


def match(alarms: pd.DataFrame, faults: pd.DataFrame) -> pd.DataFrame:
    """경보마다 매칭된 고장(fault_id, 없으면 0)과 지연. 경보 시작이 [고장 시작, 인정 끝] 안이고 같은 공간이면 매칭."""
    a = alarms.copy()
    a["fault_id"] = 0
    a["delay_min"] = np.nan
    for i, r in a.iterrows():
        f = faults[(faults["space_id"] == r["space_id"]) & (faults["start"] <= r["start"]) & (r["start"] <= faults["match_end"])]
        if len(f):
            f = f.sort_values("start").iloc[-1]          # 가장 최근에 시작한 고장
            a.at[i, "fault_id"] = int(f["fault_id"])
            a.at[i, "delay_min"] = (r["start"] - f["start"]) / pd.Timedelta(minutes=1)
    return a


def cause_ok(alarm_cause: str, f) -> bool:
    if not isinstance(alarm_cause, str):
        return False
    dev, item = split_col(alarm_cause)
    if f["kind"] in KIND_ITEM:
        return item == KIND_ITEM[f["kind"]]
    return dev == f["device_id"]


def scorecard(matched: pd.DataFrame, faults: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind in list(KINDS) + ["전체"]:
        fk = faults if kind == "전체" else faults[faults["kind"] == kind]
        hit, delays, causes = 0, [], []
        for f in fk.itertuples(index=False):
            m = matched[matched["fault_id"] == f.fault_id].sort_values("start")
            if len(m):
                hit += 1
                delays.append(float(m.iloc[0]["delay_min"]))
                causes.append(cause_ok(m.iloc[0]["cause"], f._asdict()))
        rows.append({"kind": kind, "name": KIND_NAME.get(kind, "전체"), "n": len(fk), "detected": hit,
                     "rate": hit / len(fk) if len(fk) else np.nan,
                     "delay_median_min": float(np.median(delays)) if delays else np.nan,
                     "delay_max_min": float(np.max(delays)) if delays else np.nan,
                     "cause_acc": float(np.mean(causes)) if causes else np.nan})
    return pd.DataFrame(rows)


def false_alarms(matched: pd.DataFrame, events: pd.DataFrame | None, days: int, near_min: int = 30) -> dict:
    fa = matched[matched["fault_id"] == 0].copy()
    fa["real_event"] = False
    if events is not None and len(events) and len(fa):
        pad = pd.Timedelta(minutes=near_min)
        for i, r in fa.iterrows():
            e = events[(events["space_id"] == r["space_id"]) & (events["start"] <= r["end"] + pad) & (events["end"] >= r["start"] - pad)]
            fa.at[i, "real_event"] = len(e) > 0
    n, nr = len(fa), int(fa["real_event"].sum())
    return {"false_alarms": n, "per_day": n / days, "real_event_overlap": nr, "per_day_excl_real": (n - nr) / days,
            "by_space": fa.groupby("space_id").size().to_dict(), "_rows": fa}


def criteria(card_combined: pd.DataFrame, fa_combined: dict) -> pd.DataFrame:
    rate = float(card_combined.loc[card_combined["kind"] == "전체", "rate"].iloc[0])
    return pd.DataFrame([
        {"기준": "모델+규칙 이벤트 탐지율 ≥ 80%", "결과": f"{rate * 100:.1f}%", "판정": "✅" if rate >= 0.8 else "❌"},
        {"기준": "헛경보 ≤ 하루 1회(전체 공간 합)", "결과": f"{fa_combined['per_day']:.2f}회/일 "
                                                        f"(실제 이상 후보 겹침 {fa_combined['real_event_overlap']}건 제외 시 {fa_combined['per_day_excl_real']:.2f})",
         "판정": "✅" if fa_combined["per_day"] <= 1 else "❌"},
    ])

"""규칙 경보: 데이터 끊김·값 고정·범위 밖 (research R6, 과제 1 끊김 기준 · 과제 4 값 고정 규칙)."""
from __future__ import annotations

import pandas as pd


def data_gap(raw: pd.DataFrame, now: pd.Timestamp, acfg, known: dict[str, str], gap_min: float,
             items=("co2", "temperature", "humidity")) -> list[dict]:
    """마지막 수신이 max(gap_min, 기대 주기 × 2)보다 오래된 주기형 장치. known: 서비스가 본 적 있는 장치 → 공간.
    items: raw에 든 항목. 이 항목을 주기적으로 보내지 않는 장치(꺼지면 power만 보내는 냉난방 실내기)는 판정하지 않는다."""
    last = raw.groupby("device_id")["ts"].max() if len(raw) else pd.Series(dtype="datetime64[ns, UTC]")
    for dev, sp in raw.groupby("device_id")["space_id"].last().items() if len(raw) else []:
        known[str(dev)] = str(sp)
    out = []
    for dev, space in known.items():
        if not acfg.is_periodic_in(dev, items):
            continue
        limit = max(gap_min, 2 * acfg.period_s(dev) / 60)
        seen = last.get(dev)
        age = (now - seen) / pd.Timedelta(minutes=1) if seen is not None and not pd.isna(seen) else None
        if age is None or age > limit:
            out.append({"device_id": dev, "space_id": space,
                        "last_seen": None if seen is None or pd.isna(seen) else seen.isoformat().replace("+00:00", "Z"),
                        "limit_min": round(limit, 1), "age_min": None if age is None else round(age, 1)})
    return out


def stuck(sensor: pd.DataFrame, p: dict) -> list[dict]:
    """닫힌 칸 관측값이 min_hours 이상 같고, 같은 공간 다른 장치의 같은 항목은 그동안 기준 이상 변한 장치·항목."""
    out = []
    if sensor.empty:
        return out
    obs = sensor[~sensor["filled"].astype(bool)]
    span = pd.Timedelta(hours=p["min_hours"])
    for (space, item), g in obs.groupby([obs["space_id"].astype(str), obs["item"].astype(str)]):
        devs = g["device_id"].astype(str).unique()
        for dev in devs:
            s = g[g["device_id"].astype(str) == dev].set_index("bin")["value"].sort_index()
            if len(s) < 2:
                continue
            chg = s.ne(s.shift()).cumsum()
            tail = s[chg == chg.iloc[-1]]
            if tail.index[-1] - tail.index[0] < span:
                continue
            ref = g[(g["device_id"].astype(str) != dev) & (g["bin"] >= tail.index[0])]
            if ref.empty:
                continue
            med = ref.groupby("bin")["value"].median()
            if med.max() - med.min() >= p["others_min_change"].get(item, float("inf")):
                out.append({"device_id": dev, "space_id": space, "item": item, "value": float(tail.iloc[-1]),
                            "since": tail.index[0].isoformat().replace("+00:00", "Z"),
                            "hours": round((tail.index[-1] - tail.index[0]) / pd.Timedelta(hours=1), 2)})
    return out


def out_of_range(raw: pd.DataFrame, quality: pd.DataFrame, now: pd.Timestamp, recent_min: float) -> list[dict]:
    """최근 recent_min 안에 축 A 오류(오류 코드·제원 범위 밖)를 낸 장치."""
    if raw.empty:
        return []
    m = (quality["quality"].astype(str) == "error").to_numpy() & (raw["ts"] >= now - pd.Timedelta(minutes=recent_min)).to_numpy()
    d = raw[m]
    out = []
    for (dev, item), g in d.groupby(["device_id", "item"]):
        out.append({"device_id": str(dev), "space_id": str(g["space_id"].iloc[-1]), "item": str(item),
                    "value": float(g["value"].iloc[-1]), "rule": str(quality.loc[g.index[-1], "rule"]), "count": int(len(g))})
    return out

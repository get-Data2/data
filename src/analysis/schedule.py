"""일과표·공휴일·CO₂ 반응 지연·어긋나는 날(research R9).

일과표 행(공간·요일·시작·끝)에서 날짜별 사건(시작=상승 기대, 끝=하강 기대)을 만든다. 실제 시간표 CSV(schedule.file)도 같은 형식으로 읽는다.
반응 지연 = 사건 시각부터, 사건 전 pre_min분 평균에서 기대 방향으로 delta_ppm을 처음 넘는 칸까지의 분. window_min 안에 없으면 무반응.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import holidays
import numpy as np
import pandas as pd

from .config import KST, WEEKDAYS

EVENT_NAME = {("start", "09:00"): "수업 시작", ("end", "11:30"): "점심 시작", ("start", "13:00"): "점심 끝(오후 시작)",
              ("end", "18:00"): "퇴실"}


def schedule_rows(cfg) -> pd.DataFrame:
    sc = cfg["schedule"]
    if sc.get("file"):
        df = pd.read_csv(sc["file"])
        if "weekday" in df.columns:
            df["days"] = df["weekday"].apply(lambda x: [str(x)[:3].lower()])
        return df.rename(columns={"space_id": "space"})
    return pd.DataFrame(sc["rows"])


def is_class_day(d: date, cfg, kr=None) -> bool:
    kr = kr or holidays.KR(years=d.year)
    if cfg["schedule"].get("exclude_holidays") and d in kr:
        return False
    return d.weekday() < 5


def events_for(start: date, end: date, cfg) -> pd.DataFrame:
    """[space_id, date, t(UTC), kind(start/end), hhmm, name]"""
    rows = schedule_rows(cfg)
    kr = holidays.KR(years=range(start.year, end.year + 1))
    out = []
    d = start
    while d <= end:
        if not (cfg["schedule"].get("exclude_holidays") and d in kr):
            wd = WEEKDAYS[d.weekday()]
            for r in rows.itertuples():
                if wd not in r.days:
                    continue
                for kind, hhmm in (("start", r.start), ("end", r.end)):
                    t = datetime.combine(d, time(int(hhmm[:2]), int(hhmm[3:])), KST)
                    out.append({"space_id": r.space, "date": d, "t": pd.Timestamp(t).tz_convert("UTC"), "kind": kind,
                                "hhmm": hhmm, "name": EVENT_NAME.get((kind, hhmm), f"{kind} {hhmm}")})
        d += timedelta(days=1)
    return pd.DataFrame(out).drop_duplicates(["space_id", "t", "kind"]) if out else pd.DataFrame(
        columns=["space_id", "date", "t", "kind", "hhmm", "name"])


def reaction_lags(series: pd.Series, events: pd.DataFrame, cfg) -> pd.DataFrame:
    """series: 한 공간 CO₂ 연속 칸(UTC 인덱스). 사건마다 반응 지연(분) — 무반응이면 NaN."""
    rc = cfg["schedule"]["reaction"]
    out = []
    for e in events.itertuples():
        pre = series[(series.index >= e.t - pd.Timedelta(minutes=rc["pre_min"])) & (series.index < e.t)].dropna()
        post = series[(series.index > e.t) & (series.index <= e.t + pd.Timedelta(minutes=rc["window_min"]))].dropna()
        if len(pre) == 0 or len(post) == 0:
            out.append({**e._asdict(), "base": np.nan, "lag_min": np.nan, "measured": False})
            continue
        base = float(pre.mean())
        sign = 1 if e.kind == "start" else -1
        hit = post[(post - base) * sign >= rc["delta_ppm"]]
        lag = (hit.index[0] - e.t).total_seconds() / 60 if len(hit) else np.nan
        out.append({**e._asdict(), "base": base, "lag_min": lag, "measured": True})
    df = pd.DataFrame(out)
    return df.drop(columns=["Index"], errors="ignore")


def mismatch_days(series: pd.Series, lags: pd.DataFrame, start: date, end: date, cfg) -> pd.DataFrame:
    """일정과 실제 CO₂가 어긋나는 날: 수업일인데 09시 시작에 무반응 / 비수업일인데 사용 수준으로 상승."""
    rc = cfg["schedule"]["reaction"]
    out = []
    first = lags[(lags["kind"] == "start") & (lags["hhmm"] == lags["hhmm"].min())] if len(lags) else lags
    for r in first.itertuples():
        if r.measured and np.isnan(r.lag_min):
            out.append({"date": r.date, "type": "수업일 무반응", "detail": f"{r.name} 뒤 {rc['window_min']}분 안에 +{rc['delta_ppm']}ppm 없음"})
    kr = holidays.KR(years=range(start.year, end.year + 1))
    k = series.copy()
    k.index = k.index.tz_convert(KST)
    for d, g in k.groupby(k.index.date):
        if d < start or d > end or is_class_day(d, cfg, kr):
            continue
        # 주간(08~20시)만 본다 — 자정 무렵 최고점은 전날 밤 남은 CO₂가 빠지는 중이지 사용이 아니다(2026-10-21 오판 수정)
        g = g[(g.index.hour >= 8) & (g.index.hour < 20)].dropna()
        if len(g) < 60:
            continue
        rise = float(g.max() - g.min())
        if rise >= rc["nonclass_use_ppm"]:
            why = "공휴일" if d in kr else "주말"
            out.append({"date": d, "type": f"비수업일 사용({why})", "detail": f"하루 상승폭 {rise:.0f}ppm, 최고 {g.idxmax():%H:%M}"})
    return pd.DataFrame(out, columns=["date", "type", "detail"]).sort_values("date").reset_index(drop=True)

"""가동 요약(`infer report`): 주기 수·처리 시간 분포·경보 내역·장애 시험·메모리 (research R9, tasks T032)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.pipeline import _table


def _load(recorder, kind, a, b, col) -> pd.DataFrame:
    rows = recorder.read(kind)
    df = pd.DataFrame(rows)
    if df.empty or col not in df:
        return df
    t = pd.to_datetime(df[col], utc=True, format="ISO8601")
    return df[(t >= a) & (t < b)].assign(_t=t[(t >= a) & (t < b)])


def summarize(recorder, a: pd.Timestamp, b: pd.Timestamp) -> str:
    cyc = _load(recorder, "cycle", a, b, "scheduled_at")
    al = _load(recorder, "alert", a, b, "at")
    ev = _load(recorder, "event", a, b, "at")
    hours = (b - a) / pd.Timedelta(hours=1)
    expected = int(hours * 60)
    md = [f"# 상시 추론 서비스 가동 요약", "",
          f"- 기간: {a.tz_convert('Asia/Seoul'):%Y-%m-%d %H:%M} ~ {b.tz_convert('Asia/Seoul'):%Y-%m-%d %H:%M} KST ({hours:.1f}시간)",
          "- 측정 조건: 실데이터(수집기 MySQL), Jetson Orin Nano Super 25W(클럭 고정 안 함), 주기 매분", ""]
    if cyc.empty:
        return "\n".join(md + ["주기 기록 없음"])
    st = cyc["status"].value_counts().to_dict()
    md += ["## 주기", "", _table(pd.DataFrame([{"예정": expected, "기록": len(cyc), "ok": st.get("ok", 0),
                                                "partial": st.get("partial", 0), "failed": st.get("failed", 0),
                                                "skipped": st.get("skipped", 0),
                                                "기록 없음(서비스 중단 추정)": max(0, expected - len(cyc))}])), ""]
    run = cyc[cyc["status"] != "skipped"]
    rows = []
    for c in ("read_ms", "prep_ms", "infer_ms", "publish_ms", "total_ms"):
        if c in run:
            v = pd.to_numeric(run[c], errors="coerce").dropna()
            if len(v):
                rows.append({"단계": c.replace("_ms", ""), "p50(ms)": int(v.median()), "p95(ms)": int(v.quantile(0.95)),
                             "최대(ms)": int(v.max())})
    md += ["## 처리 시간", "", _table(pd.DataFrame(rows)), ""]
    if "rss_mb" in run:
        r = pd.to_numeric(run["rss_mb"], errors="coerce").dropna()
        md += [f"- 프로세스 최대 메모리(RSS): {r.max():.0f}MB (시작 {r.iloc[0]:.0f}MB)", ""]
    err = run[run["status"] != "ok"]
    if len(err):
        md += ["## 실패·부분 주기 원인", "", _table(err.groupby("error").size().rename("주기 수").reset_index()), ""]
    md += ["## 경보", ""]
    if al.empty:
        md += ["경보 없음", ""]
    else:
        g = al.groupby(["kind", "event"]).size().unstack(fill_value=0).reset_index()
        md += [_table(g), ""]
    md += ["## 장애 시험·운영 사건", ""]
    md += [_table(ev[["at", "kind", "detail"]]) if not ev.empty else "기록 없음", ""]
    return "\n".join(md)

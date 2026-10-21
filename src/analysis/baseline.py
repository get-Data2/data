"""평가 시점과 기준선 3종(research R8). 평가 시점 파일은 과제 3~6이 그대로 쓴다.

공정성 규칙
- 입력(현재값·최근 1시간·어제 같은 시각)과 정답은 **관측한 칸만** 쓴다. 보간 칸은 앞뒤(미래) 값을 써서 만든 값이라
  예측 시점에 알 수 없다. 주기가 긴 센서의 '주기 안 직전값'도 관측으로 보지 않는다(정답이 부풀지 않게).
- 세 기준선은 같은 평가 시점 집합을 쓴다(입력·정답이 모두 있는 칸만).
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from .config import KST, WEEKDAYS
from .pipeline import EVAL_DIR, _table, load_series, sha256_file

METHODS = {"persistence": "현재값 유지", "mean_1h": "최근 1시간 평균", "yesterday": "어제 같은 시각"}
SPACE_NAME = {"lab": "실습실", "office": "사무실", "meeting": "회의실"}


def observed_grid(ss: pd.DataFrame, space: str, item: str, interval: int) -> pd.Series:
    s = ss[(ss["space_id"].astype(str) == space) & (ss["item"].astype(str) == item) & (~ss["filled"])]
    s = s.set_index("bin")["value"].sort_index()
    if s.empty:
        return s
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq=f"{interval}min"))


def usage_mask(t0: pd.DatetimeIndex, cfg) -> np.ndarray:
    u = cfg["baseline"]["usage_hours"]
    k = t0.tz_convert(KST)
    days = {WEEKDAYS.index(d) for d in u["weekdays"]}
    hhmm = k.hour * 60 + k.minute
    a = int(u["start"][:2]) * 60 + int(u["start"][3:])
    b = int(u["end"][:2]) * 60 + int(u["end"][3:])
    m = k.weekday.isin(list(days)) & (hhmm >= a) & (hhmm < b)
    if u.get("exclude_holidays"):
        kr = holidays.KR(years=sorted(set(k.year)))
        m &= ~pd.Index(k.date).isin(list(kr.keys()))
    return np.asarray(m)


def build(ss: pd.DataFrame, sp, cfg) -> pd.DataFrame:
    """평가 시점 + 세 기준선 예측을 한 번에 만든다(같은 시점 보장)."""
    bl = cfg["baseline"]
    interval = cfg["interval"]["minutes"]
    win = bl["mean_window_min"] // interval
    min_p = math.ceil(win * bl["mean_min_coverage"])
    day = 24 * 60 // interval
    a, b = sp.utc("test")
    out = []
    for space in cfg["space"]["targets"]:
        for item in bl["items"]:
            y = observed_grid(ss, space, item, interval)
            if y.empty:
                continue
            mean1h = y.rolling(win, min_periods=min_p).mean()
            for h in bl["horizons_min"]:
                hb = h // interval
                df = pd.DataFrame({"persistence": y, "mean_1h": mean1h,
                                   "yesterday": y.shift(day - hb), "y_true": y.shift(-hb)})
                df = df[(df.index >= a) & (df.index + pd.Timedelta(minutes=h) < b)].dropna()
                if df.empty:
                    continue
                df = df.rename_axis("t0").reset_index()
                df.insert(0, "space_id", space)
                df.insert(1, "item", item)
                df.insert(3, "horizon_min", h)
                out.append(df)
    res = pd.concat(out, ignore_index=True)
    res["usage_hours"] = usage_mask(pd.DatetimeIndex(res["t0"]), cfg)
    return res


def mae_table(res: pd.DataFrame, subset: str = "all", cfg=None) -> pd.DataFrame:
    """공간×항목×지평 MAE. 시점이 min_points 미만이면 숫자 대신 NaN(표에서 '평가 불가')."""
    d = res if subset == "all" else res[res["usage_hours"]]
    min_points = cfg["baseline"].get("min_points", 0) if cfg else 0
    rows = []
    for space in dict.fromkeys(res["space_id"]):
        for item in dict.fromkeys(res["item"]):
            for h in sorted(set(res["horizon_min"])):
                g = d[(d["space_id"] == space) & (d["item"] == item) & (d["horizon_min"] == h)]
                r = {"공간": f"{SPACE_NAME.get(space, space)}({space})", "항목": item, "지평(분)": h}
                ok = len(g) >= max(min_points, 1)
                for m, name in METHODS.items():
                    r[name] = round(float(np.mean(np.abs(g[m] - g["y_true"]))), 2) if ok else np.nan
                r["평가 시점 수"] = int(len(g))
                rows.append(r)
    return pd.DataFrame(rows)


def run_baseline(cfg, split_name=None) -> str:
    sp = cfg.split(split_name)
    ss = load_series("space")
    res = build(ss, sp, cfg)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    pts = res[["space_id", "item", "t0", "horizon_min", "y_true", "usage_hours"]]
    p_pts = EVAL_DIR / f"eval_points_{sp.name}.parquet"
    pts.to_parquet(p_pts, index=False)
    sha = sha256_file(p_pts)
    meta = {"split": sp.name, "rows": len(pts), "sha256": sha, "config": cfg.digest,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rule": "입력·정답은 관측 칸만, 세 기준선 공통 시점. 과제 3~6은 이 파일의 행만 예측·채점"}
    (EVAL_DIR / f"eval_points_{sp.name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    p_bl = EVAL_DIR / f"baseline_{sp.name}.parquet"
    res.to_parquet(p_bl, index=False)
    (EVAL_DIR / f"baseline_{sp.name}.json").write_text(json.dumps(
        {"split": sp.name, "eval_points_sha256": sha, "config": cfg.digest}, ensure_ascii=False, indent=1))

    t_all, t_use = mae_table(res, "all", cfg), mae_table(res, "usage", cfg)
    md = report_markdown(cfg, sp, res, t_all, t_use, sha)
    out = Path(cfg["report"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / f"baseline_{sp.name}.md").write_text(md, encoding="utf-8")
    return _table(t_all) + f"\n\n평가 시점 {len(pts):,}행, sha256 {sha[:16]} → {p_pts}\n보고서 → {out}/baseline_{sp.name}.md"


def winners(t: pd.DataFrame) -> pd.DataFrame:
    cols = list(METHODS.values())
    w = t.copy()
    na = w[cols].isna().all(axis=1)
    w["1위"] = ""
    w.loc[~na, "1위"] = w.loc[~na, cols].idxmin(axis=1)
    w.loc[na, "1위"] = "평가 불가: 정답 관측 칸 부족(10분 주기 센서는 5·15분 뒤 칸이 대부분 미관측)"
    w[cols] = w[cols].astype(object).where(~w[cols].isna(), "—")
    return w


def report_markdown(cfg, sp, res, t_all, t_use, sha) -> str:
    a, b = sp.test
    lines = [
        f"# 기준선 성능표 (과제 2, 분할 {sp.name})", "",
        "- 측정 조건",
        f"  - 실데이터, 시험 기간 {a} ~ {b}(KST), 분석 간격 {cfg['interval']['minutes']}분, 대상: 공간 대표값(센서 중앙값)",
        f"  - 설정 {cfg.digest}, 평가 시점 sha256 `{sha[:16]}…` ({len(res):,}행 = 공간×항목×지평)",
        "- 규칙: 입력·정답은 관측 칸만(보간 칸 제외). 세 방법은 같은 시점. MAE 단위는 CO₂ ppm, 온도 ℃",
        "- 방법: 현재값 유지 = y(t0) / 최근 1시간 평균 = 지난 60분 관측 평균(칸 50% 이상) / 어제 같은 시각 = y(t0 + 지평 − 24시간)", "",
        "## 전체 시간", "", _table(winners(t_all)), "",
        "## 사용 시간대(평일·공휴일 제외·08~19시)", "", _table(winners(t_use)), "",
    ]
    return "\n".join(lines)

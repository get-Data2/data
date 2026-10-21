"""심화 1: 예측 기반 환기 규칙을 지난 N일 실데이터에 적용했을 때 1,000ppm 초과 시간 추정 (research R11, tasks T035).

방법
- MySQL 1분 원본 → 서비스와 같은 전처리(prep.prepare) → 5분 칸 공간 대표값.
- 5분마다 그 시각까지의 칸만으로 서비스 경로 예측(forecast_input + ForecastRunner)을 다시 낸다(재생).
- 질량 수지: dC/dt = G(t) − (C − C_out)/τ. 실제 궤적과 자연 환기 τ_nat로 발생량 G를 거꾸로 구하고,
  환기 중에는 τ_nat/k로 바꿔 CO₂ 궤적을 다시 적분한다.
- τ_nat는 같은 기간 평일 퇴실(18:30~22:00) 감쇠에서 공간마다 추정한다.

가정은 결과 표에 함께 적는다. 실제 제어는 하지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .prep import STEP, Prepared, forecast_input, prepare

C_OUT = 450.0


def fetch_raw(src, a: pd.Timestamp, b: pd.Timestamp, chunk_h: int = 24) -> pd.DataFrame:
    parts, t = [], a
    while t < b:
        u = min(t + pd.Timedelta(hours=chunk_h), b)
        parts.append(src.fetch(t, u))
        t = u
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True).sort_values("ts").reset_index(drop=True) if parts else pd.DataFrame()


def replay(space_bins: pd.DataFrame, runner, space: str, acfg, bins: pd.DatetimeIndex, batch: int = 512) -> pd.DataFrame:
    """칸마다 (실측 CO₂, 30분 안 예측 최댓값, 60분 예측). 입력은 서비스 경로(forecast_input)로 t0 칸까지의 닫힌 칸만 본다."""
    ss = space_bins[space_bins["space_id"].astype(str) == space]
    look = runner.L * STEP
    keep, inputs = [], []
    for t0 in bins:
        sub = ss[(ss["bin"] <= t0) & (ss["bin"] > t0 - look)]
        prep = Prepared(t0 + STEP, t0 + STEP, t0, None, None, None, sub)
        inp = forecast_input(prep, runner, space, acfg) if len(sub) else None
        if inp is not None:
            keep.append(t0)
            inputs.append(inp)
    df = pd.DataFrame(index=pd.DatetimeIndex(bins, name="bin"), columns=["pred30_max", "pred60"], dtype=float)
    for i in range(0, len(inputs), batch):
        part = inputs[i:i + batch]
        y = runner.predict_many(np.stack([p[0] for p in part]), np.stack([p[1] for p in part]), [space] * len(part),
                                np.stack([p[2] for p in part]))
        idx = keep[i:i + batch]
        df.loc[idx, "pred30_max"] = y[:, :6, 0].max(axis=1)
        df.loc[idx, "pred60"] = y[:, 11, 0]
    obs = ss[(ss["item"].astype(str) == "co2") & (~ss["filled"].astype(bool))].set_index("bin")["value"]
    df["co2"] = obs[~obs.index.duplicated()].reindex(df.index)
    return df


def estimate_tau(co2: pd.Series, hours=(18.5, 22.0), min_start=700.0) -> tuple[float, int]:
    """평일 저녁 감쇠 구간에서 ln(C − C_out)의 기울기 → τ(분). (중앙값, 쓴 날 수)"""
    taus = []
    k = co2.index.tz_convert("Asia/Seoul")
    for d, g in co2.groupby(k.date):
        if pd.Timestamp(d).weekday() >= 5:
            continue
        kk = g.index.tz_convert("Asia/Seoul")
        h = kk.hour + kk.minute / 60
        s = g[(h >= hours[0]) & (h <= hours[1])].dropna()
        s = s[s > C_OUT + 30]
        if len(s) < 12 or s.iloc[0] < min_start:
            continue
        x = (s.index - s.index[0]) / pd.Timedelta(minutes=1)
        slope = np.polyfit(np.asarray(x, float), np.log(s.to_numpy() - C_OUT), 1)[0]
        if slope < 0:
            taus.append(-1 / slope)
    return (float(np.median(taus)) if taus else np.nan), len(taus)


def simulate(df: pd.DataFrame, tau_nat: float, k: float, policy: str, thr=1000.0, off_ppm=800.0,
             lead_min=30, hold_min=30) -> pd.DataFrame:
    """policy: none | reactive(현재값 ≥ thr이면 켬) | predictive(현재값 ≥ thr 또는 lead_min 안 예측 ≥ thr이면 켬).
    끔: 다시 켤 조건이 없고 모의 CO₂ < off_ppm인 상태가 hold_min 이어지면."""
    c = df["co2"].interpolate(limit=3, limit_area="inside").to_numpy(float)
    pred = df["pred30_max"].to_numpy(float)
    dt = STEP / pd.Timedelta(minutes=1)
    n = len(c)
    sim = c.copy()
    on = np.zeros(n, bool)
    vent, calm = False, 0
    hold = int(hold_min / dt)
    for i in range(n):
        if i > 0 and not np.isnan(c[i]) and not np.isnan(c[i - 1]) and not np.isnan(sim[i - 1]):
            g = (c[i] - c[i - 1]) / dt + (c[i - 1] - C_OUT) / tau_nat        # 실제 궤적의 발생량
            tau = tau_nat / k if on[i - 1] else tau_nat
            sim[i] = max(C_OUT, sim[i - 1] + dt * (g - (sim[i - 1] - C_OUT) / tau))
        elif i > 0 and np.isnan(c[i]):
            sim[i] = np.nan
        cur = sim[i]
        trig = False
        if policy != "none" and not np.isnan(cur):
            trig = cur >= thr
            if policy == "predictive" and not np.isnan(pred[i]):
                # 예측은 실제 궤적 기준이다. 모의 궤적으로 옮기려고 현재값 차이만큼 내린다.
                trig = trig or (pred[i] - (c[i] - cur)) >= thr
        if trig:
            vent, calm = True, 0
        elif vent:
            calm = calm + 1 if (not np.isnan(cur) and cur < off_ppm) else 0
            if calm >= hold:
                vent = False
        on[i] = vent
    return pd.DataFrame({"co2": c, "sim": sim, "vent": on}, index=df.index)


def summarize(sims: dict, occupied: pd.Series) -> dict:
    out = {}
    for key, s in sims.items():
        dt_h = STEP / pd.Timedelta(hours=1)
        over = (s["sim"] >= 1000).sum() * dt_h
        out[key] = {"over_h": round(float(over), 1), "vent_h": round(float(s["vent"].sum() * dt_h), 1),
                    "peak": round(float(np.nanmax(s["sim"])), 0),
                    "over_occupied_h": round(float(((s["sim"] >= 1000) & occupied.reindex(s.index).fillna(False)).sum() * dt_h), 1)}
    return out


def run(cfg, device, days: int = 7, end: pd.Timestamp | None = None, ks=(2.0, 3.0), log=print) -> dict:
    from .bundles import load_forecast
    from .window import DbSource
    fb = load_forecast(cfg, device)
    if not fb.ok:
        raise RuntimeError(f"예측 묶음 없음: {fb.availability()}")
    runner = fb.obj
    end = (end or pd.Timestamp.now(tz="UTC")).floor("5min")
    a = end - pd.Timedelta(days=days)
    raw = fetch_raw(DbSource(cfg), a - pd.Timedelta(hours=cfg["cycle"]["window_h"]), end)
    log(f"원본 {len(raw):,}행, {a.tz_convert('Asia/Seoul'):%m-%d %H:%M} ~ {end.tz_convert('Asia/Seoul'):%m-%d %H:%M} KST")
    prep = prepare(raw, cfg.acfg, end, cfg.spaces)
    bins = pd.date_range(a, end - STEP, freq="5min")
    res = {"period": [str(a), str(end)], "days": days, "spaces": {}}
    frames = {}
    for space in cfg.spaces:
        if space not in runner.spaces:
            continue
        df = replay(prep.space, runner, space, cfg.acfg, bins)
        tau, nd = estimate_tau(df["co2"])
        if np.isnan(tau):
            log(f"{space}: 감쇠 구간 부족 — 건너뜀")
            continue
        h = df.index.tz_convert("Asia/Seoul")
        occupied = pd.Series((h.weekday < 5) & (h.hour >= 9) & (h.hour < 18), index=df.index)
        sims = {"none": simulate(df, tau, 1, "none")}
        for k in ks:
            sims[f"reactive_k{k:g}"] = simulate(df, tau, k, "reactive")
            sims[f"predictive_k{k:g}"] = simulate(df, tau, k, "predictive")
        s = summarize(sims, occupied)
        valid = df["pred30_max"].notna().mean()
        res["spaces"][space] = {"tau_nat_min": round(tau, 1), "tau_days": nd, "pred_coverage": round(float(valid), 3),
                                "co2_coverage": round(float(df["co2"].notna().mean()), 3), "results": s}
        frames[space] = (df, sims)
        log(f"{space}: τ_nat {tau:.0f}분({nd}일), 초과 {s['none']['over_h']}h → 예측 규칙 k=2 {s['predictive_k2']['over_h']}h")
    res["_frames"] = frames
    return res


def to_markdown(res: dict) -> str:
    from analysis.pipeline import _table
    a, b = (pd.Timestamp(x).tz_convert("Asia/Seoul") for x in res["period"])
    rows = []
    for sp, r in res["spaces"].items():
        base = r["results"]["none"]
        for key, v in r["results"].items():
            red = (1 - v["over_h"] / base["over_h"]) * 100 if base["over_h"] else np.nan
            rows.append({"공간": sp, "τ_nat(분)": r["tau_nat_min"], "규칙": key, "1,000ppm 초과(h)": v["over_h"],
                         "감소": "" if key == "none" else (f"{red:.0f}%" if not np.isnan(red) else "—"),
                         "일과 중 초과(h)": v["over_occupied_h"], "환기 시간(h)": v["vent_h"], "최고(ppm)": int(v["peak"])})
    return "\n".join([f"기간 {a:%Y-%m-%d %H:%M} ~ {b:%Y-%m-%d %H:%M} KST ({res['days']}일)", "", _table(pd.DataFrame(rows))])

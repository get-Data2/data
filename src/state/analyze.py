"""오류 분석(US5)과 심화(US6): 전환 뒤 경과별 오류, 인원별 재현율, 상위 혼동, 환기 헛경보 시각대,
처음 보는 공간, 노쇼·예약 없는 사용 (research R10·R11)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.config import KST

from .models import Candidate, Model
from .selection import metrics

BUCKETS = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 10**6)]


def since_change(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """공간별로 정답이 바뀐 뒤 지난 분과 직전 전환 방향(입실 → 사용중, 퇴실 → 비어있음)."""
    out = []
    for s, g in df.sort_values("bin").groupby("space_id"):
        y = g[label_col].to_numpy()
        t = pd.DatetimeIndex(g["bin"])
        last_t, last_dir = None, None
        mins, dirs = [], []
        for i in range(len(g)):
            if i and not np.isnan(y[i]) and not np.isnan(y[i - 1]) and y[i] != y[i - 1] and (t[i] - t[i - 1]) <= pd.Timedelta(minutes=10):
                last_t, last_dir = t[i], "입실" if y[i] == 1 else "퇴실"
            mins.append((t[i] - last_t) / pd.Timedelta(minutes=1) if last_t is not None else np.nan)
            dirs.append(last_dir)
        out.append(g.assign(since_min=mins, since_dir=dirs))
    return pd.concat(out) if out else df.assign(since_min=np.nan, since_dir=None)


def transition_errors(df: pd.DataFrame, label_col: str, pred_col: str) -> pd.DataFrame:
    d = since_change(df.dropna(subset=[label_col]), label_col)
    rows = []
    for dr in ("입실", "퇴실"):
        g = d[d["since_dir"] == dr]
        for a, b in BUCKETS:
            h = g[(g["since_min"] >= a) & (g["since_min"] < b)]
            if len(h):
                rows.append({"전환": dr, "경과(분)": f"{a}~{b if b < 10**6 else ''}", "칸 수": len(h),
                             "오류율": round(float((h[pred_col] != h[label_col]).mean()), 3)})
    return pd.DataFrame(rows)


def people_recall(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    d = df[(df["obs_label"] == 1) & df["obs_people"].notna()]
    if d.empty:
        return pd.DataFrame()
    bins = pd.cut(d["obs_people"], [0, 2, 5, 10, 20, 1000], labels=["1~2", "3~5", "6~10", "11~20", "21+"])
    return d.groupby(bins, observed=True).apply(lambda g: pd.Series({"칸 수": len(g), "재현율": round(float(g[pred_col].mean()), 3)})).reset_index()


def top_confusions(df: pd.DataFrame, label_col: str, pred_col: str, n=3) -> pd.DataFrame:
    d = df.dropna(subset=[label_col]).copy()
    d = d[d[pred_col] != d[label_col]]
    if d.empty:
        return pd.DataFrame()
    d["hour"] = pd.to_datetime(d["bin"]).dt.tz_convert(KST).dt.hour
    d["유형"] = np.where(d[pred_col] == 1, "비어있음→사용중", "사용중→비어있음")
    g = d.groupby(["space_id", "hour", "유형"]).agg(칸수=("bin", "size"), 예시=("bin", "first")).reset_index()
    g["예시"] = pd.to_datetime(g["예시"]).dt.tz_convert(KST).dt.strftime("%m-%d %H:%M")
    return g.sort_values("칸수", ascending=False).head(n)


def vent_false_alarms(te: pd.DataFrame, days: int) -> pd.DataFrame:
    fa = te[(te["pred"] == 1) & (te["y"] == 0)].copy()
    fa["hour"] = pd.to_datetime(fa["bin"]).dt.tz_convert(KST).dt.hour
    t = fa.groupby(["space_id", "hour"]).size().rename("헛경보 칸").reset_index()
    t["하루당"] = (t["헛경보 칸"] / days).round(2)
    return t


def unseen_space(cfg, root, sel) -> pd.DataFrame:
    """심화 1: 일정 정답이 있는 공간 하나로만 학습하고 다른 공간을 시험(회의실은 일정 정답이 없어 관찰로만)."""
    from .pipeline import dataset
    d = dataset(root, cfg, "occupancy")
    p = sel["occupancy"]
    cols = p["features"]
    cand = Candidate(p["kind"], p["params"])
    spaces = sorted(d.dropna(subset=["y"])["space_id"].unique())
    rows = []
    for held in spaces + ["meeting"]:
        train_sp = [s for s in spaces if s != held]
        tr = d[(d["split"] == "train") & d["y"].notna() & d["space_id"].isin(train_sp)]
        va = d[(d["split"] == "val") & d["y"].notna() & d["space_id"].isin(train_sp)]
        if tr["y"].nunique() < 2:
            continue
        m = Model(cand).fit(tr[cols].to_numpy(), tr["y"].to_numpy(), 0)
        from .selection import occ_threshold
        thr, _ = occ_threshold(va["y"], m.predict_proba(va[cols].to_numpy()))
        for basis, ycol, sub in (("일정(시험)", "y", d[(d["split"] == "test") & d["y"].notna() & (d["space_id"] == held)]),
                                 ("관찰", "obs_label", d[d["obs_label"].notna() & (d["space_id"] == held)])):
            if sub.empty:
                continue
            mm = metrics(sub[ycol], m.predict_proba(sub[cols].to_numpy()) >= thr)
            rows.append({"뺀 공간": held, "학습 공간": ",".join(train_sp), "정답": basis, "칸 수": mm["n"],
                         "macro-F1": round(mm["macro_f1"], 3), "비어있음→사용중": round(mm["false_occupied_rate"], 3)
                         if not np.isnan(mm["false_occupied_rate"]) else None})
    return pd.DataFrame(rows)


def noshow_unbooked(pred: pd.DataFrame, bookings: pd.DataFrame | None, cfg) -> pd.DataFrame:
    """심화 2: 노쇼(예약 시작 뒤 15분 계속 비어있음), 예약 없는 사용(예약 밖 15분 이상 사용중)."""
    if bookings is None or bookings.empty:
        return pd.DataFrame(columns=["space_id", "kind", "start", "end"])
    a = cfg["advanced"]
    out = []
    for s, g in pred.sort_values("bin").groupby("space_id"):
        b = bookings[bookings["space_id"] == s]
        if b.empty:
            continue
        t = pd.DatetimeIndex(g["bin"])
        occ = g["pred"].to_numpy()
        inb = np.zeros(len(t), bool)
        for st, en in zip(pd.to_datetime(b["start"], utc=True), pd.to_datetime(b["end"], utc=True)):
            inb |= (t >= st) & (t < en)
            w = (t >= st) & (t < st + pd.Timedelta(minutes=a["noshow"]["after_start_min"]))
            if w.any() and occ[w].max() == 0:
                out.append({"space_id": s, "kind": "노쇼", "start": st, "end": en})
        run = 0
        for i in range(len(t)):
            run = run + 1 if (occ[i] == 1 and not inb[i]) else 0
            if run * 5 == a["unbooked"]["min_minutes"]:
                out.append({"space_id": s, "kind": "예약 없는 사용", "start": t[i - run + 1], "end": t[i]})
    return pd.DataFrame(out, columns=["space_id", "kind", "start", "end"])


def analyze_all(cfg, data, log=print) -> dict:
    from .pipeline import Data  # noqa: F401
    from .selection import require_selection
    root = data.root
    sel = require_selection(root / "selection.json", root / "runs")
    occ = pd.read_parquet(root / "eval" / "pred_occupancy.parquet")
    te = pd.read_parquet(root / "eval" / "pred_vent.parquet")
    days = (cfg.split.test[1] - cfg.split.test[0]).days + 1
    sch = occ[(occ["split"] == "test") & occ["y"].notna()]
    obs = occ[occ["obs_label"].notna()]
    res = {"transition_schedule": transition_errors(sch, "y", "pred").to_dict("records"),
           "transition_observation": transition_errors(obs, "obs_label", "pred").to_dict("records") if len(obs) else [],
           "people_recall": people_recall(obs, "pred").to_dict("records") if len(obs) else [],
           "top_confusions_schedule": top_confusions(sch, "y", "pred").to_dict("records"),
           "top_confusions_observation": top_confusions(obs, "obs_label", "pred").to_dict("records") if len(obs) else [],
           "vent_false_alarms": vent_false_alarms(te, days).to_dict("records"),
           "unseen_space": unseen_space(cfg, root, sel).to_dict("records") if cfg["advanced"].get("unseen_space") else []}
    ns = noshow_unbooked(occ, data.bookings, cfg)
    res["noshow_unbooked"] = {"source": "booking_file" if cfg["inputs"].get("booking_file") else
                              ("booking_estimated(순환 — 동작 예시만)" if data.bookings is not None else "없음"),
                              "counts": ns["kind"].value_counts().to_dict() if len(ns) else {}}
    if len(obs) and len(ns):
        # 관찰과 대조: 노쇼 구간 시작 15분 동안 관찰이 '비어있음'이면 맞힘
        hit = []
        for r in ns[ns["kind"] == "노쇼"].itertuples():
            o = obs[(obs["space_id"] == r.space_id) & (pd.to_datetime(obs["bin"], utc=True) >= r.start) &
                    (pd.to_datetime(obs["bin"], utc=True) < r.start + pd.Timedelta(minutes=15))]
            if len(o):
                hit.append(bool((o["obs_label"] == 0).all()))
        res["noshow_unbooked"]["observation_check"] = {"checked": len(hit), "correct": int(sum(hit))}
    (root / "eval" / "analysis.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    log(f"오류 분석 → {root / 'eval' / 'analysis.json'}")
    return res

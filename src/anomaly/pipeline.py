"""단계 실행: inject → train → score → threshold → evaluate. 결과 파일은 var/anomaly/ (data-model)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pipeline import sha256_file

from . import drift, rules
from .alarms import alarms_from, candidates, choose_threshold, combine, freeze_thresholds, require_thresholds, to_quantile
from .data import load_sensor, sensor_stats, space_cols, split_col, wide
from .evaluate import criteria, false_alarms, match, scorecard

ROOT = Path("var/anomaly")
THRESHOLDED = ("ae", "drift_peer", "forecast_err")        # 연속 점수 → 검증 기간으로 임계값
FIXED = ("rule_spike", "rule_stuck", "drift_night")        # 규칙(파라미터 고정, 검증 경보율만 보고)
GROUPS = {"model": ["ae"], "rules": ["rule_spike", "rule_stuck"], "combined": ["ae", "rule_spike", "rule_stuck"],
          "forecast_err": ["forecast_err"], "drift": ["drift_night", "drift_peer"],
          "combined_plus": ["ae", "rule_spike", "rule_stuck", "drift_night", "drift_peer"]}


class Work:
    """한 번 읽은 입력과 공간별 표를 들고 다닌다."""

    def __init__(self, cfg, root: Path = ROOT, sensor: pd.DataFrame | None = None):
        self.cfg, self.root = cfg, root
        self.sensor = load_sensor(cfg) if sensor is None else sensor
        self.cols = {s: space_cols(self.sensor, s, cfg) for s in cfg.spaces}
        self._w = {}

    def w(self, space: str, period: str) -> pd.DataFrame:
        key = (space, period)
        if key not in self._w:
            a, b = self.cfg.split.utc(period)
            pad = pd.Timedelta(hours=2) if period != "train" else pd.Timedelta(0)   # 창·이동 중앙값용 앞 여유
            self._w[key] = wide(self.sensor, space, self.cols[space], pd.Timestamp(a) - pad, b)
        return self._w[key]

    def period_slice(self, df, period, col="t"):
        a, b = self.cfg.split.utc(period)
        return df[(df[col] >= pd.Timestamp(a)) & (df[col] < pd.Timestamp(b))]

    def board(self, space: str) -> pd.DataFrame:
        b = pd.read_parquet(self.root / "board.parquet")
        b = b[b["space_id"] == space]
        w = b.pivot_table(index="bin", columns="col", values="value_injected", aggfunc="first")
        w.index = pd.DatetimeIndex(w.index).as_unit("ns")
        base = self.w(space, "test")
        out = base.copy()
        out.loc[w.index, w.columns] = w
        return out


# ---------- 1. 주입 ----------

def run_inject(work: Work, log=print) -> pd.DataFrame:
    from .inject import inject
    boards = {s: work.w(s, "test") for s in work.cfg.spaces}
    a, b = work.cfg.split.utc("test")
    core = {s: w[(w.index >= pd.Timestamp(a))] for s, w in boards.items()}
    out, faults, fid = inject(core, work.cfg)
    rows = []
    for s, w in out.items():
        long = w.stack(future_stack=True).rename("value_injected").reset_index()
        long.columns = ["bin", "col", "value_injected"]
        long["fault_id"] = fid[s].stack(future_stack=True).to_numpy()
        long = long[long["fault_id"] > 0]
        long["space_id"] = s
        rows.append(long)
    board = pd.concat(rows, ignore_index=True)
    work.root.mkdir(parents=True, exist_ok=True)
    faults.to_parquet(work.root / "faults.parquet", index=False)
    board.to_parquet(work.root / "board.parquet", index=False)
    meta = {"seed": work.cfg["inject"]["seed"], "config": work.cfg.digest,
            "faults_sha256": sha256_file(work.root / "faults.parquet"), "board_sha256": sha256_file(work.root / "board.parquet"),
            "changed_cells": int(len(board))}
    (work.root / "inject.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    log(pd.crosstab(faults["kind"], faults["space_id"]).to_string())
    log(f"고장 {len(faults)}건, 바뀐 칸 {len(board):,}, 정답 표 sha256 {meta['faults_sha256'][:16]}")
    return faults


# ---------- 2. 학습(오토인코더) ----------

def _exclude_mask(work, space, t: pd.DatetimeIndex) -> np.ndarray:
    m = np.zeros(len(t), bool)
    ev = pd.read_parquet(work.cfg["inputs"]["events"])
    ev = ev[(ev["space_id"] == space) & ev["kind"].isin(work.cfg["autoencoder"]["exclude_events"])]
    st = pd.read_parquet(work.cfg["inputs"]["stuck"])
    st = st[st["space_id"] == space]
    for a, b in list(zip(ev["start"], ev["end"])) + list(zip(st["start"], st["end"])):
        m |= (t >= a) & (t <= b)
    return m


def run_train(work: Work, device, log=print) -> dict:
    import torch
    from .autoencoder import column_errors, make_inputs, train_ae, window_starts
    a = work.cfg["autoencoder"]
    W = a["window"]
    (work.root / "models").mkdir(parents=True, exist_ok=True)
    recs = {}
    for space in work.cfg.spaces:
        tr, va = work.w(space, "train"), work.w(space, "val")
        st = sensor_stats(tr)
        st.to_parquet(work.root / f"stats_{space}.parquet")
        Xtr, Ztr = make_inputs(tr, st, work.cfg.acfg)
        Xva, Zva = make_inputs(va, st, work.cfg.acfg)
        ends_tr = window_starts(len(tr), W, _exclude_mask(work, space, tr.index))
        ends_va = window_starts(len(va), W)
        log(f"[{space}] 열 {tr.shape[1]}, 학습 창 {len(ends_tr):,}, 검증 창 {len(ends_va):,}")
        for seed in a["seeds"]:
            model, rec = train_ae(Xtr, Ztr, ends_tr, Xva, Zva, ends_va, work.cfg, device, seed, log)
            torch.save(model.state_dict(), work.root / "models" / f"{space}-s{seed}.pt")
            err = column_errors(model, Xva, Zva, W, a["score_tail"], device)
            va_core = work.period_slice(pd.DataFrame({"t": va.index}), "val").index
            norm = np.nanmedian(err[va_core], 0)
            norm = np.where(np.isnan(norm) | (norm <= 0), 1.0, norm)
            np.save(work.root / "models" / f"{space}-s{seed}-colnorm.npy", norm)
            rec.update(cols=list(tr.columns), device=str(device))
            recs[f"{space}-s{seed}"] = rec
    (work.root / "models" / "train.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1, default=str))
    return recs


# ---------- 3. 점수 ----------

def _wide_for(work, space, period):
    return work.w(space, "val") if period == "val" else work.board(space)


def score_ae(work, device, period) -> pd.DataFrame:
    import torch
    from .autoencoder import ConvAE, column_errors, make_inputs, space_score
    a = work.cfg["autoencoder"]
    out = []
    for space in work.cfg.spaces:
        st = pd.read_parquet(work.root / f"stats_{space}.parquet")
        w = _wide_for(work, space, period)
        X, Z = make_inputs(w, st, work.cfg.acfg)
        seed = a["seeds"][0]
        m = ConvAE(X.shape[1], Z.shape[1], a["window"], a["channels"], a["bottleneck"]).to(device)
        m.load_state_dict(torch.load(work.root / "models" / f"{space}-s{seed}.pt", map_location=device))
        norm = np.load(work.root / "models" / f"{space}-s{seed}-colnorm.npy")
        err = column_errors(m, X, Z, a["window"], a["score_tail"], device)
        out.append(space_score(err, norm, list(w.columns), w.index, space))
    return work.period_slice(pd.concat(out, ignore_index=True), period)


def score_rules(work, period) -> dict[str, pd.DataFrame]:
    sp, stk = [], []
    for space in work.cfg.spaces:
        st = sensor_stats(work.w(space, "train"))
        w = _wide_for(work, space, period)
        sp.append(rules.spike(w, st, work.cfg).assign(space_id=space))
        stk.append(rules.stuck(w, work.cfg, work.cfg.acfg).assign(space_id=space))
    return {"rule_spike": work.period_slice(pd.concat(sp, ignore_index=True), period),
            "rule_stuck": work.period_slice(pd.concat(stk, ignore_index=True), period)}


def score_drift(work, period) -> dict[str, pd.DataFrame]:
    nn, pp = [], []
    for space in work.cfg.spaces:
        tr = work.w(space, "train")
        w = _wide_for(work, space, period)
        nn.append(drift.night_alarms(w, drift.night_baseline(tr, work.cfg), work.cfg, space))
        pp.append(drift.peer_scores(w, drift.peer_stats(tr, work.cfg), work.cfg, space))
    return {"drift_night": work.period_slice(pd.concat(nn, ignore_index=True), period),
            "drift_peer": work.period_slice(pd.concat(pp, ignore_index=True), period)}


def run_score(work: Work, device, detectors=None, log=print):
    (work.root / "scores").mkdir(parents=True, exist_ok=True)
    detectors = detectors or ["ae", "rule_spike", "rule_stuck", "drift_night", "drift_peer"]
    for period in ("val", "test"):
        res = {}
        if "ae" in detectors:
            res["ae"] = score_ae(work, device, period)
        if {"rule_spike", "rule_stuck"} & set(detectors):
            res.update(score_rules(work, period))
        if {"drift_night", "drift_peer"} & set(detectors):
            res.update(score_drift(work, period))
        for d, df in res.items():
            if d not in detectors:
                continue
            df.to_parquet(work.root / "scores" / f"{d}__{period}.parquet", index=False)
            log(f"  점수 {d} {period}: {len(df):,}칸")
    if "forecast_err" in detectors:
        # 과제 3 모델 묶음 필요. 잔차는 검증 기간 MAD로만 정규화한다
        from .forecast_err import finalize_scores, score_forecast_err
        val, test = finalize_scores(score_forecast_err(work, device, "val"), score_forecast_err(work, device, "test"))
        for period, df in (("val", val), ("test", test)):
            df.to_parquet(work.root / "scores" / f"forecast_err__{period}.parquet", index=False)
            log(f"  점수 forecast_err {period}: {len(df):,}칸")


# ---------- 4. 임계값(검증 기간만) ----------

def _points(work, det, period, val_ref=None) -> pd.DataFrame:
    df = pd.read_parquet(work.root / "scores" / f"{det}__{period}.parquet")
    if det in FIXED:
        return df.assign(q=1.0)
    ref = val_ref if val_ref is not None else df
    parts = []
    for space, g in df.groupby("space_id"):
        parts.append(g.assign(q=to_quantile(g["score"], ref[ref["space_id"] == space]["score"]).to_numpy()))
    return pd.concat(parts, ignore_index=True) if parts else df.assign(q=np.nan)


def run_threshold(work: Work, log=print) -> dict:
    al = work.cfg["alarms"]
    days = work.cfg.val_days()
    body = {"target_per_day": al["target_per_day"], "val_days": days, "detectors": {}, "inputs_sha": {}}
    for det in THRESHOLDED + FIXED:
        p = work.root / "scores" / f"{det}__val.parquet"
        if not p.exists():
            continue
        body["inputs_sha"][det] = sha256_file(p)
        pts = _points(work, det, "val")
        if det in FIXED:
            n = len(alarms_from(pts, 0.0, al["merge_min"]))
            body["detectors"][det] = {"threshold_q": 0.0, "val_per_day": n / days, "reason": "규칙(파라미터 고정)"}
        else:
            pick, rows, why = choose_threshold(pts, candidates(work.cfg), al["target_per_day"], days, al["merge_min"])
            body["detectors"][det] = {"threshold_q": pick["threshold_q"], "val_per_day": pick["val_per_day"],
                                      "reason": why, "candidates": rows}
        d = body["detectors"][det]
        log(f"  {det}: 임계 분위수 {d['threshold_q']:.5f}, 검증 하루당 경보 {d['val_per_day']:.2f} — {d['reason']}")
    return freeze_thresholds(work.root / "thresholds.json", body)


# ---------- 5. 평가 ----------

def detector_alarms(work, det, thr_body, thr_override=None) -> pd.DataFrame:
    val = pd.read_parquet(work.root / "scores" / f"{det}__val.parquet") if det in THRESHOLDED else None
    pts = _points(work, det, "test", val_ref=val)
    thr = thr_override if thr_override is not None else thr_body["detectors"][det]["threshold_q"]
    return alarms_from(pts, thr, work.cfg["alarms"]["merge_min"])


def run_evaluate(work: Work, log=print) -> dict:
    thr = require_thresholds(work.root / "thresholds.json")
    faults = pd.read_parquet(work.root / "faults.parquet")
    ev = pd.read_parquet(work.cfg["inputs"]["events"])
    ev = ev[ev["kind"].isin(["B0", "B3"])]
    days = work.cfg.test_days()
    merge = work.cfg["alarms"]["merge_min"]
    have = {d for d in thr["detectors"]}
    base = {d: detector_alarms(work, d, thr) for d in have}
    (work.root / "alarms").mkdir(parents=True, exist_ok=True)
    res = {"thresholds_sha256": thr["sha256"], "test_days": days, "groups": {}}
    for g, dets in GROUPS.items():
        if not set(dets) <= have:
            continue
        al = combine([base[d] for d in dets], merge) if len(dets) > 1 else base[dets[0]]
        m = match(al, faults)
        fa = false_alarms(m, ev, days)
        card = scorecard(m, faults)
        m.drop(columns=[], errors="ignore").to_parquet(work.root / "alarms" / f"{g}.parquet", index=False)
        res["groups"][g] = {"scorecard": card.to_dict("records"),
                            "false": {k: v for k, v in fa.items() if k != "_rows"}}
        log(f"[{g}] 탐지율 {card.iloc[-1]['rate'] * 100:.1f}% ({card.iloc[-1]['detected']}/{card.iloc[-1]['n']}), "
            f"헛경보 {fa['per_day']:.2f}/일")
    if "combined" in res["groups"]:
        c = res["groups"]["combined"]
        crit = criteria(pd.DataFrame(c["scorecard"]), c["false"])
        res["criteria"] = crit.to_dict("records")
        log(crit.to_string(index=False))
    res["curve"] = threshold_curve(work, thr, faults, ev, days)
    (work.root / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return res


def threshold_curve(work, thr, faults, ev, days) -> list[dict]:
    """임계값 후보별 시험 탐지율·헛경보(산출물 2). 선택은 이미 고정돼 있다 — 보고용."""
    rows = []
    merge = work.cfg["alarms"]["merge_min"]
    for det in THRESHOLDED:
        if det not in thr["detectors"]:
            continue
        rule_al = [detector_alarms(work, d, thr) for d in ("rule_spike", "rule_stuck") if d in thr["detectors"]]
        for c in thr["detectors"][det]["candidates"]:
            al = detector_alarms(work, det, thr, c["threshold_q"])
            for name, a in ((det, al), (f"{det}+rules", combine([al] + rule_al, merge))):
                m = match(a, faults)
                card = scorecard(m, faults)
                fa = false_alarms(m, None, days)
                rows.append({"detector": name, "threshold_q": c["threshold_q"], "val_per_day": c["val_per_day"],
                             "test_rate": float(card.iloc[-1]["rate"]), "test_false_per_day": fa["per_day"],
                             "delay_median": float(card.iloc[-1]["delay_median_min"]),
                             "chosen": c["threshold_q"] == thr["detectors"][det]["threshold_q"]})
    return rows

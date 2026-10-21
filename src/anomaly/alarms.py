"""점수 → 분위수 → 경보(병합), 검증 기간으로 임계값 선택·고정, 경보 결합 (research R5·R6)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ALARM_COLS = ["space_id", "start", "end", "peak_q", "cause"]


class ThresholdMissingError(RuntimeError):
    pass


def to_quantile(score: pd.Series, val_score: pd.Series) -> pd.Series:
    """검증 기간 점수 분포 대비 분위수(0~1). 검증 최댓값보다 크면 1."""
    ref = np.sort(val_score.dropna().to_numpy())
    if len(ref) == 0:
        return pd.Series(np.nan, index=score.index)
    return pd.Series(np.searchsorted(ref, score.to_numpy(), side="right") / len(ref), index=score.index)


def alarms_from(points: pd.DataFrame, thr: float, merge_min: int) -> pd.DataFrame:
    """points: space_id, t, q, cause. q ≥ thr인 칸을 공간별로 묶는다(merge_min 안이면 한 경보)."""
    p = points[points["q"] >= thr].sort_values(["space_id", "t"])
    out = []
    gap = pd.Timedelta(minutes=merge_min)
    for space, g in p.groupby("space_id"):
        cur = None
        for r in g.itertuples(index=False):
            if cur is not None and r.t - cur["end"] <= gap:
                cur["end"] = r.t
                if r.q > cur["peak_q"]:
                    cur["peak_q"], cur["cause"] = r.q, r.cause
            else:
                if cur is not None:
                    out.append(cur)
                cur = {"space_id": space, "start": r.t, "end": r.t, "peak_q": r.q, "cause": r.cause}
        if cur is not None:
            out.append(cur)
    return pd.DataFrame(out, columns=ALARM_COLS)


def combine(alarm_sets: list[pd.DataFrame], merge_min: int) -> pd.DataFrame:
    """경보 합집합 → 다시 병합. 원인은 최고 q 쪽, 같으면 먼저 온 쪽."""
    a = pd.concat([x for x in alarm_sets if len(x)], ignore_index=True) if any(len(x) for x in alarm_sets) else \
        pd.DataFrame(columns=ALARM_COLS)
    if a.empty:
        return a
    pts = pd.concat([a[["space_id", "start", "peak_q", "cause"]].rename(columns={"start": "t", "peak_q": "q"}),
                     a[["space_id", "end", "peak_q", "cause"]].rename(columns={"end": "t", "peak_q": "q"})])
    return alarms_from(pts, -np.inf, merge_min)


def candidates(cfg) -> np.ndarray:
    c = cfg["alarms"]["q_candidates"]
    return 1 - np.logspace(np.log10(1 - c["min"]), np.log10(1 - c["max"]), c["n"])


def choose_threshold(val_points: pd.DataFrame, cands, target_per_day: float, days: int, merge_min: int):
    """검증 기간 하루당 경보(전체 공간 합)가 목표 이하인 가장 낮은 임계값. 시험 기간 정보는 입력에 없다."""
    rows = []
    for thr in sorted(cands):
        n = len(alarms_from(val_points, thr, merge_min))
        rows.append({"threshold_q": float(thr), "val_alarms": n, "val_per_day": n / days})
    ok = [r for r in rows if r["val_per_day"] <= target_per_day]
    pick = ok[0] if ok else rows[-1]
    why = (f"검증 하루당 경보 {pick['val_per_day']:.2f} ≤ 목표 {target_per_day}인 최저 임계값" if ok
           else f"목표 {target_per_day}을 만족하는 후보가 없어 가장 높은 임계값(검증 {pick['val_per_day']:.2f}/일)")
    return pick, rows, why


def _digest(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def freeze_thresholds(path: Path, body: dict) -> dict:
    body = dict(body, frozen_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    body["sha256"] = _digest(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=str))
    return body


def require_thresholds(path: Path) -> dict:
    if not path.exists():
        raise ThresholdMissingError("thresholds.json 없음 — 검증 기간으로 임계값을 먼저 고정(`anomaly threshold`)")
    body = json.loads(path.read_text())
    sha = body.pop("sha256", None)
    if _digest(body) != sha:
        raise ThresholdMissingError("thresholds.json 해시 불일치(고정 뒤 변경됨)")
    body["sha256"] = sha
    return body

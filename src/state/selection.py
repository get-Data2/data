"""지표·임계값·검증 기준 선택과 고정 (research R6~R8). selection.json이 고정된 뒤에만 시험·관찰 평가가 동작한다."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class SelectionMissingError(RuntimeError):
    pass


def confusion(y, pred) -> dict:
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    return {"tp": int(((y == 1) & (pred == 1)).sum()), "fp": int(((y == 0) & (pred == 1)).sum()),
            "fn": int(((y == 1) & (pred == 0)).sum()), "tn": int(((y == 0) & (pred == 0)).sum())}


def metrics(y, pred) -> dict:
    c = confusion(y, pred)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n = tp + fp + fn + tn
    rec = tp / (tp + fn) if tp + fn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (tp and (prec + rec)) else 0.0
    rec0 = tn / (tn + fp) if tn + fp else np.nan
    prec0 = tn / (tn + fn) if tn + fn else np.nan
    f10 = 2 * prec0 * rec0 / (prec0 + rec0) if (tn and (prec0 + rec0)) else 0.0
    return {**c, "n": n, "accuracy": (tp + tn) / n if n else np.nan, "recall": rec, "precision": prec, "f1": f1,
            "macro_f1": (f1 + f10) / 2, "false_occupied_rate": fp / (fp + tn) if fp + tn else np.nan,
            "always_negative_accuracy": (tn + fp) / n if n else np.nan}


def vent_threshold(y, p, min_recall: float) -> tuple[float, bool, dict]:
    """재현율 ≥ min_recall인 임계값 중 정밀도 최대(동률이면 높은 임계값). 없으면 재현율 최대 값과 미달 표시."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    cands = np.unique(np.concatenate([[0.0], np.quantile(p, np.linspace(0, 1, 401)), [1.0 + 1e-9]]))
    best, best_m = None, None
    for thr in cands:
        m = metrics(y, p >= thr)
        if not np.isnan(m["recall"]) and m["recall"] >= min_recall:
            prec = 0.0 if np.isnan(m["precision"]) else m["precision"]
            if best is None or prec > best_m["_prec"] or (prec == best_m["_prec"] and thr > best):
                best, best_m = float(thr), dict(m, _prec=prec)
    if best is not None:
        best_m.pop("_prec")
        return best, True, best_m
    rec = [(metrics(y, p >= t)["recall"], t) for t in cands]
    t = max(rec, key=lambda r: (r[0], -r[1]))[1]
    return float(t), False, metrics(y, p >= t)


def occ_threshold(y, p) -> tuple[float, dict]:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    best, bm = 0.5, None
    for thr in np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99))):
        m = metrics(y, p >= thr)
        if bm is None or m["macro_f1"] > bm["macro_f1"]:
            best, bm = float(thr), m
    return best, bm


def pick(stats: list[dict]) -> tuple[int, str]:
    """stats[i] = {mean, sd, simplicity}. 검증 지표 최대, 단 1표준편차 안이면 단순한 쪽(logreg < hgb < mlp)."""
    best = max(range(len(stats)), key=lambda i: stats[i]["mean"])
    lim = stats[best]["mean"] - stats[best]["sd"]
    near = [i for i in range(len(stats)) if stats[i]["mean"] >= lim]
    ch = min(near, key=lambda i: (stats[i]["simplicity"], -stats[i]["mean"]))
    return ch, ("검증 지표 최대" if ch == best else "최대와 1표준편차 안 → 더 단순한 모델")


def _digest(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def freeze_selection(path: Path, body: dict) -> dict:
    body = dict(body, frozen_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    body["sha256"] = _digest(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=str))
    return body


def require_selection(path: Path, runs_root: Path) -> dict:
    if not path.exists():
        raise SelectionMissingError("selection.json 없음 — 검증 기준 선택(`state select`)을 먼저 고정해야 평가 가능")
    body = json.loads(path.read_text())
    sha = body.pop("sha256", None)
    if _digest(body) != sha:
        raise SelectionMissingError("selection.json 해시 불일치(고정 뒤 변경됨)")
    for prob in ("occupancy", "vent", "leak"):
        for rid, h in (body.get(prob) or {}).get("model_sha256", {}).items():
            f = runs_root / prob / rid / "model.joblib"
            if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()[:16] != h:
                raise SelectionMissingError(f"모델이 선택 고정 때와 다름: {prob}/{rid}")
    body["sha256"] = sha
    return body

"""크기 × 컨텍스트 선택(검증), 앙상블 가중치(검증), 선택 고정·검사 (research R5·R10).

선택은 검증 기간 예측만 본다. selection.json이 고정된 뒤에만 시험 평가를 허용한다(과제 3 require_selection과 같은 방식).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT
from .metrics import norm_mae


class SelectionMissingError(RuntimeError):
    pass


def choose(cfg, grid_rows: list[dict]) -> tuple[dict, str]:
    """grid_rows: {model, ctx_min, val_norm_mae}. 최소의 tie_rel 안이면 작은 모델 → 짧은 컨텍스트."""
    ok = [r for r in grid_rows if not np.isnan(r["val_norm_mae"])]
    if not ok:
        raise SelectionMissingError("검증 예측이 없음")
    best = min(ok, key=lambda r: r["val_norm_mae"])
    lim = best["val_norm_mae"] * (1 + cfg.raw.get("tie_rel", 0.01))
    near = [r for r in ok if r["val_norm_mae"] <= lim]
    pick = min(near, key=lambda r: (cfg.size_rank(r["model"]), r["ctx_min"]))
    why = "검증 오차 최소" if pick is best else f"최소({best['model']} ctx{best['ctx_min']})의 {cfg.raw.get('tie_rel', 0.01) * 100:.0f}% 안 → 작은 모델·짧은 컨텍스트"
    return pick, why


def ensemble_weights(cfg, direct_val: pd.DataFrame, pre_val: pd.DataFrame) -> dict:
    """ŷ = w·직접 + (1−w)·사전학습. 하나의 w / 공간 × 지평별 w를 검증 MAE로 비교해 고른다."""
    k = ["space_id", "t0", "horizon_min"]
    m = direct_val[k + ["y_pred"]].rename(columns={"y_pred": "d"}).merge(
        pre_val[k + ["y_pred", "y_true"]].rename(columns={"y_pred": "p"}), on=k)
    g = cfg["ensemble"]["w_grid"]
    ws = np.round(np.arange(g["min"], g["max"] + 1e-9, g["step"]), 3)

    def mae(d, w):
        return float((w * d["d"] + (1 - w) * d["p"] - d["y_true"]).abs().mean())
    glob = {float(w): mae(m, w) for w in ws}
    w_g = min(glob, key=glob.get)
    per = {}
    for (s, h), d in m.groupby(["space_id", "horizon_min"]):
        sc = {float(w): mae(d, w) for w in ws}
        per[f"{s}/{h}"] = min(sc, key=sc.get)
    blend = m.copy()
    blend["w"] = [per[f"{s}/{h}"] for s, h in zip(blend["space_id"], blend["horizon_min"])]
    per_mae = float((blend["w"] * blend["d"] + (1 - blend["w"]) * blend["p"] - blend["y_true"]).abs().mean())
    mode = "per_cell" if per_mae < glob[w_g] else "global"
    return {"mode": mode, "w_global": w_g, "w_per_cell": per, "val_mae_global": glob[w_g], "val_mae_per_cell": per_mae,
            "val_rows": int(len(m))}


def apply_ensemble(ens: dict, direct: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    k = ["space_id", "item", "t0", "horizon_min"]
    m = pre.merge(direct[k + ["y_pred"]].rename(columns={"y_pred": "d"}), on=k, how="left", validate="one_to_one")
    w = (np.full(len(m), ens["w_global"]) if ens["mode"] == "global"
         else np.array([ens["w_per_cell"].get(f"{s}/{h}", ens["w_global"]) for s, h in zip(m["space_id"], m["horizon_min"])]))
    m["y_pred"] = np.where(m["d"].notna(), w * m["d"] + (1 - w) * m["y_pred"], m["y_pred"])
    return m.drop(columns=["d"])


def _digest(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def freeze(body: dict, root: Path = ROOT) -> dict:
    body = dict(body, frozen_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    body["sha256"] = _digest(body)
    root.mkdir(parents=True, exist_ok=True)
    (root / "selection.json").write_text(json.dumps(body, ensure_ascii=False, indent=1, default=str))
    return body


def require(root: Path = ROOT) -> dict:
    p = root / "selection.json"
    if not p.exists():
        raise SelectionMissingError("selection.json 없음 — 검증 기준 선택(`pretrained select`)을 먼저 고정해야 시험 평가 가능")
    body = json.loads(p.read_text())
    sha = body.pop("sha256", None)
    if _digest(body) != sha:
        raise SelectionMissingError("selection.json 해시 불일치(고정 뒤 변경됨)")
    body["sha256"] = sha
    return body


def grid_rows(cfg, val_preds: dict[tuple[str, int], pd.DataFrame], norm: dict) -> list[dict]:
    return [{"model": m, "ctx_min": c, "val_norm_mae": norm_mae(cfg, df, norm),
             "val_lab_60": float((lambda g: (g["y_pred"] - g["y_true"]).abs().mean())(
                 df[(df["space_id"] == "lab") & (df["horizon_min"] == 60)])) if len(df) else float("nan")}
            for (m, c), df in val_preds.items()]

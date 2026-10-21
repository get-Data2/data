"""학습 루프: 관측 칸만의 L1, 검증 손실로 학습률 감소·조기 종료, 최고 가중치 복원 (research R8)."""
from __future__ import annotations

import hashlib
import io
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .env import env_info, peak_memory, set_determinism
from .models import ForecastModel, count_params

RUNS = Path("var/forecast/runs")
REPORT_H = (1, 3, 6, 12)            # 5·15·30·60분
ITEM_NAMES = ("co2", "temperature")
MIN_VAL_POINTS = 100


def run_dir(exp, seed: int, root: Path = RUNS) -> Path:
    return root / f"{exp.hash}-s{seed}"


def masked_l1(pred, target, mask):
    m = mask.float()
    return (torch.abs(pred - target) * m).sum() / m.sum().clamp_min(1)


def _batches(idx: np.ndarray, size: int, rng: np.random.Generator | None = None):
    order = rng.permutation(len(idx)) if rng is not None else np.arange(len(idx))
    for i in range(0, len(idx), size):
        yield idx[order[i:i + size]]


@torch.no_grad()
def predict(model, W, idx: np.ndarray, batch: int = 2048):
    """원 단위 예측 (n, H, 2)과 정답(n, H, 2, NaN)."""
    model.eval()
    preds, ys = [], []
    for bi in _batches(idx, batch):
        b = W.batch(bi)
        p = W.to_original(model(b), b)
        y = torch.where(b["mask"], W.Y[b["s"][:, None], b["t"][:, None] + torch.arange(1, W.H + 1, device=W.device)],
                        torch.full_like(p, float("nan")))
        preds.append(p.cpu().numpy())
        ys.append(y.cpu().numpy())
    if not preds:
        return np.zeros((0, W.H, 2)), np.zeros((0, W.H, 2))
    return np.concatenate(preds), np.concatenate(ys)


@torch.no_grad()
def eval_loss(model, W, idx, batch: int = 2048) -> float:
    model.eval()
    tot, n = 0.0, 0
    for bi in _batches(idx, batch):
        b = W.batch(bi)
        m = b["mask"].float()
        tot += float((torch.abs(model(b) - b["target"]) * m).sum())
        n += float(m.sum())
    return tot / max(n, 1)


def val_metrics(pred, y, idx, spaces, sd: np.ndarray) -> dict:
    """공간 × 항목 × 지평(5·15·30·60) MAE(원 단위)와 정규화 MAE 평균(선택 지표)."""
    cells, norm = {}, []
    for si, space in enumerate(spaces):
        m = idx[:, 0] == si
        for ii, item in enumerate(ITEM_NAMES):
            for h in REPORT_H:
                e = np.abs(pred[m, h - 1, ii] - y[m, h - 1, ii])
                e = e[~np.isnan(e)]
                key = f"{space}/{item}/{h * 5}"
                if len(e) >= MIN_VAL_POINTS:
                    cells[key] = {"mae": round(float(e.mean()), 4), "n": int(len(e))}
                    norm.append(float(e.mean()) / float(sd[si, ii]))
    return {"cells": cells, "val_norm_mae": round(float(np.mean(norm)), 6) if norm else math.nan}


def train_one(exp, seed: int, W, train_idx: np.ndarray, val_idx: np.ndarray, device, spaces, meta: dict,
              out: Path | None = None, log=print) -> dict:
    set_determinism(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    model = ForecastModel(exp, W.fp, W.ff, W.n_space, W.H).to(device)
    tr = exp.train
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=tr["patience_lr"])
    rng = np.random.default_rng(seed)
    best, best_ep, best_state, history = math.inf, -1, None, []
    status, error = "done", None
    for ep in range(tr["max_epochs"]):
        model.train()
        tl, tn = 0.0, 0
        for bi in _batches(train_idx, tr["batch"], rng):
            b = W.batch(bi)
            loss = masked_l1(model(b), b["target"], b["mask"])
            if not torch.isfinite(loss):
                status, error = "failed", f"손실 발산(에폭 {ep})"
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += float(loss.detach()) * len(bi)
            tn += len(bi)
        if status == "failed":
            break
        vl = eval_loss(model, W, val_idx)
        sched.step(vl)
        history.append({"epoch": ep, "train": round(tl / max(tn, 1), 6), "val": round(vl, 6),
                        "lr": opt.param_groups[0]["lr"]})
        if vl < best - 1e-6:
            best, best_ep = vl, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif ep - best_ep >= tr["patience_stop"]:
            break
    rec = {"label": exp.label(), "config_hash": exp.hash, "seed": seed, "status": status, "error": error,
           "config": exp.key(), **env_info(device), **meta, "params": count_params(model),
           "epochs": len(history), "best_epoch": best_ep, "history": history,
           "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if best_state is not None:
        model.load_state_dict(best_state)
        pred, y = predict(model, W, val_idx)
        rec["val"] = val_metrics(pred, y, val_idx, spaces, W.sd.cpu().numpy())
    rec["train_seconds"] = round(time.time() - t_start, 1)
    rec.update(peak_memory(device))
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        if best_state is not None:
            buf = io.BytesIO()
            torch.save(model.state_dict(), buf)
            (out / "model.pt").write_bytes(buf.getvalue())
            rec["model_sha256"] = hashlib.sha256(buf.getvalue()).hexdigest()[:16]
            _save_val(out, pred, y, val_idx, W, spaces)
        (out / "run.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str))
    v = rec.get("val", {}).get("val_norm_mae")
    log(f"  {exp.label()} s{seed}: {status} 에폭 {len(history)}(최고 {best_ep}) 검증 {v} {rec['train_seconds']}초")
    rec["_model"] = model
    return rec


def _save_val(out: Path, pred, y, idx, W, spaces):
    rows = []
    t = W._t_index
    for h in REPORT_H:
        for ii, item in enumerate(ITEM_NAMES):
            rows.append(pd.DataFrame({"space_id": np.array(spaces)[idx[:, 0]], "t0": t[idx[:, 1]], "item": item,
                                      "horizon_min": h * 5, "y_true": y[:, h - 1, ii], "y_pred": pred[:, h - 1, ii]}))
    pd.concat(rows, ignore_index=True).dropna(subset=["y_true"]).to_parquet(out / "val_pred.parquet", index=False)

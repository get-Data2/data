"""모델 묶음: 과제 7(최적화)·8(서비스)이 읽을 최종 모델·설정·정규화·입력 규약 (data-model "모델 묶음")."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

from .data import ITEMS, norm_table
from .experiments import exp_from_key
from .features import BOOK_COLS, SCHED_COLS, TIME_COLS

INPUT_SPEC_VERSION = 1


def input_spec(exp) -> dict:
    cov = [*(TIME_COLS if "time" in exp.features else []), *(SCHED_COLS if "schedule" in exp.features else []),
           *(BOOK_COLS if "booking" in exp.features else [])]
    past = [f"x_{i}" for i in ITEMS] + [f"obs_{i}" for i in ITEMS] + [f"age_{i}" for i in ITEMS] + cov
    return {"version": INPUT_SPEC_VERSION, "step_min": 5, "lookback_bins": exp.lookback_min // 5, "horizon_bins": 12,
            "past_columns": past, "future_columns": cov, "space_onehot": "space" in exp.features,
            "output": "(12, 2) [co2, temperature]", "target": exp.target,
            "notes": "x_* = (앞으로 채운 관측값 − 평균)/표준편차, age_* = 마지막 관측 뒤 시간/age_cap_h"}


def _test_mae(ctx, sel) -> dict:
    """서비스 자기 성능 감시 기준: 최종 모델 시험 MAE(씨앗 평균) {"공간/항목/지평": MAE}."""
    from .evaluate import load_test_preds, mae_by_cell
    df = load_test_preds(ctx, sel, "final").get(sel["final_label"])
    if df is None:
        return {}
    t = mae_by_cell(df, "all", 0)
    return {f"{r.space_id}/{r.item}/{r.horizon_min}": round(float(r.mae), 4) for r in t.itertuples() if r.mae == r.mae}


def save_bundle(ctx, sel, log=print):
    out = ctx.root / "bundle"
    out.mkdir(parents=True, exist_ok=True)
    finals = [(rid, r) for rid, r in sel["runs"].items() if r["group"] == "final"]
    best = min(finals, key=lambda x: json.loads((ctx.runs / x[0] / "run.json").read_text())["val"]["val_norm_mae"])
    rid, r = best
    exp = exp_from_key(r["key"], [r["seed"]])
    shutil.copy(ctx.runs / rid / "model.pt", out / "model.pt")
    nt = norm_table(ctx.inputs["norm"], tuple(ctx.cfg["spaces"]), exp.norm, exp.train_spaces)
    run = json.loads((ctx.runs / rid / "run.json").read_text())
    body = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run": rid, "label": exp.label(),
            "config": exp.key(), "seed": r["seed"], "spaces": list(ctx.cfg["spaces"]),
            "norm": {s: {it: {"mean": float(nt[si, ii, 0]), "std": float(nt[si, ii, 1])} for ii, it in enumerate(ITEMS)}
                     for si, s in enumerate(ctx.cfg["spaces"])},
            "age_cap_h": ctx.cfg["age_cap_h"], "input_spec": input_spec(exp), "model_sha256": run["model_sha256"],
            "data": {k: run.get(k) for k in ("split", "eval_sha256", "series_sha16", "norm_sha16")},
            "selection_sha256": sel["sha256"], "schedule_source": "config/analysis.yaml schedule",
            "config_hash": exp.hash, "test_mae": _test_mae(ctx, sel)}
    (out / "bundle.json").write_text(json.dumps(body, ensure_ascii=False, indent=1))
    log(f"모델 묶음 → {out} ({exp.label()} s{r['seed']})")
    return out

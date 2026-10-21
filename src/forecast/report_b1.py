"""모델 × 과거 길이(B1) 9조합의 시험 성적 — 보고용(선택은 이미 검증으로 고정).

`forecast test`는 선택 파일에 적힌 실행(최종·비교 쌍·심화)만 시험한다. 산출물 ① 모델 비교표에 세 모델을 모두 싣기 위해
B1 후보의 씨앗별 모델로 시험 예측을 더 낸다. 선택 고정 파일이 없으면 거부한다(누수 게이트 그대로).
실행: `.venv/bin/python -m forecast.report_b1` (GPU 그룹 필요 시 tools/forecast와 같은 sg 래퍼 아래에서)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from .config import load_forecast_config
from .env import pick_device
from .evaluate import mae_by_cell, predict_eval_points
from .experiments import Context, exp_from_key, require_selection

OUT = "b1_test.parquet"


def run(log=print) -> pd.DataFrame:
    cfg = load_forecast_config()
    ctx = Context.load(cfg, pick_device("auto"))
    sel = require_selection(ctx)
    b1 = json.loads((ctx.root / "stages.json").read_text())["B1"]["candidates"]
    rows = []
    for c in b1:
        preds = []
        for seed in (0, 1, 2):
            rid = f"{c['hash']}-s{seed}"
            d = ctx.runs / rid
            p = d / "test_pred.parquet"
            if not (d / "model.pt").exists():
                log(f"  {c['label']} s{seed}: 모델 없음 — 건너뜀")
                continue
            if not p.exists():
                exp = exp_from_key(c["key"], [seed])
                df = predict_eval_points(ctx, exp, seed)
                df.to_parquet(p, index=False)
                rec = json.loads((d / "run.json").read_text())
                rec |= {"test_evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "selection_sha256": sel["sha256"], "test_reason": "보고용(B1 비교표), 선택 고정 뒤"}
                (d / "run.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str))
                log(f"  시험 예측: {c['label']} s{seed} — {len(df):,}행")
            preds.append(pd.read_parquet(p))
        if not preds:
            continue
        m = mae_by_cell(pd.concat(preds, ignore_index=True))
        mu = mae_by_cell(pd.concat(preds, ignore_index=True), "usage")
        m = m.merge(mu[["space_id", "item", "horizon_min", "mae"]].rename(columns={"mae": "mae_usage"}),
                    on=["space_id", "item", "horizon_min"], how="left")
        rows.append(m.assign(label=c["label"], model=c["key"]["model"], lookback_min=c["key"]["lookback_min"],
                             val_mean=c["mean"], val_sd=c["sd"]))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(ctx.root / OUT, index=False)
    log(f"→ {ctx.root / OUT}")
    return out


if __name__ == "__main__":
    run()

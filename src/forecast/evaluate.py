"""시험 평가: 평가 시점 파일 행만 예측·채점, 기준선과 결합, 비교표·확인 기준 판정 (FR-002·030·031)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from analysis.baseline import METHODS, SPACE_NAME

from .data import eval_samples
from .experiments import exp_from_key, require_selection
from .models import ForecastModel
from .train import predict, run_dir

ITEMS = ("co2", "temperature")
HORIZONS = (5, 15, 30, 60)


def load_model(ctx, exp, seed, W):
    m = ForecastModel(exp, W.fp, W.ff, W.n_space, W.H).to(ctx.device)
    m.load_state_dict(torch.load(run_dir(exp, seed, ctx.runs) / "model.pt", map_location=ctx.device))
    m.eval()
    return m


def predict_eval_points(ctx, exp, seed, W=None) -> pd.DataFrame:
    """평가 시점 파일의 모든 행에 예측을 하나씩 붙인다. 입력이 부족한 표본은 현재값 유지로 대체(unpredictable)."""
    W = W or ctx.windows(exp)
    ev = ctx.inputs["eval"]
    idx, unpred, keys = eval_samples(ctx.arrays, ev, ctx.cfg, W.L)
    model = load_model(ctx, exp, seed, W)
    ok = ~unpred
    pred = np.full((len(idx), W.H, 2), np.nan, np.float32)
    if ok.any():
        p, _ = predict(model, W, idx[ok])
        pred[ok] = p
    rows = []
    for ii, item in enumerate(ITEMS):
        for h in sorted(ev["horizon_min"].unique()):
            rows.append(pd.DataFrame({"space_id": keys["space_id"], "t0": keys["t0"], "item": item,
                                      "horizon_min": h, "y_pred": pred[:, h // 5 - 1, ii], "unpredictable": unpred}))
    p = pd.concat(rows, ignore_index=True)
    out = ev.merge(p, on=["space_id", "item", "t0", "horizon_min"], how="left", validate="one_to_one")
    if len(out) != len(ev) or out["y_pred"].isna().sum() != out["unpredictable"].sum():
        raise RuntimeError("평가 시점 행과 예측이 일치하지 않음")
    bl = pd.read_parquet(ctx.cfg["inputs"]["baseline"])
    out = out.merge(bl[["space_id", "item", "t0", "horizon_min", "persistence"]],
                    on=["space_id", "item", "t0", "horizon_min"], how="left", validate="one_to_one")
    out["y_pred"] = out["y_pred"].fillna(out["persistence"])
    out["seed"] = seed
    return out.drop(columns=["persistence"])


def run_test(ctx, log=print) -> dict:
    sel = require_selection(ctx)
    done = {}
    for rid, r in sel["runs"].items():
        exp = exp_from_key(r["key"], [r["seed"]])
        out = ctx.runs / rid
        p = out / "test_pred.parquet"
        if not p.exists():
            df = predict_eval_points(ctx, exp, r["seed"])
            df.to_parquet(p, index=False)
            rec = json.loads((out / "run.json").read_text())
            rec["test_evaluated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            rec["selection_sha256"] = sel["sha256"]
            (out / "run.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str))
            log(f"  시험 예측: {r['group']} {r['label']} s{r['seed']} — {len(df):,}행, 대체 {int(df['unpredictable'].sum())}")
        done[rid] = r
    return sel


def load_test_preds(ctx, sel, group: str) -> dict[str, pd.DataFrame]:
    """그룹의 실험별(라벨) 씨앗 예측을 모은다."""
    out = {}
    for rid, r in sel["runs"].items():
        if r["group"] != group:
            continue
        df = pd.read_parquet(ctx.runs / rid / "test_pred.parquet")
        out.setdefault(r["label"], []).append(df)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def mae_by_cell(df: pd.DataFrame, subset="all", min_points=1000) -> pd.DataFrame:
    """씨앗별 MAE → 공간 × 항목 × 지평의 평균·표준편차."""
    d = df if subset == "all" else df[df["usage_hours"]]
    g = d.assign(e=(d["y_pred"] - d["y_true"]).abs()).groupby(["space_id", "item", "horizon_min", "seed"])
    m = g["e"].agg(["mean", "size"]).reset_index()
    s = m.groupby(["space_id", "item", "horizon_min"]).agg(mae=("mean", "mean"), sd=("mean", "std"),
                                                           n=("size", "first")).reset_index()
    s.loc[s["n"] < min_points, ["mae", "sd"]] = np.nan
    return s


def baseline_by_cell(bl: pd.DataFrame, subset="all", min_points=1000) -> pd.DataFrame:
    d = bl if subset == "all" else bl[bl["usage_hours"]]
    rows = []
    for (s, i, h), g in d.groupby(["space_id", "item", "horizon_min"]):
        r = {"space_id": s, "item": i, "horizon_min": h, "n": len(g)}
        for m in METHODS:
            r[m] = float((g[m] - g["y_true"]).abs().mean()) if len(g) >= min_points else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def comparison_table(ctx, model_df: pd.DataFrame, subset="all") -> pd.DataFrame:
    mp = ctx.cfg.acfg["baseline"].get("min_points", 1000)
    bl = baseline_by_cell(pd.read_parquet(ctx.cfg["inputs"]["baseline"]), subset, mp)
    m = mae_by_cell(model_df, subset, mp)
    t = bl.merge(m[["space_id", "item", "horizon_min", "mae", "sd"]], on=["space_id", "item", "horizon_min"], how="left")
    t["개선율(%)"] = (1 - t["mae"] / t["persistence"]) * 100
    t = t[t["horizon_min"].isin(HORIZONS)]
    order = {s: i for i, s in enumerate(ctx.cfg["spaces"])}
    return t.sort_values(["space_id", "item", "horizon_min"], key=lambda c: c.map(order) if c.name == "space_id" else c)


def criteria(ctx, model_df: pd.DataFrame) -> pd.DataFrame:
    """확인 기준 1·2: CO₂ 60분 30% 이상 개선, 온도 60분 개선(공간별)."""
    rows = []
    for subset in ("all", "usage"):
        t = comparison_table(ctx, model_df, subset)
        for r in t[t["horizon_min"] == 60].itertuples():
            if np.isnan(r.mae):
                continue
            goal = r.persistence * (0.7 if r.item == "co2" else 1.0)
            rows.append({"기준": "CO₂ 60분 30% 개선" if r.item == "co2" else "온도 60분 개선",
                         "공간": SPACE_NAME.get(r.space_id, r.space_id), "구간": "전체" if subset == "all" else "사용 시간대",
                         "현재값 유지": round(r.persistence, 2), "목표": f"≤ {goal:.2f}" if r.item == "co2" else f"< {goal:.2f}",
                         "모델": f"{r.mae:.2f} ± {0 if np.isnan(r.sd) else r.sd:.2f}", "개선율(%)": round((1 - r.mae / r.persistence) * 100, 1),
                         "판정": "✅" if (r.mae <= goal if r.item == "co2" else r.mae < goal) else "❌"})
    return pd.DataFrame(rows)

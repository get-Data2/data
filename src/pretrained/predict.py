"""기간별 예측 → parquet(키·p10·p50·p90·y_pred·unpredictable·y_true·usage_hours). 이미 있으면 건너뛴다 (data-model "예측")."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT
from .context import contexts, grid, eval_keys, val_t0, val_targets

QCOLS = ("p10", "p50", "p90")


def pred_path(name: str, ctx_min: int, period: str, cov: bool = False, root: Path = ROOT) -> Path:
    return root / "preds" / f"{name}__ctx{ctx_min}__{period}{'__cov' if cov else ''}.parquet"


def _covariates(cfg, series, space, g_index, t0s, L, H):
    """과거 공변량(온도·습도, 관측만 NaN) + 미리 아는 공변량(시간·일과표, 과거 창과 미래 12칸)."""
    from forecast.features import SCHED_COLS, TIME_COLS, schedule_features, time_features
    idx = g_index.get_indexer(pd.DatetimeIndex(t0s).as_unit("ns"))
    past_win = idx[:, None] + np.arange(-L + 1, 1)[None]
    fut_win = idx[:, None] + np.arange(1, H + 1)[None]
    past, fut = {}, {}
    for item in cfg["covariates"]["past"]:
        v = grid(series, space, item, g_index[0], g_index[-1]).reindex(g_index).to_numpy(np.float32)
        past[item] = np.where(past_win >= 0, v[np.clip(past_win, 0, None)], np.nan)
    t_all = g_index.append(pd.date_range(g_index[-1] + pd.Timedelta(minutes=5), periods=H, freq="5min").as_unit("ns"))
    known = {}
    if "time" in cfg["covariates"]["known"]:
        tf = time_features(t_all, cfg.acfg)
        known |= {c: tf[:, i] for i, c in enumerate(TIME_COLS)}
    if "schedule" in cfg["covariates"]["known"]:
        sf = schedule_features(t_all, space, cfg.acfg)
        known |= {c: sf[:, i] for i, c in enumerate(SCHED_COLS)}
    for k, v in known.items():
        past[k] = np.where(past_win >= 0, v[np.clip(past_win, 0, None)], np.nan).astype(np.float32)
        fut[k] = v[fut_win].astype(np.float32)
    return past, fut


def predict_period(cfg, predictor, name: str, ctx_min: int, period: str, series: pd.DataFrame,
                   ev: pd.DataFrame | None = None, cov: bool = False, root: Path = ROOT, log=print) -> pd.DataFrame:
    out = pred_path(name, ctx_min, period, cov, root)
    if out.exists():
        return pd.read_parquet(out)
    H = cfg["horizon_bins"]
    L = ctx_min // 5
    a, b = cfg.split.utc(period)
    qs = list(cfg["quantiles"])
    qi = [qs.index(x) for x in (0.1, 0.5, 0.9)]
    frames, t_start = [], time.time()
    tk = eval_keys(cfg, ev) if period == "test" else None
    for space in cfg["spaces"]:
        if period == "test":
            t0s = pd.DatetimeIndex(tk[tk["space_id"] == space]["t0"]).as_unit("ns")
            tgt = ev[(ev["item"] == cfg["item"]) & (ev["space_id"] == space)][
                ["space_id", "item", "t0", "horizon_min", "y_true", "usage_hours"]]
        else:
            t0s = val_t0(cfg, series, space)
            tgt = val_targets(cfg, series, space, t0s)
        if len(t0s) == 0:
            continue
        g = grid(series, space, cfg["item"], pd.Timestamp(a) - pd.Timedelta(minutes=ctx_min + 5),
                 pd.Timestamp(b) + pd.Timedelta(minutes=5 * (H + 1)))
        ctx, ratio = contexts(g, t0s, L, cfg["missing"])
        unpred = ratio < cfg["min_context_obs"]
        cur = g.reindex(t0s).to_numpy(np.float32)
        q = np.full((len(t0s), H, len(qs)), np.nan, np.float32)
        ok = ~unpred
        if ok.any():
            if cov:
                past, fut = _covariates(cfg, series, space, g.index, t0s[ok], L, H)
                q[ok] = predictor.predict_cov(ctx[ok], past, fut, H, qs)
            else:
                q[ok] = predictor.predict(ctx[ok], H, qs)
        rows = []
        for h in cfg.horizons:
            k = h // 5 - 1
            p = q[:, k][:, qi]
            p = np.where(unpred[:, None], cur[:, None], p)          # 예측 불가 → 현재값 유지(구간 폭 0)
            rows.append(pd.DataFrame({"space_id": space, "item": cfg["item"], "t0": t0s, "horizon_min": h,
                                      "p10": p[:, 0], "p50": p[:, 1], "p90": p[:, 2], "unpredictable": unpred}))
        pr = pd.concat(rows, ignore_index=True)
        tgt = tgt.assign(t0=pd.DatetimeIndex(tgt["t0"]).as_unit("ns"))
        how = "left" if period == "test" else "inner"
        frames.append(tgt.merge(pr, on=["space_id", "item", "t0", "horizon_min"], how=how, validate="one_to_one"))
    df = pd.concat(frames, ignore_index=True)
    df["y_pred"] = df["p50"]
    if period == "test" and df["y_pred"].isna().any():
        raise RuntimeError("평가 시점 행 중 예측이 없는 행이 있음")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    meta = {"model": name, "ctx_min": ctx_min, "period": period, "cov": cov, "rows": len(df),
            "unpredictable_rows": int(df["unpredictable"].sum()), "seconds": round(time.time() - t_start, 1),
            "missing": cfg["missing"], "device": getattr(predictor, "device", None),
            "dtype": getattr(predictor, "dtype", None), "postprocess": getattr(predictor, "stats", None)}
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    log(f"  {name} ctx{ctx_min} {period}{' cov' if cov else ''}: {len(df):,}행, {meta['seconds']}초")
    return df

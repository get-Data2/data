"""과제 3 예측 모델로 환기 경보 판정 (research R9, 생각해 볼 점 Q4).

t의 5~30분 CO₂ 예측 최댓값 ≥ 1,000 − δ이면 경보. δ는 검증 기간 재현율 85% 기준으로 정한다(시험 미사용).
과제 3 모델 묶음(var/forecast/bundle/)이 없으면 건너뛰고 그 사실을 결과에 남긴다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .selection import metrics, vent_threshold


def _forecast_max30(cfg, data, rows: pd.DataFrame, bundle_dir: Path) -> np.ndarray:
    import torch
    from forecast.config import load_forecast_config
    from forecast.data import build_arrays
    from forecast.dataset import Windows
    from forecast.env import pick_device
    from forecast.experiments import exp_from_key
    from forecast.models import ForecastModel
    from forecast.train import predict
    b = json.loads((bundle_dir / "bundle.json").read_text())
    fcfg = load_forecast_config()
    exp = exp_from_key(b["config"], [b["seed"]])
    dev = pick_device("auto")
    arr = build_arrays(fcfg, data.series)
    W = Windows(arr, exp, json.loads(Path(fcfg["inputs"]["norm"]).read_text()), fcfg["horizon_bins"], dev)
    m = ForecastModel(exp, W.fp, W.ff, W.n_space, W.H).to(dev)
    m.load_state_dict(torch.load(bundle_dir / "model.pt", map_location=dev))
    si = rows["space_id"].map({s: i for i, s in enumerate(arr.spaces)}).to_numpy()
    ti = arr.index_of(rows["bin"])
    ok = (ti >= W.L - 1) & (ti + W.H < len(arr.t))
    out = np.full(len(rows), np.nan)
    if ok.any():
        p, _ = predict(m, W, np.column_stack([si[ok], ti[ok]]))
        out[ok] = p[:, :6, 0].max(axis=1)             # 5~30분 CO₂ 예측 최댓값
    return out


def forecast_vent_eval(cfg, data, test_rows: pd.DataFrame, sel: dict, log=print) -> dict:
    bdir = Path(cfg["inputs"]["forecast_bundle"])
    if not (bdir / "bundle.json").exists():
        log("  과제 3 모델 묶음 없음 — 예측 기반 환기 판정은 과제 3 `forecast test` 뒤에 다시 평가")
        return {"status": "skipped", "reason": "과제 3 모델 묶음(var/forecast/bundle/) 없음"}
    from .pipeline import dataset
    d = dataset(data.root, cfg, "vent")
    va = d[d["split"] == "val"].copy()
    va["fmax"] = _forecast_max30(cfg, data, va, bdir)
    va = va.dropna(subset=["fmax"])
    # 점수 = 예측 최댓값. 검증 재현율 85% 기준으로 임계값(= 1,000 − δ)을 고른다
    thr, ok, mv = vent_threshold(va["y"], va["fmax"], cfg["selection"]["vent"]["threshold"]["min_recall"])
    te = test_rows.copy()
    te["fmax"] = _forecast_max30(cfg, data, te, bdir)
    te = te.dropna(subset=["fmax"])
    m = metrics(te["y"], te["fmax"] >= thr)
    return {"status": "done", "threshold_ppm": thr, "delta_ppm": cfg["labels"]["vent"]["threshold_ppm"] - thr,
            "val": mv, "recall_ok_val": ok, "test": m}

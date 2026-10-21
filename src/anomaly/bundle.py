"""과제 8용 탐지기 묶음: 오토인코더 가중치·열 순서·정규화·검증 분위수 표·임계값·규칙 파라미터·알림 대상 (data-model "탐지기 묶음")."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .alarms import require_thresholds

INPUT_SPEC_VERSION = 1


def save_bundle(work, log=print):
    thr = require_thresholds(work.root / "thresholds.json")
    out = work.root / "bundle"
    out.mkdir(parents=True, exist_ok=True)
    a = work.cfg["autoencoder"]
    seed = a["seeds"][0]
    spaces = {}
    val = pd.read_parquet(work.root / "scores" / "ae__val.parquet") if (work.root / "scores" / "ae__val.parquet").exists() else None
    for space in work.cfg.spaces:
        src = work.root / "models" / f"{space}-s{seed}.pt"
        if not src.exists():
            continue
        shutil.copy(src, out / f"ae_{space}.pt")
        st = pd.read_parquet(work.root / f"stats_{space}.parquet")
        ref = np.sort(val[val["space_id"] == space]["score"].to_numpy()) if val is not None else np.zeros(0)
        # 점수 → 분위수 변환표(1,001점). 서비스는 보간으로 q를 얻는다
        qs = np.linspace(0, 1, 1001)
        spaces[space] = {"cols": list(work.cols[space]), "stats": st[["mean", "std", "mad_diff"]].to_dict("index"),
                         "colnorm": np.load(work.root / "models" / f"{space}-s{seed}-colnorm.npy").tolist(),
                         "score_quantiles": np.quantile(ref, qs).tolist() if len(ref) else []}
    body = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "input_spec_version": INPUT_SPEC_VERSION,
            "input_spec": {"version": INPUT_SPEC_VERSION}, "service_entry": "anomaly.serve:load",
            "config": work.cfg.digest, "window": a["window"], "score_tail": a["score_tail"], "channels": a["channels"],
            "bottleneck": a["bottleneck"], "spaces": spaces,
            "thresholds": {d: v["threshold_q"] for d, v in thr["detectors"].items()}, "thresholds_sha256": thr["sha256"],
            "rules": work.cfg["rules"], "drift": work.cfg["drift"], "alarms": work.cfg["alarms"],
            "operations": work.cfg["operations"], "time_features": "forecast.features.time_features(5열)"}
    (out / "bundle.json").write_text(json.dumps(body, ensure_ascii=False, indent=1, default=str))
    log(f"탐지기 묶음 → {out}")
    return out

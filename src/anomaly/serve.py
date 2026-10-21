"""과제 8 서비스 진입점(bundle.json "service_entry": "anomaly.serve:load").

서비스가 넘기는 센서 5분 표(prep.sensor, 공통 전처리)에서 공간별 최근 창을 만들어 오토인코더 점수 → 검증 분위수 → 임계값 판정과 원인 상위 3을 낸다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.config import load_analysis_config

from .autoencoder import ConvAE, column_errors, make_inputs
from .data import split_col


class AnomalyService:
    def __init__(self, path, device=None):
        self.path = Path(path)
        self.b = json.loads((self.path / "bundle.json").read_text())
        self.device = torch.device(device) if isinstance(device, str) else (device or torch.device("cpu"))
        self.acfg = load_analysis_config()
        self.version = f"anomaly-{self.b['config']}"
        self.sha256 = hashlib.sha256((self.path / "bundle.json").read_bytes()).hexdigest()[:16]
        self.thr = float(self.b["thresholds"].get("ae", 1.0))
        self.models, self.stats = {}, {}
        for space, s in self.b["spaces"].items():
            cols = s["cols"]
            n_in = 2 * len(cols) + 5
            m = ConvAE(n_in, len(cols), self.b["window"], self.b["channels"], self.b["bottleneck"]).to(self.device)
            m.load_state_dict(torch.load(self.path / f"ae_{space}.pt", map_location=self.device))
            m.eval()
            self.models[space] = m
            self.stats[space] = pd.DataFrame(s["stats"]).T

    def _wide(self, prep, space, cols) -> pd.DataFrame:
        s = prep.sensor
        W = self.b["window"]
        t = pd.date_range(prep.t0_bin - pd.Timedelta(minutes=5 * (W - 1)), prep.t0_bin, freq="5min")
        if s is None or not len(s):
            return pd.DataFrame(index=t, columns=cols, dtype=float)
        sub = s[(s["space_id"].astype(str) == space) & (~s["filled"].astype(bool)) & s["bin"].isin(t)]
        sub = sub.assign(col=sub["device_id"].astype(str) + "/" + sub["item"].astype(str))
        w = sub.pivot_table(index="bin", columns="col", values="value", aggfunc="first")
        return w.reindex(index=t, columns=cols)

    def predict(self, prep) -> dict:
        out = {}
        for space, s in self.b["spaces"].items():
            cols = s["cols"]
            w = self._wide(prep, space, cols)
            if w.notna().sum().sum() == 0:
                continue
            X, Z = make_inputs(w, self.stats[space], self.acfg)
            err = column_errors(self.models[space], X, Z, self.b["window"], self.b["score_tail"], self.device)[-1]
            r = err / np.asarray(s["colnorm"])
            if np.isnan(r).all():
                continue
            score = float(np.nansum(r))
            qs = np.asarray(s["score_quantiles"])
            q = float(np.interp(score, qs, np.linspace(0, 1, len(qs)))) if len(qs) else float("nan")
            order = np.argsort(-np.nan_to_num(r, nan=-np.inf))[:3]
            tot = float(np.nansum(np.clip(r, 0, None))) or 1.0
            causes = [{"device_id": split_col(cols[i])[0], "item": split_col(cols[i])[1],
                       "contribution": round(float(r[i]) / tot, 3)} for i in order if not np.isnan(r[i])]
            out[space] = {"is_anomaly": bool(q >= self.thr), "score": round(score, 4), "score_q": round(q, 6),
                          "threshold_q": self.thr, "causes": causes}
        return out


def load(path, device=None) -> AnomalyService:
    return AnomalyService(path, device)

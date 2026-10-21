"""자기 성능 감시(심화 2, research R12): 예측을 기록하고 정답 칸이 닫히면 채점, 24시간 이동 MAE가 과제 3 시험 MAE × k를 넘으면 model_drift."""
from __future__ import annotations

from collections import deque

import pandas as pd

WATCH_H = (5, 15, 30, 60)


class SelfMonitor:
    def __init__(self, cfg, recorder=None):
        s = cfg["selfmonitor"]
        self.window = pd.Timedelta(hours=s["window_h"])
        self.factor = float(s["drift_factor"])
        self.rec = recorder
        self.pending: dict[tuple, float] = {}          # (목표 칸 시작, 공간, 항목, 지평) → 예측
        self.scored: deque = deque()                   # (채점 시각, 공간, 항목, 지평, 절대오차)

    def log(self, t0_bin: pd.Timestamp, space: str, forecast: dict):
        for i, h in enumerate(forecast["horizons_min"]):
            if h not in WATCH_H:
                continue
            target = t0_bin + pd.Timedelta(minutes=h)
            for item in ("co2", "temperature"):
                self.pending[(target, space, item, h)] = float(forecast[item][i])
                if self.rec:
                    self.rec.put("forecast_log", {"t0": t0_bin.isoformat(), "space_id": space, "item": item,
                                                  "horizon_min": h, "y_pred": float(forecast[item][i])})

    def score(self, space_bins: pd.DataFrame, now: pd.Timestamp):
        """닫힌 관측 칸이 생긴 예측을 채점한다."""
        if not self.pending or space_bins.empty:
            return
        obs = space_bins[~space_bins["filled"].astype(bool)]
        truth = {(b, str(s), str(i)): float(v) for b, s, i, v in obs[["bin", "space_id", "item", "value"]].itertuples(index=False)}
        latest = obs["bin"].max()
        for key in list(self.pending):
            target, space, item, h = key
            if target > latest:
                continue
            y = truth.get((target, space, item))
            pred = self.pending.pop(key)
            if y is None:
                continue
            self.scored.append((now, space, item, h, abs(pred - y)))
            if self.rec:
                self.rec.put("forecast_log", {"t0": (target - pd.Timedelta(minutes=h)).isoformat(), "space_id": space,
                                              "item": item, "horizon_min": h, "y_pred": pred, "y_true": y,
                                              "abs_err": abs(pred - y)})
        while self.scored and now - self.scored[0][0] > self.window:
            self.scored.popleft()

    def rolling(self) -> dict:
        if not self.scored:
            return {}
        df = pd.DataFrame(list(self.scored), columns=["at", "space", "item", "h", "err"])
        g = df.groupby(["space", "item", "h"])["err"].agg(["mean", "size"]).reset_index()
        return {f"{r.space}/{r.item}/{r.h}": {"mae": round(float(r.mean), 3), "n": int(r.size)} for r in g.itertuples()}

    def drift(self, test_mae: dict, min_n: int = 30) -> list[dict]:
        """test_mae: {"공간/항목/지평": MAE}(과제 3 묶음). 이동 MAE > 시험 MAE × factor → 경보 근거."""
        out = []
        for k, v in self.rolling().items():
            ref = test_mae.get(k)
            if ref and v["n"] >= min_n and v["mae"] > ref * self.factor:
                space = k.split("/")[0]
                out.append({"kind": "model_drift", "space_id": space, "device_id": None,
                            "evidence": {"key": k, "rolling_mae": v["mae"], "test_mae": ref, "n": v["n"]}})
        return out

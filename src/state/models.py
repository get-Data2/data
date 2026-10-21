"""분류기 후보: HistGradientBoosting·로지스틱 회귀·작은 MLP — 공통 인터페이스 (research R6)."""
from __future__ import annotations

import hashlib
import io
import itertools
import json
from dataclasses import dataclass, field

import joblib
import numpy as np

SIMPLICITY = {"logreg": 0, "hgb": 1, "mlp": 2}


@dataclass(frozen=True)
class Candidate:
    kind: str
    params: dict = field(default_factory=dict)

    @property
    def hash(self) -> str:
        return hashlib.sha256(json.dumps({"kind": self.kind, "params": self.params}, sort_keys=True).encode()).hexdigest()[:12]

    def label(self) -> str:
        return self.kind + "(" + ", ".join(f"{k}={v}" for k, v in sorted(self.params.items())) + ")"


def candidates(cfg) -> list[Candidate]:
    out = []
    for kind, grid in cfg["models"]["candidates"].items():
        keys = [k for k, v in grid.items() if isinstance(v, list)]
        fixed = {k: v for k, v in grid.items() if not isinstance(v, list)}
        for combo in itertools.product(*(grid[k] for k in keys)):
            out.append(Candidate(kind, dict(fixed, **dict(zip(keys, combo)))))
    return out


def _class_weight(y):
    pos = max(float(y.mean()), 1e-6)
    return {0: 0.5 / (1 - pos), 1: 0.5 / pos}


class Model:
    """fit(X, y, seed) · predict_proba(X) → 양성 확률 · to_bytes / from_bytes."""

    def __init__(self, cand: Candidate):
        self.cand = cand
        self.med = None
        self.m = None

    def _prep(self, X):
        X = np.asarray(X, np.float32)
        if self.cand.kind == "hgb":
            return X                          # 트리 모델은 NaN을 직접 다룬다
        return np.where(np.isnan(X), self.med, X)

    def fit(self, X, y, seed: int):
        X = np.asarray(X, np.float32)
        y = np.asarray(y, int)
        self.med = np.nan_to_num(np.nanmedian(X, axis=0))
        Xp = self._prep(X)
        p = self.cand.params
        cw = "balanced" if p.get("class_weight") == "balanced" else None
        if self.cand.kind == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.m = HistGradientBoostingClassifier(max_depth=p.get("max_depth"), learning_rate=p.get("learning_rate", 0.1),
                                                    class_weight=cw, random_state=seed, early_stopping=False, max_iter=200)
            self.m.fit(Xp, y)
        elif self.cand.kind == "logreg":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            self.m = make_pipeline(StandardScaler(), LogisticRegression(C=p.get("C", 1.0), class_weight=cw, max_iter=2000,
                                                                        random_state=seed))
            self.m.fit(Xp, y)
        elif self.cand.kind == "mlp":
            self.m = _fit_mlp(Xp, y, p, seed, _class_weight(y) if cw else None)
        else:
            raise ValueError(f"모르는 모델 {self.cand.kind}")
        return self

    def predict_proba(self, X) -> np.ndarray:
        Xp = self._prep(X)
        if self.cand.kind == "mlp":
            return _mlp_proba(self.m, Xp)
        return self.m.predict_proba(Xp)[:, 1]

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        state = {"cand": (self.cand.kind, self.cand.params), "med": self.med}
        if self.cand.kind == "mlp":
            import torch
            mb = io.BytesIO()
            torch.save({"state": self.m["net"].state_dict(), "mu": self.m["mu"], "sd": self.m["sd"],
                        "arch": self.m["arch"]}, mb)
            state["mlp"] = mb.getvalue()
        else:
            state["sk"] = self.m
        joblib.dump(state, buf)
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, b: bytes) -> "Model":
        state = joblib.load(io.BytesIO(b))
        m = cls(Candidate(*state["cand"]))
        m.med = state["med"]
        if "mlp" in state:
            import torch
            d = torch.load(io.BytesIO(state["mlp"]), map_location="cpu", weights_only=False)
            net = _mlp_net(*d["arch"])
            net.load_state_dict(d["state"])
            m.m = {"net": net.eval(), "mu": d["mu"], "sd": d["sd"], "arch": d["arch"]}
        else:
            m.m = state["sk"]
        return m


def _mlp_net(n_in, hidden, layers):
    from torch import nn
    mods, d = [], n_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, 1))
    return nn.Sequential(*mods)


def _fit_mlp(X, y, p, seed, cw):
    """작은 MLP(과제 7 TensorRT 변환 대상). CPU로 충분히 빠르다(research R12)."""
    import torch
    from forecast.env import set_determinism
    set_determinism(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    arch = (X.shape[1], int(p.get("hidden", 64)), int(p.get("layers", 2)))
    net = _mlp_net(*arch)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    pw = torch.tensor(cw[1] / cw[0]) if cw else None
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    g = torch.Generator().manual_seed(seed)
    for _ in range(int(p.get("epochs", 30))):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), 1024):
            b = perm[i:i + 1024]
            loss = lossf(net(Xt[b]).squeeze(1), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return {"net": net.eval(), "mu": mu, "sd": sd, "arch": arch}


def _mlp_proba(m, X):
    import torch
    with torch.no_grad():
        z = m["net"](torch.tensor((X - m["mu"]) / m["sd"], dtype=torch.float32)).squeeze(1)
    return torch.sigmoid(z).numpy()

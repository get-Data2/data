"""사전학습 모델 불러오기·분위수 예측 (research R1·R2·R4·R11).

예측기는 같은 규약을 따른다: predict(contexts (n, L) NaN 포함, H, quantiles) → (n, H, len(q)).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import ROOT


def postprocess(q: np.ndarray) -> tuple[np.ndarray, dict]:
    """분위수 정렬(교차 제거)·0 미만 자르기, 이상 건수 기록."""
    crossed = int((np.diff(q, axis=-1) < 0).any(-1).sum())
    neg = int((q < 0).sum())
    q = np.sort(q, axis=-1)
    return np.clip(q, 0, None), {"crossed": crossed, "negative": neg}


def model_record(name: str) -> dict:
    p = ROOT / "models.json"
    rec = json.loads(p.read_text()) if p.exists() else {}
    if name not in rec:
        raise FileNotFoundError(f"{name} 가중치 기록 없음 — `pretrained fetch` 먼저")
    return rec[name]


class ChronosPredictor:
    """Chronos-Bolt·Chronos-2 공통. 받은 리비전만 쓴다(오프라인)."""

    def __init__(self, cfg, name: str, device: str = "cpu"):
        import torch
        from chronos import BaseChronosPipeline
        self.name, self.device = name, device
        rec = model_record(name)
        dt = getattr(torch, cfg["dtype"]["cuda" if device == "cuda" else "cpu"])
        self.dtype = str(dt).replace("torch.", "")
        self.pipe = BaseChronosPipeline.from_pretrained(rec["repo"], revision=rec["revision"], device_map=device,
                                                        torch_dtype=dt)
        self.batch = cfg.raw.get("batch_size", 256)
        self.covariates = bool(cfg["models"][name].get("covariates"))
        self.stats = {"crossed": 0, "negative": 0}

    def predict(self, ctx: np.ndarray, H: int, quantiles: list[float]) -> np.ndarray:
        import torch
        out = []
        for i in range(0, len(ctx), self.batch):
            x = [torch.tensor(c) for c in ctx[i:i + self.batch]]
            q, _ = self.pipe.predict_quantiles(x, prediction_length=H, quantile_levels=list(quantiles))
            if isinstance(q, list):                     # Chronos-2: 목록, 각 (변수 1, H, Q)
                q = torch.stack([t[0] for t in q])
            out.append(q.float().cpu().numpy())
        q, st = postprocess(np.concatenate(out) if out else np.zeros((0, H, len(quantiles))))
        for k in st:
            self.stats[k] += st[k]
        return q

    def predict_cov(self, ctx: np.ndarray, past_cov: dict[str, np.ndarray], future_cov: dict[str, np.ndarray],
                    H: int, quantiles: list[float]) -> np.ndarray:
        """Chronos-2 공변량 입력: past_cov[k] (n, L), future_cov[k] (n, H) — 미래 공변량은 과거 창에도 같은 이름으로 넣는다."""
        out = []
        for i in range(0, len(ctx), self.batch):
            items = []
            for j in range(i, min(i + self.batch, len(ctx))):
                items.append({"target": ctx[j], "past_covariates": {k: v[j] for k, v in past_cov.items()},
                              "future_covariates": {k: v[j] for k, v in future_cov.items()}})
            q, _ = self.pipe.predict_quantiles(items, prediction_length=H, quantile_levels=list(quantiles))
            out.append(np.stack([t[0].float().cpu().numpy() for t in q]))
        q, st = postprocess(np.concatenate(out) if out else np.zeros((0, H, len(quantiles))))
        for k in st:
            self.stats[k] += st[k]
        return q


def fetch(cfg, names=None, log=print) -> dict:
    """리비전 고정으로 받아 models.json에 커밋 해시·파일 해시·파라미터 수를 기록한다. 이미 받은 모델은 건너뛴다."""
    import hashlib
    from datetime import datetime, timezone
    from huggingface_hub import HfApi, snapshot_download
    ROOT.mkdir(parents=True, exist_ok=True)
    p = ROOT / "models.json"
    rec = json.loads(p.read_text()) if p.exists() else {}
    api = HfApi()
    for name, m in cfg["models"].items():
        if names and name not in names:
            continue
        if name in rec and (m.get("revision") in (None, rec[name]["revision"])):
            log(f"  {name}: 이미 받음({rec[name]['revision'][:12]})")
            continue
        info = api.model_info(m["repo"], revision=m.get("revision"))
        path = Path(snapshot_download(m["repo"], revision=info.sha))
        files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in sorted(path.iterdir())
                 if f.suffix in (".safetensors", ".bin", ".json")}
        params = None
        st = path / "model.safetensors"
        if st.exists():
            from safetensors import safe_open
            with safe_open(str(st), "pt") as f:
                params = int(sum(np.prod(f.get_slice(k).get_shape()) for k in f.keys()))
        rec[name] = {"repo": m["repo"], "revision": info.sha, "params": params, "files_sha256": files,
                     "license": (info.card_data or {}).get("license") if info.card_data else None,
                     "covariates": bool(m.get("covariates")), "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        log(f"  {name}: {m['repo']}@{info.sha[:12]} 파라미터 {params}")
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    return rec

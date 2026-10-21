"""(공간, t0) → 입력·미래 정보·정답 텐서 (data-model "표본"). 창은 미리 펼치지 않고 배치마다 인덱스로 자른다."""
from __future__ import annotations

import numpy as np
import torch

from .data import Arrays, norm_table


class Windows:
    """한 실험의 특징 행렬을 장치에 올려 두고 배치를 잘라 준다."""

    def __init__(self, arr: Arrays, exp, norm: dict, horizon_bins: int, device):
        self.exp = exp
        self.L = exp.lookback_min // 5
        self.H = horizon_bins
        self.device = device
        nt = norm_table(norm, arr.spaces, exp.norm, exp.train_spaces)            # (S, 3, 2)
        self.mu = torch.tensor(nt[:, :2, 0], device=device)                      # 목표 CO₂·온도
        self.sd = torch.tensor(nt[:, :2, 1], device=device)
        cap = arr.age_cap                                                        # 설정 상수(데이터에 의존하지 않음)
        self.age_step = 5 / 60 / cap
        xval = (arr.raw_ffill - nt[:, None, :, 0]) / nt[:, None, :, 1]
        xval = np.nan_to_num(xval, nan=0.0)
        cov = [arr.cov[k] for k in ("time", "schedule", "booking") if k in exp.features]
        covm = np.concatenate(cov, axis=2) if cov else np.zeros(arr.obs.shape[:2] + (0,), np.float32)
        past = np.concatenate([xval, arr.obs.astype(np.float32), arr.age_h / cap, covm], axis=2).astype(np.float32)
        self.P = torch.tensor(past, device=device)                               # (S, N, Fp)
        self.F = torch.tensor(covm, device=device)                               # (S, N, Ff)
        self.Y = torch.tensor(arr.y, device=device)                              # (S, N, 2) 원값, NaN
        self.cur = torch.tensor(arr.raw_ffill[:, :, :2], device=device)          # 현재값(인과적)
        self.n_space = len(arr.spaces)
        self._t_index = arr.t
        self.use_space = "space" in exp.features
        self.fp, self.ff = past.shape[2], covm.shape[2]

    def batch(self, idx: np.ndarray):
        s = torch.as_tensor(idx[:, 0], device=self.device)
        t = torch.as_tensor(idx[:, 1], device=self.device)
        back = torch.arange(-self.L + 1, 1, device=self.device)
        fwd = torch.arange(1, self.H + 1, device=self.device)
        past = self.P[s[:, None], t[:, None] + back]                             # (B, L, Fp)
        fut = self.F[s[:, None], t[:, None] + fwd]                               # (B, H, Ff)
        space = torch.nn.functional.one_hot(s, self.n_space).float() if self.use_space else None
        y = self.Y[s[:, None], t[:, None] + fwd]                                 # (B, H, 2)
        cur = self.cur[s, t]                                                     # (B, 2)
        mask = ~torch.isnan(y)
        mu, sd = self.mu[s], self.sd[s]
        if self.exp.target == "delta":
            tgt = (y - cur[:, None]) / sd[:, None]
        else:
            tgt = (y - mu[:, None]) / sd[:, None]
        tgt = torch.where(mask, tgt, torch.zeros_like(tgt))
        return {"past": past, "future": fut, "space": space, "target": tgt, "mask": mask,
                "current": cur, "mu": mu, "sd": sd, "s": s, "t": t, "age_step": self.age_step,
                "target_kind": self.exp.target}

    def to_original(self, pred: torch.Tensor, b: dict) -> torch.Tensor:
        """정규화 목표 → 원 단위 예측(B, H, 2)."""
        if self.exp.target == "delta":
            return b["current"][:, None] + pred * b["sd"][:, None]
        return b["mu"][:, None] + pred * b["sd"][:, None]

"""GRU·LSTM·TCN 인코더 + 미래 정보 결합 헤드 (research R7).

direct: 12칸을 한 번에 낸다. recursive: 한 칸 모델을 12번 굴린다(예측을 입력 끝에 붙임). 학습도 12칸을 굴려서 한다.
"""
from __future__ import annotations

import math

import torch
from torch import nn

N_SENSOR = 3        # 입력 앞 9열 = 값 3 · 관측 표시 3 · 경과 시간 3


class RNNEncoder(nn.Module):
    def __init__(self, kind: str, n_in: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        cls = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = cls(n_in, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.out_dim = hidden

    def forward(self, x):
        out, _ = self.rnn(x)
        return out[:, -1]


class CausalBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dilation, dropout):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.c1 = nn.Conv1d(c_in, c_out, k, dilation=dilation)
        self.c2 = nn.Conv1d(c_out, c_out, k, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.drop(torch.relu(self.c1(nn.functional.pad(x, (self.pad, 0)))))
        h = self.drop(torch.relu(self.c2(nn.functional.pad(h, (self.pad, 0)))))
        return torch.relu(h + self.res(x))


def tcn_levels(lookback_bins: int, k: int) -> int:
    """수용 영역 1 + 2(k-1)(2^n - 1) ≥ 과거 길이가 되는 최소 층 수."""
    return max(1, math.ceil(math.log2((lookback_bins - 1) / (2 * (k - 1)) + 1)))


class TCNEncoder(nn.Module):
    def __init__(self, n_in: int, channels: int, k: int, lookback_bins: int, dropout: float):
        super().__init__()
        n = tcn_levels(lookback_bins, k)
        self.receptive = 1 + 2 * (k - 1) * (2 ** n - 1)
        self.net = nn.Sequential(*[CausalBlock(n_in if i == 0 else channels, channels, k, 2 ** i, dropout) for i in range(n)])
        self.out_dim = channels

    def forward(self, x):
        return self.net(x.transpose(1, 2))[:, :, -1]


class ForecastModel(nn.Module):
    def __init__(self, exp, n_past: int, n_future: int, n_space: int, horizon: int):
        super().__init__()
        sz = {"hidden": 64, "layers": 2, "channels": 64, "kernel": 3, "head": 128, "dropout": 0.1} | dict(exp.size)
        L = exp.lookback_min // 5
        if exp.model == "tcn":
            self.enc = TCNEncoder(n_past, sz["channels"], sz["kernel"], L, sz["dropout"])
        else:
            self.enc = RNNEncoder(exp.model, n_past, sz["hidden"], sz["layers"], sz["dropout"])
        self.recursive = exp.output == "recursive"
        self.H = horizon
        steps = 1 if self.recursive else horizon
        n_head = self.enc.out_dim + (n_space if "space" in exp.features else 0) + n_future * steps
        self.head = nn.Sequential(nn.Linear(n_head, sz["head"]), nn.ReLU(), nn.Dropout(sz["dropout"]),
                                  nn.Linear(sz["head"], steps * 2))

    def _step(self, past, fut, space):
        parts = [self.enc(past)]
        if space is not None:
            parts.append(space)
        if fut.shape[-1]:
            parts.append(fut.flatten(1))
        return self.head(torch.cat(parts, 1))

    def forward(self, b: dict) -> torch.Tensor:
        past, fut, space = b["past"], b["future"], b["space"]
        if not self.recursive:
            return self._step(past, fut, space).view(-1, self.H, 2)
        outs = []
        for k in range(self.H):
            o = self._step(past, fut[:, k:k + 1], space)                     # (B, 2) 목표 공간
            outs.append(o)
            # 예측을 원 단위 → 입력 정규화 값으로 바꿔 창 끝에 붙인다(습도는 직전 값 유지, 관측 표시 0)
            if b["target_kind"] == "delta":
                y = b["current"] + o * b["sd"]
            else:
                y = b["mu"] + o * b["sd"]
            last = past[:, -1]
            xval = torch.cat([(y - b["mu"]) / b["sd"], last[:, 2:3]], 1)
            obs = torch.zeros_like(last[:, N_SENSOR:2 * N_SENSOR])
            age = last[:, 2 * N_SENSOR:3 * N_SENSOR] + b["age_step"]
            new = torch.cat([xval, obs, age, fut[:, k]], 1)[:, None]
            past = torch.cat([past[:, 1:], new], 1)
        return torch.stack(outs, 1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

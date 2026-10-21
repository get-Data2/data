"""공간별 1D 합성곱 오토인코더: 정상(학습 기간)만 학습, 창 끝 칸 복원 오차 = 이상 점수, 열 오차 = 원인 기여 (research R3)."""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from forecast.features import time_features

TIME_F = 5


def make_inputs(w: pd.DataFrame, stats: pd.DataFrame, acfg) -> tuple[np.ndarray, np.ndarray]:
    """칸마다 [정규화 값(앞으로 채움, 처음 전 0) C, 관측 표시 C, 시간 특징 5]와 정규화 값(빈 칸 NaN)."""
    mu = stats.loc[w.columns, "mean"].to_numpy(float)
    sd = stats.loc[w.columns, "std"].to_numpy(float)
    z = (w.to_numpy(float) - mu) / sd
    obs = ~np.isnan(z)
    zf = pd.DataFrame(z).ffill().fillna(0.0).to_numpy()
    X = np.concatenate([zf, obs.astype(float), time_features(pd.DatetimeIndex(w.index), acfg)], 1).astype(np.float32)
    return X, z.astype(np.float32)


class ConvAE(nn.Module):
    def __init__(self, n_in: int, n_out: int, window: int, ch: int = 32, bottleneck: int = 8):
        super().__init__()
        self.window = window
        self.enc = nn.Sequential(nn.Conv1d(n_in, ch, 3, stride=2, padding=1), nn.ReLU(),
                                 nn.Conv1d(ch, bottleneck, 3, stride=2, padding=1), nn.ReLU())
        self.dec = nn.Sequential(nn.ConvTranspose1d(bottleneck, ch, 4, stride=2, padding=1), nn.ReLU(),
                                 nn.ConvTranspose1d(ch, n_out, 4, stride=2, padding=1))

    def forward(self, x):                       # x: (B, W, n_in) → (B, W, n_out)
        y = self.dec(self.enc(x.transpose(1, 2)))
        return y[:, :, : self.window].transpose(1, 2)


def window_starts(n: int, window: int, exclude: np.ndarray | None = None) -> np.ndarray:
    """창 [i−W+1, i]의 끝 i 목록. exclude(칸 bool)가 창에 하나라도 걸리면 뺀다."""
    ends = np.arange(window - 1, n)
    if exclude is not None and exclude.any():
        c = np.concatenate([[0], np.cumsum(exclude)])
        ends = ends[(c[ends + 1] - c[ends + 1 - window]) == 0]
    return ends


def _batch(X, Z, ends, window, device):
    idx = ends[:, None] + np.arange(-window + 1, 1)[None]
    x = torch.as_tensor(X[idx], device=device)
    z = torch.as_tensor(Z[idx], device=device)
    m = ~torch.isnan(z)
    return x, torch.nan_to_num(z), m


def masked_mse(pred, target, mask):
    mf = mask.float()
    return (((pred - target) ** 2) * mf).sum() / mf.sum().clamp_min(1)


def train_ae(Xtr, Ztr, ends_tr, Xva, Zva, ends_va, cfg, device, seed: int, log=print) -> tuple[nn.Module, dict]:
    from forecast.env import set_determinism
    a = cfg["autoencoder"]
    W = a["window"]
    set_determinism(seed)
    model = ConvAE(Xtr.shape[1], Ztr.shape[1], W, a["channels"], a["bottleneck"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a["train"]["lr"])
    rng = np.random.default_rng(seed)
    best, best_ep, best_state, hist = math.inf, -1, None, []
    t0 = time.time()
    for ep in range(a["train"]["max_epochs"]):
        model.train()
        order = rng.permutation(len(ends_tr))
        for i in range(0, len(order), a["train"]["batch"]):
            x, z, m = _batch(Xtr, Ztr, ends_tr[order[i:i + a["train"]["batch"]]], W, device)
            loss = masked_mse(model(x), z, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        vl = eval_loss(model, Xva, Zva, ends_va, W, device)
        hist.append({"epoch": ep, "val": round(vl, 6)})
        if vl < best - 1e-6:
            best, best_ep = vl, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif ep - best_ep >= a["train"]["patience_stop"]:
            break
    model.load_state_dict(best_state)
    rec = {"seed": seed, "epochs": len(hist), "best_epoch": best_ep, "val_loss": best, "history": hist,
           "train_seconds": round(time.time() - t0, 1), "params": sum(p.numel() for p in model.parameters()),
           "train_windows": int(len(ends_tr)), "val_windows": int(len(ends_va))}
    log(f"  오토인코더: 에폭 {len(hist)}(최고 {best_ep}) 검증 {best:.4f} {rec['train_seconds']}초")
    return model, rec


@torch.no_grad()
def eval_loss(model, X, Z, ends, W, device, batch=4096) -> float:
    model.eval()
    tot, n = 0.0, 0.0
    for i in range(0, len(ends), batch):
        x, z, m = _batch(X, Z, ends[i:i + batch], W, device)
        mf = m.float()
        tot += float((((model(x) - z) ** 2) * mf).sum())
        n += float(mf.sum())
    return tot / max(n, 1)


@torch.no_grad()
def column_errors(model, X, Z, W, tail, device, batch=4096) -> np.ndarray:
    """칸마다(창 끝) 열별 오차: 창 마지막 tail칸 중 관측 칸의 제곱오차 평균. 관측이 없으면 NaN. (N, C)"""
    model.eval()
    n, C = Z.shape
    out = np.full((n, C), np.nan, np.float32)
    ends = np.arange(W - 1, n)
    for i in range(0, len(ends), batch):
        e = ends[i:i + batch]
        x, z, m = _batch(X, Z, e, W, device)
        err = ((model(x) - z) ** 2)[:, -tail:]
        mf = m[:, -tail:].float()
        s = (err * mf).sum(1) / mf.sum(1)
        out[e] = torch.where(mf.sum(1) > 0, s, torch.full_like(s, float("nan"))).cpu().numpy()
    return out


def space_score(col_err: np.ndarray, col_norm: np.ndarray, cols: list[str], t: pd.DatetimeIndex, space: str) -> pd.DataFrame:
    """열 오차를 검증 중앙값으로 나눠 합한 공간 점수와 상위 3 원인."""
    r = col_err / col_norm[None]
    score = np.nansum(r, 1)
    valid = ~np.isnan(r).all(1)
    rr = np.where(np.isnan(r), -np.inf, r)
    top = np.argsort(-rr, 1)[:, :3]
    c = np.array(cols)
    df = pd.DataFrame({"space_id": space, "t": t, "score": score, "cause": c[top[:, 0]],
                       "cause2": c[top[:, min(1, top.shape[1] - 1)]], "cause3": c[top[:, min(2, top.shape[1] - 1)]]})
    return df[valid].reset_index(drop=True)

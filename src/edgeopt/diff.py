"""결과 차이(확인 기준 1, research R7): 기준 = PyTorch CPU FP32. 원 단위 복원은 엔진 밖 FP32.

- 예측: CO₂·온도 원 단위 차이(최대·평균·99%), 시험 MAE 변화 → CO₂ 최대 ≤ 1ppm
- 오토인코더: 공간 점수 차이, 과제 4 임계값 기준 경보 판정 바뀐 비율
- 분류기: 클래스 바뀐 비율 ≤ 0.1%
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def forecast_diff(ref: np.ndarray, out: np.ndarray, sd: np.ndarray, y_true: np.ndarray | None = None,
                  base: np.ndarray | None = None, co2_max_ppm: float = 1.0) -> dict:
    """ref·out: 정규화 출력 (n, 12, 2). sd: (n, 2). 원 단위 차이 = |out − ref| × sd(복원식의 상수항은 같다).
    y_true·base(현재값 또는 평균, (n, 2))가 있으면 원 단위 MAE 변화도 낸다."""
    d = np.abs(out - ref) * sd[:, None, :]
    co2, tmp = d[..., 0].ravel(), d[..., 1].ravel()
    r = {"co2_max_ppm": float(co2.max()), "co2_mean_ppm": float(co2.mean()), "co2_p99_ppm": float(np.percentile(co2, 99)),
         "temp_max_c": float(tmp.max()), "temp_mean_c": float(tmp.mean()), "n_samples": int(len(ref))}
    if y_true is not None and base is not None:
        pr = base[:, None, :] + ref * sd[:, None, :]
        po = base[:, None, :] + out * sd[:, None, :]
        m = ~np.isnan(y_true)
        for i, it in enumerate(("co2", "temp")):
            mi = m[..., i]
            r[f"{it}_mae_ref"] = float(np.abs(pr[..., i] - y_true[..., i])[mi].mean())
            r[f"{it}_mae_out"] = float(np.abs(po[..., i] - y_true[..., i])[mi].mean())
    r["worst_index"] = int(np.unravel_index(np.argmax(d[..., 0]), d[..., 0].shape)[0])
    r["ok"] = bool(r["co2_max_ppm"] <= co2_max_ppm)
    return r


def ae_scores(recon: np.ndarray, z: np.ndarray, colnorm: np.ndarray, tail: int) -> np.ndarray:
    """창 (n, W, C) 복원 → 공간 점수 (n,). 과제 4 space_score와 같은 식(창 끝 tail칸, 관측 칸만, colnorm으로 나눠 합)."""
    err = (recon - np.nan_to_num(z)) ** 2
    m = ~np.isnan(z)
    e, mm = err[:, -tail:], m[:, -tail:]
    col = np.where(mm.sum(1) > 0, (e * mm).sum(1) / np.maximum(mm.sum(1), 1), np.nan)
    return np.nansum(col / colnorm[None], 1)


def to_q(score: np.ndarray, quantiles: list[float]) -> np.ndarray:
    qs = np.asarray(quantiles, float)
    if not len(qs):
        return np.full(len(score), np.nan)
    return np.interp(score, qs, np.linspace(0, 1, len(qs)), left=0.0, right=1.0)


def ae_diff(ref_recon, out_recon, z, colnorm, tail, quantiles, thr_q, flip_rate_max: float = 0.001) -> dict:
    sr, so = ae_scores(ref_recon, z, colnorm, tail), ae_scores(out_recon, z, colnorm, tail)
    r = {"score_max_abs": float(np.nanmax(np.abs(so - sr))), "score_mean_abs": float(np.nanmean(np.abs(so - sr))),
         "n_samples": int(len(sr))}
    if thr_q is not None and len(quantiles):
        fr, fo = to_q(sr, quantiles) >= thr_q, to_q(so, quantiles) >= thr_q
        r["alarm_flip_rate"] = float(np.mean(fr != fo))
        r["ok"] = bool(r["alarm_flip_rate"] <= flip_rate_max)
    else:
        r["ok"] = None
    return r


def class_flip(ref_logits: np.ndarray, out_logits: np.ndarray, flip_rate_max: float = 0.001) -> dict:
    a, b = ref_logits.argmax(-1), out_logits.argmax(-1)
    rate = float(np.mean(a != b))
    return {"flip_rate": rate, "n_samples": int(len(a)), "ok": bool(rate <= flip_rate_max)}


def run_chunks(runner, inputs: dict[str, np.ndarray], batch: int = 4) -> np.ndarray:
    """엔진 프로필 최대 배치(4) 단위로 나눠 실행해 이어 붙인다."""
    n = len(next(iter(inputs.values())))
    outs = []
    for i in range(0, n, batch):
        outs.append(runner.run({k: v[i:i + batch] for k, v in inputs.items()}))
    return np.concatenate(outs)


# ---------- 실데이터 입력(과제 3 시험 표본, 과제 4 평가판) ----------

def forecast_eval_inputs(ld) -> dict:
    """과제 3 평가 시점 파일의 고유 (공간, t0) 전체(예측 불가 제외) → 모델 입력과 복원값·정답."""
    import torch
    from forecast.config import load_forecast_config
    from forecast.data import build_arrays, eval_samples, load_inputs
    from forecast.dataset import Windows
    from forecast.experiments import exp_from_key
    fcfg = load_forecast_config()
    inp = load_inputs(fcfg)
    arr = build_arrays(fcfg, inp["series"])
    exp = exp_from_key(ld.meta["config"], [0])
    W = Windows(arr, exp, inp["norm"], fcfg["horizon_bins"], torch.device("cpu"))
    idx, unpred, keys = eval_samples(arr, inp["eval"], fcfg, W.L)
    idx = idx[~unpred]
    out = {n: [] for n in ld.input_names} | {"_sd": [], "_base": [], "_y": []}
    for i in range(0, len(idx), 4096):
        b = W.batch(idx[i:i + 4096])
        for n in ld.input_names:
            out[n].append(b[n].numpy())
        out["_sd"].append(b["sd"].numpy())
        out["_base"].append((b["current"] if exp.target == "delta" else b["mu"]).numpy())
        y = W.Y[b["s"][:, None], b["t"][:, None] + torch.arange(1, W.H + 1)]
        out["_y"].append(y.numpy())
    res = {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}
    res["_keys"] = keys[~unpred].reset_index(drop=True)
    return res


def ae_eval_inputs(ld) -> dict:
    """과제 4 평가판(주입된 시험 기간) 공간 창 전체 → 입력 x와 정규화 정답 z."""
    import pandas as pd
    from anomaly.autoencoder import make_inputs
    from anomaly.config import load_anomaly_config
    from anomaly.pipeline import Work
    work = Work(load_anomaly_config())
    space = ld.meta["space"]
    w = work.board(space)[ld.meta["cols"]]
    b = json.loads(Path("var/anomaly/bundle/bundle.json").read_text())
    st = pd.DataFrame(b["spaces"][space]["stats"]).T
    X, Z = make_inputs(w, st, work.cfg.acfg)
    Wn = ld.meta["window"]
    ends = np.arange(Wn - 1, len(X))
    ix = ends[:, None] + np.arange(-Wn + 1, 1)[None]
    return {"x": X[ix].astype(np.float32), "_z": Z[ix].astype(np.float32)}


def run_diff(cfg, root: Path, runners: list[str], specs, log=print) -> dict:
    from .models import load_model
    from .runners import artifacts, make_runner
    res_p = root / "results.json"
    results = json.loads(res_p.read_text()) if res_p.exists() else {}
    crit = cfg["criteria"]
    (root / "diff").mkdir(parents=True, exist_ok=True)
    for spec in specs:
        if spec.status != "ready" or spec.kind == "classifier":
            results.setdefault(spec.name, {})["status"] = spec.status if spec.kind != "classifier" else "classifier(과제 5 묶음 형식 확인 뒤)"
            continue
        ld = load_model(spec)
        data = forecast_eval_inputs(ld) if spec.kind == "forecast" else ae_eval_inputs(ld)
        inputs = {n: data[n] for n in ld.input_names}
        ref = run_chunks(make_runner("torch_cpu", ld, {}), inputs, 512)
        entry = results.setdefault(spec.name, {})
        entry["status"] = "ok"
        for rn in runners:
            if rn == "torch_cpu":
                continue
            try:
                r = make_runner(rn, ld, artifacts(root, spec.name))
                out = run_chunks(r, inputs, 4 if rn.startswith("trt") else 512)
                if spec.kind == "forecast":
                    s = forecast_diff(ref, out, data["_sd"], data["_y"], data["_base"], crit["co2_max_ppm"])
                    if not s["ok"]:
                        k = data["_keys"].iloc[s["worst_index"]]
                        s["worst_sample"] = {"space_id": k["space_id"], "t0": str(k["t0"])}
                    co2 = (np.abs(out - ref) * data["_sd"][:, None, :])[..., 0].max(1)
                    pd.DataFrame({"space_id": data["_keys"]["space_id"], "t0": data["_keys"]["t0"], "co2_max_diff_ppm": co2}) \
                        .to_parquet(root / "diff" / f"{rn}__{spec.name}.parquet", index=False)
                else:
                    s = ae_diff(ref, out, data["_z"], np.asarray(ld.meta["colnorm"]), ld.meta["score_tail"],
                                ld.meta["score_quantiles"], ld.meta["threshold_q"], crit["class_flip_rate"])
                entry[rn] = s
                log(f"  {spec.name} × {rn}: {'✅' if s.get('ok') else ('—' if s.get('ok') is None else '❌')} "
                    + (f"CO₂ 최대 {s['co2_max_ppm']:.4f}ppm" if 'co2_max_ppm' in s else f"경보 바뀜 {s.get('alarm_flip_rate', float('nan')):.4%}"))
            except Exception as e:
                entry[rn] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
                log(f"  {spec.name} × {rn}: 실패 — {entry[rn]['error'][:120]}")
    res_p.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str))
    return results


def verdict_table(results: dict) -> pd.DataFrame:
    rows = []
    for m, e in results.items():
        for rn, s in e.items():
            if not isinstance(s, dict):
                continue
            rows.append({"모델": m, "방식": rn,
                         "CO₂ 최대 차이(ppm)": round(s["co2_max_ppm"], 4) if "co2_max_ppm" in s else "—",
                         "온도 최대 차이(℃)": round(s["temp_max_c"], 5) if "temp_max_c" in s else "—",
                         "경보·분류 바뀐 비율": f"{s.get('alarm_flip_rate', s.get('flip_rate')):.4%}" if ("alarm_flip_rate" in s or "flip_rate" in s) else "—",
                         "판정": "✅" if s.get("ok") else ("❌" if s.get("ok") is False else s.get("status", "—"))})
    return pd.DataFrame(rows)

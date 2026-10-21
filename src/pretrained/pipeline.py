"""단계 실행: predict val → select(고정) → test → bench. 결과는 var/pretrained/ (data-model)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT
from .context import load_eval, load_series
from .predict import pred_path, predict_period
from .selection import apply_ensemble, choose, ensemble_weights, freeze, grid_rows, require


def _norm(cfg) -> dict:
    return json.loads(Path(cfg["inputs"]["norm"]).read_text())


def _factory(cfg, device):
    from .models import ChronosPredictor
    return lambda name: ChronosPredictor(cfg, name, device)


# ---------- 과제 3 직접 학습 예측(최종 그룹, 씨앗 평균) ----------

def direct_preds(cfg, period: str) -> pd.DataFrame | None:
    """과제 3 selection.json의 final 그룹 실행 예측. 없으면 None(과제 3 시험 평가 전)."""
    froot = Path(cfg["inputs"]["forecast_root"])
    sp = froot / "selection.json"
    if not sp.exists():
        return None
    sel = json.loads(sp.read_text())
    fname = "test_pred.parquet" if period == "test" else "val_pred.parquet"
    dfs = []
    for rid, r in sel["runs"].items():
        p = froot / "runs" / rid / fname
        if r["group"] == "final" and p.exists():
            d = pd.read_parquet(p)
            dfs.append(d[d["item"] == cfg["item"]])
    if not dfs:
        return None
    d = pd.concat(dfs, ignore_index=True)
    k = ["space_id", "item", "t0", "horizon_min"]
    agg = {"y_pred": ("y_pred", "mean"), "y_true": ("y_true", "first")}
    if "usage_hours" in d.columns:
        agg["usage_hours"] = ("usage_hours", "first")
    out = d.groupby(k, as_index=False).agg(**agg)
    out["t0"] = pd.DatetimeIndex(out["t0"]).as_unit("ns")
    return out


# ---------- 단계 ----------

def run_predict(cfg, device, period: str, models=None, ctxs=None, cov=False, factory=None, root: Path = ROOT, log=print):
    series = load_series(cfg)
    ev = load_eval(cfg) if period == "test" else None
    make = factory or _factory(cfg, device)
    for name in models or [m for m in cfg["models"] if not cfg["models"][m].get("covariates") or cov]:
        todo = [c for c in (ctxs or cfg["context_min"]) if not pred_path(name, c, period, cov, root).exists()]
        if not todo:
            continue
        pred = make(name)
        for c in todo:
            predict_period(cfg, pred, name, c, period, series, ev, cov, root, log)


def run_select(cfg, root: Path = ROOT, log=print) -> dict:
    norm = _norm(cfg)
    val = {}
    for p in sorted((root / "preds").glob("*__val.parquet")):
        name, ctx, _ = p.stem.split("__")
        val[(name, int(ctx[3:]))] = pd.read_parquet(p)
    rows = grid_rows(cfg, val, norm)
    pick, why = choose(cfg, rows)
    body = {"split": cfg["split"], "config": cfg.digest, "grid": rows,
            "chosen": {"model": pick["model"], "ctx_min": pick["ctx_min"], "reason": why}}
    d = direct_preds(cfg, "val")
    if d is not None:
        body["ensemble"] = ensemble_weights(cfg, d, val[(pick["model"], pick["ctx_min"])])
    else:
        body["ensemble"] = None
        log("  경고: 과제 3 검증 예측 없음 — 앙상블 가중치는 정하지 않음")
    cov = {}
    for p in sorted((root / "preds").glob("*__val__cov.parquet")):
        name, ctx, _, _ = p.stem.split("__")
        cov[(name, int(ctx[3:]))] = pd.read_parquet(p)
    if cov:
        crow = grid_rows(cfg, cov, norm)
        cpick, cwhy = choose(cfg, crow)
        body["covariate_model"] = {"model": cpick["model"], "ctx_min": cpick["ctx_min"], "reason": cwhy, "grid": crow}
    b = freeze(body, root)
    log(f"선택 고정: {pick['model']} ctx{pick['ctx_min']}분 ({why}), sha256 {b['sha256'][:16]}")
    return b


def run_test(cfg, device, factory=None, root: Path = ROOT, log=print) -> dict:
    sel = require(root)                                   # 고정 전이면 여기서 거부
    ch = sel["chosen"]
    others = [m for m in cfg["models"] if not cfg["models"][m].get("covariates") and m != ch["model"]]
    run_predict(cfg, device, "test", [ch["model"]] + others, [ch["ctx_min"]], factory=factory, root=root, log=log)
    if sel.get("covariate_model"):
        cm = sel["covariate_model"]
        run_predict(cfg, device, "test", [cm["model"]], [cm["ctx_min"]], cov=True, factory=factory, root=root, log=log)
    ev = load_eval(cfg)
    res = {"selection_sha256": sel["sha256"], "chosen": ch, "rows_check": {}}
    keys = ev[(ev["item"] == cfg["item"]) & ev["space_id"].isin(cfg["spaces"])][["space_id", "t0", "horizon_min"]]
    for p in sorted((root / "preds").glob("*__test*.parquet")):
        d = pd.read_parquet(p)
        res["rows_check"][p.stem] = bool(len(d) == len(keys))
    (root / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    log(f"시험 예측 완료: {list(res['rows_check'])}")
    return res


def eval_frames(cfg, root: Path = ROOT) -> dict[str, pd.DataFrame]:
    """비교 열: 사전학습(선택 + 다른 크기 + 공변량), 직접 학습, 섞은 예측."""
    sel = require(root)
    ch = sel["chosen"]
    out = {}
    for p in sorted((root / "preds").glob(f"*__ctx{ch['ctx_min']}__test.parquet")):
        out[p.stem.split("__")[0]] = pd.read_parquet(p)
    if sel.get("covariate_model"):
        cm = sel["covariate_model"]
        p = pred_path(cm["model"], cm["ctx_min"], "test", True, root)
        if p.exists():
            out[f"{cm['model']}+공변량"] = pd.read_parquet(p)
    d = direct_preds(cfg, "test")
    if d is not None:
        out["직접 학습(과제 3)"] = d
        if sel.get("ensemble") and ch["model"] in out:
            out["섞은 예측"] = apply_ensemble(sel["ensemble"], d, out[ch["model"]])
    return out


# ---------- 비용 측정(모델·장치마다 별도 프로세스 — 최대 메모리를 따로 재기 위해) ----------

def bench_one(cfg, name: str, device: str, root: Path = ROOT) -> dict:
    import torch
    from .bench import TegraStats, conditions, latency, peak_rss_mb
    from .context import contexts, grid, eval_keys
    sel = require(root)
    L = sel["chosen"]["ctx_min"] // 5
    series = load_series(cfg)
    ev = load_eval(cfg)
    tk = eval_keys(cfg, ev)
    rng = np.random.default_rng(0)
    pick = tk.iloc[rng.choice(len(tk), size=min(64, len(tk)), replace=False)]
    ctxs = []
    for s, g in pick.groupby("space_id"):
        gg = grid(series, s, cfg["item"], g["t0"].min() - pd.Timedelta(minutes=5 * L + 5), g["t0"].max())
        ctxs.append(contexts(gg, g["t0"], L, cfg["missing"])[0])
    ctx = np.concatenate(ctxs)
    del series
    sync = torch.cuda.synchronize if device == "cuda" else None
    if name == "forecast":                               # 과제 3 최종 모델(같은 하네스)
        fn_for = _forecast_fn(device)
    else:
        from .models import ChronosPredictor
        pr = ChronosPredictor(cfg, name, device)
        fn_for = lambda b: (lambda: pr.predict(ctx[:b], cfg["horizon_bins"], list(cfg["quantiles"])))
    bc = cfg["bench"]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    out = {"model": name, "device": device, "context_len": L, "conditions": conditions(), "batches": {}}
    with TegraStats(bc["tegrastats_interval_ms"]) as idle:
        import time
        time.sleep(min(bc["idle_seconds"], 30))
    out["idle"] = idle.summary()
    for b in bc["batches"]:
        with TegraStats(bc["tegrastats_interval_ms"]) as ts:
            lat = latency(fn_for(b), bc["warmup"], bc["repeats"], sync)
        out["batches"][str(b)] = {"latency_ms": lat, "power": ts.summary()}
    out["peak_rss_mb"] = peak_rss_mb()
    if device == "cuda":
        out["gpu_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    (root / "bench").mkdir(parents=True, exist_ok=True)
    (root / "bench" / f"{name}__{device}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return out


def _forecast_fn(device):
    """과제 3 묶음 모델을 같은 입력 규약의 임의 입력으로 측정(지연·메모리만, 정확도는 과제 3 결과)."""
    import torch
    from forecast.experiments import exp_from_key
    from forecast.models import ForecastModel
    b = json.loads(Path("var/forecast/bundle/bundle.json").read_text())
    exp = exp_from_key(b["config"], [b["seed"]])
    spec = b["input_spec"]
    fp, ff = len(spec["past_columns"]), len(spec["future_columns"])
    ns = len(b["spaces"])
    m = ForecastModel(exp, fp, ff, ns, spec["horizon_bins"]).to(device).eval()
    m.load_state_dict(torch.load("var/forecast/bundle/model.pt", map_location=device))

    def make(bs):
        x = {"past": torch.randn(bs, spec["lookback_bins"], fp, device=device),
             "future": torch.randn(bs, spec["horizon_bins"], ff, device=device),
             "space": torch.nn.functional.one_hot(torch.arange(bs, device=device) % ns, ns).float() if spec["space_onehot"] else None,
             "current": torch.zeros(bs, 2, device=device), "mu": torch.zeros(bs, 2, device=device),
             "sd": torch.ones(bs, 2, device=device), "age_step": 0.01, "target_kind": exp.target}

        @torch.no_grad()
        def f():
            m(x)
        return f
    return make


def run_bench(cfg, devices=None, models=None, log=print):
    """모델 × 장치마다 하위 프로세스로 잰다(서로의 메모리가 섞이지 않게)."""
    names = models or (list(json.loads((ROOT / "models.json").read_text())) + ["forecast"])
    for name in names:
        for dev in devices or cfg["bench"]["devices"]:
            log(f"  측정: {name} {dev}")
            subprocess.run([sys.executable, "-m", "pretrained", "--device", dev, "bench-one", name], check=False)

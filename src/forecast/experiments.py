"""실험 단계 실행·검증 기준 선택·선택 고정 (research R9, contracts/cli.md).

선택은 검증 MAE로만 한다. selection.json이 고정된 뒤에만 시험 평가를 허용한다(require_selection).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import STAGE_ORDER, ExperimentConfig
from .data import build_arrays, load_inputs, split_t0
from .dataset import Windows
from .train import RUNS, run_dir, train_one

ROOT = Path("var/forecast")
PICK_STAGES = ("D1", "D2", "D3", "D4", "B1")


class SelectionMissingError(RuntimeError):
    pass


def exp_from_key(key: dict, seeds) -> ExperimentConfig:
    k = dict(key)
    k["features"] = tuple(k["features"])
    k["train_spaces"] = tuple(k["train_spaces"])
    return ExperimentConfig(**k, seeds=tuple(seeds))


@dataclass
class Context:
    cfg: object
    device: object
    inputs: dict
    arrays: object
    root: Path = ROOT

    @classmethod
    def load(cls, cfg, device, root: Path = ROOT, inputs=None, arrays=None):
        inputs = inputs or load_inputs(cfg)
        arrays = arrays or build_arrays(cfg, inputs["series"])
        inputs.pop("series", None)          # 배열을 만든 뒤 원본 표(약 200MB)는 놓는다 — 학습 메모리 2GB 이하(SC-008)
        return cls(cfg, device, inputs, arrays, root)

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    def meta(self) -> dict:
        return {"split": self.cfg["split"], "eval_sha256": self.inputs.get("eval_sha256"),
                "series_sha16": self.inputs.get("series_sha16"), "norm_sha16": self.inputs.get("norm_sha16")}

    def windows(self, exp) -> Windows:
        return Windows(self.arrays, exp, self.inputs["norm"], self.cfg["horizon_bins"], self.device)

    def indices(self, exp, W):
        tr = split_t0(self.arrays, self.cfg, "train", W.L, exp.train_spaces)
        va = split_t0(self.arrays, self.cfg, "val", W.L, exp.train_spaces)
        return tr, va


def load_run(ctx: Context, exp, seed) -> dict | None:
    p = run_dir(exp, seed, ctx.runs) / "run.json"
    return json.loads(p.read_text()) if p.exists() else None


def train_exp(ctx: Context, exp, seeds=None, retry_failed=False, log=print) -> list[dict]:
    recs = []
    W = None
    for seed in seeds if seeds is not None else exp.seeds:
        r = load_run(ctx, exp, seed)
        if r and (r["status"] == "done" or not retry_failed):
            recs.append(r)
            continue
        if W is None:
            W = ctx.windows(exp)
            tr, va = ctx.indices(exp, W)
            log(f"  표본: 학습 {len(tr):,} · 검증 {len(va):,}")
        try:
            r = train_one(exp, seed, W, tr, va, ctx.device, ctx.arrays.spaces, ctx.meta(), run_dir(exp, seed, ctx.runs), log)
            r.pop("_model", None)
        except Exception as e:      # 한 실행 실패가 전체를 멈추지 않는다(헌법 VI)
            out = run_dir(exp, seed, ctx.runs)
            out.mkdir(parents=True, exist_ok=True)
            r = {"label": exp.label(), "config_hash": exp.hash, "seed": seed, "status": "failed",
                 "error": f"{type(e).__name__}: {e}", "config": exp.key()}
            (out / "run.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
            log(f"  {exp.label()} s{seed}: 실패 — {r['error']}")
        recs.append(r)
    return recs


def summarize(recs: list[dict]) -> dict:
    v = [r["val"]["val_norm_mae"] for r in recs if r.get("status") == "done" and r.get("val")]
    return {"mean": float(np.mean(v)) if v else float("inf"), "sd": float(np.std(v)) if len(v) > 1 else 0.0,
            "n": len(v)}


def pick(cands: list[ExperimentConfig], stats: list[dict]) -> tuple[ExperimentConfig, str]:
    """검증 평균 최소. 단, 최소와의 차이가 그 표준편차 안이면 더 단순한 쪽."""
    best = min(range(len(cands)), key=lambda i: stats[i]["mean"])
    lim = stats[best]["mean"] + stats[best]["sd"]
    near = [i for i in range(len(cands)) if stats[i]["mean"] <= lim]
    chosen = min(near, key=lambda i: cands[i].simplicity())
    if chosen == best:
        return cands[chosen], "검증 오차 최소"
    return cands[chosen], f"최소({cands[best].label()})와 1표준편차 안 → 더 단순한 쪽"


def stages_path(ctx) -> Path:
    return ctx.root / "stages.json"


def load_stages(ctx) -> dict:
    p = stages_path(ctx)
    return json.loads(p.read_text()) if p.exists() else {}


def base_for(ctx, stage: str) -> ExperimentConfig:
    """이 단계 앞까지의 선택이 누적된 설정."""
    st = load_stages(ctx)
    base = ctx.cfg.defaults()
    for s in STAGE_ORDER[:STAGE_ORDER.index(stage)]:
        if s in PICK_STAGES:
            if s not in st or "picked" not in st[s]:
                raise SelectionMissingError(f"{stage} 앞 단계 {s}의 선택이 없음 — `forecast run {s}` 먼저")
            base = exp_from_key(st[s]["picked"], base.seeds)
    return base


def run_stage(ctx: Context, stage: str, retry_failed=False, log=print) -> dict:
    base = base_for(ctx, stage)
    cands = ctx.cfg.stage_candidates(stage, base)
    log(f"[{stage}] 후보 {len(cands)}개 × 씨앗 {len(base.seeds)}")
    stats = []
    for c in cands:
        stats.append(summarize(train_exp(ctx, c, retry_failed=retry_failed, log=log)))
    entry = {"candidates": [{"label": c.label(), "hash": c.hash, "key": c.key(), **s} for c, s in zip(cands, stats)],
             "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if stage in PICK_STAGES:
        chosen, why = pick(cands, stats)
        entry |= {"picked": chosen.key(), "picked_label": chosen.label(), "picked_hash": chosen.hash, "reason": why}
        log(f"[{stage}] 선택: {chosen.label()} ({why})")
    st = load_stages(ctx)
    st[stage] = entry
    stages_path(ctx).parent.mkdir(parents=True, exist_ok=True)
    stages_path(ctx).write_text(json.dumps(st, ensure_ascii=False, indent=1))
    return entry


# ---------- 선택 고정 ----------

def selection_path(ctx) -> Path:
    return ctx.root / "selection.json"


def test_set(ctx, final: ExperimentConfig) -> dict:
    """시험 평가할 실험: 최종, 일정 제외 쌍(S), 심화 2·3. B1 최적은 최종과 같다."""
    c = ctx.cfg
    return {"final": [final], "S": c.stage_candidates("S", final)[1:], "A2": c.stage_candidates("A2", final)[1:],
            "A3": c.stage_candidates("A3", final)}


def freeze_selection(ctx, log=print) -> dict:
    st = load_stages(ctx)
    missing = [s for s in PICK_STAGES if "picked" not in st.get(s, {})]
    if missing:
        raise SelectionMissingError(f"선택이 없는 단계: {missing}")
    final = exp_from_key(st["B1"]["picked"], ctx.cfg.defaults().seeds)
    runs = {}
    for group, exps in test_set(ctx, final).items():
        for e in exps:
            for seed in e.seeds:
                r = load_run(ctx, e, seed)
                if not r or r["status"] != "done":
                    raise SelectionMissingError(f"학습되지 않은 시험 대상: {group} {e.label()} s{seed}")
                runs[f"{e.hash}-s{seed}"] = {"group": group, "label": e.label(), "key": e.key(), "seed": seed,
                                             "model_sha256": r["model_sha256"]}
    body = {"split": ctx.cfg["split"], "stages": {s: {k: st[s][k] for k in ("picked_label", "picked_hash", "reason")}
                                                  for s in PICK_STAGES},
            "final": final.key(), "final_label": final.label(), "final_hash": final.hash, "runs": runs,
            "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    text = json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True)
    body["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    selection_path(ctx).write_text(json.dumps(body, ensure_ascii=False, indent=1))
    log(f"선택 고정: {final.label()} — 시험 대상 실행 {len(runs)}개, sha256 {body['sha256'][:16]}")
    return body


def require_selection(ctx) -> dict:
    """시험 평가 전 검사: 선택 파일이 있고, 변조되지 않았고, 가중치가 기록과 같아야 한다."""
    p = selection_path(ctx)
    if not p.exists():
        raise SelectionMissingError("selection.json 없음 — 검증 기준 선택(`forecast select`)을 먼저 고정해야 시험 평가 가능")
    body = json.loads(p.read_text())
    sha = body.pop("sha256", None)
    if hashlib.sha256(json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True).encode()).hexdigest() != sha:
        raise SelectionMissingError("selection.json 해시 불일치(고정 뒤 변경됨)")
    for rid, r in body["runs"].items():
        m = ctx.runs / rid / "model.pt"
        if not m.exists() or hashlib.sha256(m.read_bytes()).hexdigest()[:16] != r["model_sha256"]:
            raise SelectionMissingError(f"가중치가 선택 고정 때와 다름: {rid}")
    body["sha256"] = sha
    return body

"""명령줄: forecast <명령> (contracts/cli.md)

종료 코드: 0 성공, 2 설정·입력 오류, 5 선택 미고정(시험 평가 거부).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .config import STAGE_ORDER, ForecastConfigError, load_forecast_config

EXIT_CONFIG, EXIT_SELECTION = 2, 5


def _ctx(args, cfg):
    from .env import pick_device
    from .experiments import Context
    return Context.load(cfg, pick_device(args.device))


def cmd_check(args, cfg):
    import torch
    from .data import InputHashError, load_inputs
    from .env import env_info, pick_device
    dev = pick_device(args.device)
    info = env_info(dev)
    print(f"장치: {dev.type}" + (f" ({info['gpu']})" if dev.type == "cuda" else "") +
          f", torch {info['torch']}, CUDA {info['cuda']}, TF32 끔, 결정적 설정")
    if dev.type == "cpu" and torch.cuda.device_count() == 0:
        print("  (GPU 안 보임 — `id`에 video·render가 있는지 확인. tools/forecast는 sg로 다시 실행한다)")
    try:
        inp = load_inputs(cfg)
    except (InputHashError, FileNotFoundError) as e:
        print(f"입력 오류: {e}")
        return EXIT_CONFIG
    print(f"평가 시점 해시 일치: {inp['eval_sha256'][:16]} ({len(inp['eval']):,}행)")
    print(f"공간 시계열 {inp['series_sha16']}, 정규화 {inp['norm_sha16']}, 분할 {cfg['split']}")
    print("설정 ok")
    return 0


def cmd_build(args, cfg):
    import numpy as np
    from .data import eval_samples, split_t0
    ctx = _ctx(args, cfg)
    a = ctx.arrays
    L = cfg.defaults().lookback_min // 5
    print(f"배열: {len(a.t):,}칸 × 공간 {len(a.spaces)} ({a.t[0]} ~ {a.t[-1]})")
    print("| 공간 | 학습 표본 | 검증 표본 | 시험 표본 | 시험 예측 불가 |\n|---|---|---|---|---|")
    _, unpred, keys = eval_samples(a, ctx.inputs["eval"], cfg, L)
    for si, s in enumerate(a.spaces):
        tr = split_t0(a, cfg, "train", L, (s,))
        va = split_t0(a, cfg, "val", L, (s,))
        m = (keys["space_id"] == s).to_numpy()
        print(f"| {s} | {len(tr):,} | {len(va):,} | {int(m.sum()):,} | {int(unpred[m].sum())} |")
    bk = a.bookings
    ctx.root.mkdir(parents=True, exist_ok=True)
    bk.to_parquet(ctx.root / "booking_estimated.parquet", index=False)
    if len(bk):
        dur = (bk["end"] - bk["start"]).dt.total_seconds() / 60
        print(f"예약({', '.join(sorted(bk['source'].unique()))}): {len(bk)}건, 평균 {dur.mean():.0f}분 → {ctx.root}/booking_estimated.parquet")
    return 0


def cmd_train(args, cfg):
    from .experiments import base_for, train_exp
    ctx = _ctx(args, cfg)
    stage = args.stage
    base = base_for(ctx, stage) if stage != "D0" else cfg.defaults()
    cands = cfg.stage_candidates(stage, base)
    seeds = [args.seed] if args.seed is not None else None
    for c in cands[: (1 if args.first else None)]:
        print(f"[{stage}] {c.label()} ({c.hash})")
        train_exp(ctx, c, seeds, retry_failed=args.retry_failed)
    return 0


def cmd_run(args, cfg):
    from .experiments import run_stage
    stages = list(STAGE_ORDER) if args.stages == ["all"] else args.stages
    bad = [s for s in stages if s not in STAGE_ORDER]
    if bad:
        print(f"알 수 없는 단계: {bad} (가능: {', '.join(STAGE_ORDER)}, all)")
        return EXIT_CONFIG
    ctx = _ctx(args, cfg)
    t0 = time.time()
    for s in stages:
        ts = time.time()
        run_stage(ctx, s, retry_failed=args.retry_failed)
        print(f"[{s}] {time.time() - ts:.0f}초")
    print(f"전체 {time.time() - t0:.0f}초")
    if args.to_report:
        for f in (cmd_select, cmd_test, cmd_report):
            rc = f(args, cfg)
            if rc:
                return rc
    return 0


def cmd_select(args, cfg):
    from .experiments import Context, SelectionMissingError, freeze_selection
    try:
        freeze_selection(Context(cfg, None, {}, None))
    except SelectionMissingError as e:
        print(f"선택 고정 불가: {e}")
        return EXIT_SELECTION
    return 0


def cmd_test(args, cfg):
    from .evaluate import criteria, load_test_preds, run_test
    from .experiments import SelectionMissingError
    from analysis.pipeline import _table
    ctx = _ctx(args, cfg)
    try:
        sel = run_test(ctx)
    except SelectionMissingError as e:
        print(f"시험 평가 거부: {e}")
        return EXIT_SELECTION
    final = load_test_preds(ctx, sel, "final")[sel["final_label"]]
    print(f"최종 모델: {sel['final_label']}")
    print(_table(criteria(ctx, final)))
    from .bundle import save_bundle
    save_bundle(ctx, sel)
    return 0


def cmd_b1_test(args, cfg):
    from .experiments import SelectionMissingError
    from .report_b1 import run
    try:
        run()
    except SelectionMissingError as e:
        print(f"시험 평가 거부: {e}")
        return EXIT_SELECTION
    return 0


def cmd_report(args, cfg):
    from .report import write_reports
    ctx = _ctx(args, cfg)
    for p in write_reports(ctx):
        print(f"→ {p}")
    return 0


def cmd_info(args, cfg):
    from pathlib import Path
    from .experiments import ROOT, load_stages, selection_path
    from .experiments import Context
    runs = sorted((ROOT / "runs").glob("*/run.json"))
    st = {"done": 0, "failed": 0}
    for p in runs:
        st[json.loads(p.read_text())["status"]] += 1
    print(f"실행 기록: 완료 {st['done']}, 실패 {st['failed']}")
    stages = load_stages(Context(cfg, None, {}, None))
    for s in STAGE_ORDER:
        e = stages.get(s)
        if e:
            print(f"  {s}: 후보 {len(e['candidates'])}" + (f" → {e['picked_label']} ({e['reason']})" if "picked_label" in e else ""))
    sp = selection_path(Context(cfg, None, {}, None))
    print(f"선택 고정: {'있음 ' + json.loads(sp.read_text())['frozen_at'] if sp.exists() else '없음'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="forecast", description="과제 3 60분 예측")
    ap.add_argument("--config", default="config/forecast.yaml")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    sub.add_parser("build")
    p = sub.add_parser("train")
    p.add_argument("stage", choices=STAGE_ORDER)
    p.add_argument("--seed", type=int)
    p.add_argument("--first", action="store_true", help="단계의 첫 후보만")
    p.add_argument("--retry-failed", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("stages", nargs="+")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--to-report", action="store_true", help="끝나면 select → test → report까지")
    sub.add_parser("select")
    sub.add_parser("test")
    sub.add_parser("b1-test", help="모델 × 과거 길이 9조합 시험 성적(보고용, 선택 고정 뒤)")
    sub.add_parser("report")
    sub.add_parser("info")
    args = ap.parse_args(argv)
    try:
        cfg = load_forecast_config(args.config)
    except ForecastConfigError as e:
        print(e)
        return EXIT_CONFIG
    return {"check": cmd_check, "build": cmd_build, "train": cmd_train, "run": cmd_run, "select": cmd_select,
            "test": cmd_test, "b1-test": cmd_b1_test, "report": cmd_report, "info": cmd_info}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())

"""명령줄: anomaly <명령> (contracts/cli.md)

종료 코드: 0 성공, 2 설정·입력 오류, 5 임계값 미고정(평가 거부).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .config import AnomalyConfigError, load_anomaly_config

EXIT_CONFIG, EXIT_THRESHOLD = 2, 5


def _work(cfg):
    from .pipeline import Work
    return Work(cfg)


def _device(args):
    from forecast.env import pick_device
    return pick_device(args.device)


def cmd_check(args, cfg):
    from pathlib import Path
    from analysis.pipeline import sha256_file
    dev = _device(args)
    print(f"장치: {dev.type}, 설정 해시 {cfg.digest}, 분할 {cfg['split']}(시험 {cfg.test_days()}일, 검증 {cfg.val_days()}일)")
    for k, p in cfg["inputs"].items():
        p = Path(p)
        ok = p.exists()
        print(f"  {k}: {'있음 ' + sha256_file(p)[:16] if ok and p.is_file() else ('있음' if ok else '없음')}")
        if not ok and k in ("sensor_series", "events", "stuck"):
            return EXIT_CONFIG
    return 0


def cmd_inject(args, cfg):
    from .pipeline import run_inject
    run_inject(_work(cfg))
    return 0


def cmd_train(args, cfg):
    from .pipeline import run_train
    run_train(_work(cfg), _device(args))
    return 0


def cmd_score(args, cfg):
    from .pipeline import run_score
    run_score(_work(cfg), _device(args), args.detectors or None)
    return 0


def cmd_threshold(args, cfg):
    from .pipeline import run_threshold
    run_threshold(_work(cfg))
    return 0


def cmd_evaluate(args, cfg):
    from .alarms import ThresholdMissingError
    from .pipeline import run_evaluate
    try:
        work = _work(cfg)
        run_evaluate(work)
        from .bundle import save_bundle
        save_bundle(work)
    except ThresholdMissingError as e:
        print(f"평가 거부: {e}")
        return EXIT_THRESHOLD
    return 0


def cmd_adaptive(args, cfg):
    """헛경보 대응 실험: 최근 N일 점수로 매일 다시 잡는 적응 임계값(src/anomaly/adaptive.py)."""
    from .adaptive import run, run_scaled
    work = _work(cfg)
    out = {}
    for n in args.ref_days:
        out[f"quantile_{n}"] = run(work, args.detector, n)
        out[f"scaled_{n}"] = run_scaled(work, args.detector, n)
    (work.root / "adaptive.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(f"→ {work.root / 'adaptive.json'}")
    return 0


def cmd_report(args, cfg):
    from .report import write_reports
    for p in write_reports(_work(cfg)):
        print(f"→ {p}")
    return 0


def cmd_info(args, cfg):
    from .pipeline import ROOT
    for name in ("inject.json", "faults.parquet", "board.parquet", "models/train.json", "thresholds.json", "results.json"):
        print(f"  {name}: {'있음' if (ROOT / name).exists() else '없음'}")
    if (ROOT / "results.json").exists():
        r = json.loads((ROOT / "results.json").read_text())
        for g, v in r["groups"].items():
            tot = v["scorecard"][-1]
            print(f"  [{g}] 탐지율 {tot['rate'] * 100:.1f}%, 헛경보 {v['false']['per_day']:.2f}/일")
    return 0


def cmd_all(args, cfg):
    t0 = time.time()
    for f in (cmd_inject, cmd_train, cmd_score, cmd_threshold, cmd_evaluate, cmd_report):
        rc = f(args, cfg)
        if rc:
            return rc
    print(f"전체 {time.time() - t0:.0f}초")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="anomaly", description="과제 4 이상 탐지")
    ap.add_argument("--config", default="config/anomaly.yaml")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("check", "inject", "train", "threshold", "evaluate", "report", "info", "all"):
        sub.add_parser(c)
    p = sub.add_parser("score")
    p.add_argument("detectors", nargs="*")
    p = sub.add_parser("adaptive", help="적응 임계값 실험(최근 N일 점수 기준)")
    p.add_argument("--detector", default="ae")
    p.add_argument("--ref-days", type=int, nargs="+", default=[7, 14, 28])
    args = ap.parse_args(argv)
    if not hasattr(args, "detectors"):
        args.detectors = None
    try:
        cfg = load_anomaly_config(args.config)
    except AnomalyConfigError as e:
        print(e)
        return EXIT_CONFIG
    return {"check": cmd_check, "inject": cmd_inject, "train": cmd_train, "score": cmd_score, "threshold": cmd_threshold,
            "evaluate": cmd_evaluate, "report": cmd_report, "info": cmd_info, "all": cmd_all, "adaptive": cmd_adaptive}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())

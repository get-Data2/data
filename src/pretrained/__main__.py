"""명령줄: pretrained <명령> (contracts/cli.md)

종료 코드: 0 성공, 2 설정·입력 오류, 5 선택 미고정(시험 평가 거부).
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import ROOT, PretrainedConfigError, load_pretrained_config

EXIT_CONFIG, EXIT_SELECTION = 2, 5


def _device(args):
    from forecast.env import pick_device
    return pick_device(args.device).type


def cmd_check(args, cfg):
    from pathlib import Path
    from .context import InputHashError, load_eval
    print(f"장치: {_device(args)}, 설정 해시 {cfg.digest}, 분할 {cfg['split']}, 결측 처리 {cfg['missing']}")
    try:
        ev = load_eval(cfg)
    except (InputHashError, FileNotFoundError) as e:
        print(f"입력 오류: {e}")
        return EXIT_CONFIG
    print(f"평가 시점 해시 일치({len(ev[ev['item'] == cfg['item']]):,} CO₂ 행)")
    rec = json.loads((ROOT / "models.json").read_text()) if (ROOT / "models.json").exists() else {}
    for name in cfg["models"]:
        print(f"  {name}: {'받음 ' + rec[name]['revision'][:12] if name in rec else '없음(`pretrained fetch`)'}")
    fs = Path(cfg["inputs"]["forecast_root"]) / "selection.json"
    print(f"과제 3 선택 고정: {'있음' if fs.exists() else '없음(직접 학습 열·앙상블은 과제 3 시험 평가 뒤)'}")
    return 0


def cmd_fetch(args, cfg):
    from .models import fetch
    fetch(cfg, args.models or None)
    return 0


def cmd_predict(args, cfg):
    from .pipeline import run_predict
    run_predict(cfg, _device(args), args.period, args.models or None, args.ctx or None, cov=args.cov)
    return 0


def cmd_select(args, cfg):
    from .pipeline import run_select
    from .selection import SelectionMissingError
    try:
        run_select(cfg)
    except SelectionMissingError as e:
        print(f"선택 불가: {e}")
        return EXIT_SELECTION
    return 0


def cmd_test(args, cfg):
    from .pipeline import run_test
    from .selection import SelectionMissingError
    try:
        run_test(cfg, _device(args))
    except SelectionMissingError as e:
        print(f"시험 평가 거부: {e}")
        return EXIT_SELECTION
    return 0


def cmd_bench(args, cfg):
    from .pipeline import run_bench
    run_bench(cfg, [args.device] if args.device != "auto" else None, args.models or None)
    return 0


def cmd_bench_one(args, cfg):
    from .pipeline import bench_one
    r = bench_one(cfg, args.model, _device(args))
    print(json.dumps({k: r[k] for k in ("model", "device", "peak_rss_mb")}, ensure_ascii=False))
    return 0


def cmd_report(args, cfg):
    from .report import write_reports
    for p in write_reports(cfg):
        print(f"→ {p}")
    return 0


def cmd_info(args, cfg):
    for name in ("models.json", "selection.json", "results.json"):
        print(f"  {name}: {'있음' if (ROOT / name).exists() else '없음'}")
    preds = sorted((ROOT / "preds").glob("*.parquet")) if (ROOT / "preds").exists() else []
    print(f"  예측 파일 {len(preds)}개: " + ", ".join(p.stem for p in preds[:12]))
    bench = sorted((ROOT / "bench").glob("*.json")) if (ROOT / "bench").exists() else []
    print(f"  비용 측정 {len(bench)}개")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pretrained", description="과제 6 사전학습 모델 비교")
    ap.add_argument("--config", default="config/pretrained.yaml")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("check", "select", "test", "report", "info"):
        sub.add_parser(c)
    p = sub.add_parser("fetch")
    p.add_argument("models", nargs="*")
    p = sub.add_parser("predict")
    p.add_argument("period", choices=["val", "test"])
    p.add_argument("--models", nargs="*")
    p.add_argument("--ctx", nargs="*", type=int)
    p.add_argument("--cov", action="store_true", help="공변량 모델(심화 1)")
    p = sub.add_parser("bench")
    p.add_argument("--models", nargs="*")
    p = sub.add_parser("bench-one")
    p.add_argument("model")
    args = ap.parse_args(argv)
    try:
        cfg = load_pretrained_config(args.config)
    except PretrainedConfigError as e:
        print(e)
        return EXIT_CONFIG
    return {"check": cmd_check, "fetch": cmd_fetch, "predict": cmd_predict, "select": cmd_select, "test": cmd_test,
            "bench": cmd_bench, "bench-one": cmd_bench_one, "report": cmd_report, "info": cmd_info}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())

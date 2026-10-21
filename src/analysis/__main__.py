"""명령줄: analysis <명령> (contracts/cli.md)

종료 코드: 0 성공, 2 설정 오류, 3 DB 접속 실패, 4 데이터 부족.
"""
from __future__ import annotations

import argparse
import resource
import sys
import time

from .config import AnalysisConfigError, load_analysis_config

EXIT_CONFIG, EXIT_DB, EXIT_DATA = 2, 3, 4


class DataShortage(RuntimeError):
    pass


def _conn(args):
    from collector import store
    try:
        return store.connect(args.env)
    except Exception as e:
        print(f"MySQL 접속 실패: {type(e).__name__}", file=sys.stderr)
        sys.exit(EXIT_DB)


def cmd_extract(args, cfg):
    from .extract import extract
    print("사본 만들기(닫힌 달은 건너뜀):")
    man = extract(cfg, _conn(args), refresh=args.refresh)
    total = sum(v["rows"] for v in man["months"].values())
    print(f"사본 {len(man['months'])}개 달, 합계 {total:,}행 → {cfg['extract']['cache_dir']}/raw/")


def cmd_clean(args, cfg):
    from .pipeline import run_clean
    print(run_clean(cfg, args.split))


def cmd_events(args, cfg):
    from .pipeline import run_events
    print(run_events(cfg, args.split))


def cmd_split(args, cfg):
    from .pipeline import run_split
    print(run_split(cfg, args.split))


def cmd_diagnose(args, cfg):
    from .diagnose import run_diagnose
    print(run_diagnose(cfg, args.split))


def cmd_baseline(args, cfg):
    from .baseline import run_baseline
    print(run_baseline(cfg, args.split))


def cmd_info(args, cfg):
    from .pipeline import info
    print(info(cfg, args.split))


def cmd_all(args, cfg):
    steps = [("extract", cmd_extract), ("clean", cmd_clean), ("split", cmd_split), ("events", cmd_events),
             ("baseline", cmd_baseline), ("diagnose", cmd_diagnose)]
    t_all = time.monotonic()
    times = []
    for name, fn in steps:
        t = time.monotonic()
        print(f"\n===== {name} =====")
        fn(args, cfg)
        times.append((name, time.monotonic() - t))
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print("\n단계별 시간: " + ", ".join(f"{n} {s:.0f}초" for n, s in times))
    print(f"전체 {time.monotonic() - t_all:.0f}초, 최대 메모리 {peak_mb:.0f}MB")


def build_parser():
    p = argparse.ArgumentParser(prog="analysis", description="과제 2 데이터 이해와 기준선")
    p.add_argument("--config", default="config/analysis.yaml")
    p.add_argument("--collector-config", default="config/collector.yaml")
    p.add_argument("--env", default=".env")
    p.add_argument("--split", default=None, help="분할 버전(기본: split.active)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("extract", help="MySQL → 달별 Parquet 사본")
    s.add_argument("--refresh", action="store_true", help="닫힌 달도 다시 읽기")
    s.set_defaults(func=cmd_extract)
    sub.add_parser("clean", help="축 A 품질·결측·5분 공간 시계열").set_defaults(func=cmd_clean)
    sub.add_parser("events", help="축 B 사건 후보·의심·값 고정").set_defaults(func=cmd_events)
    sub.add_parser("split", help="분할 검증·학습 기간 정규화 통계").set_defaults(func=cmd_split)
    sub.add_parser("diagnose", help="진단서·그래프").set_defaults(func=cmd_diagnose)
    sub.add_parser("baseline", help="평가 시점·기준선 3종 MAE").set_defaults(func=cmd_baseline)
    sub.add_parser("info", help="파일 상태·해시").set_defaults(func=cmd_info)
    s = sub.add_parser("all", help="전부 순서대로")
    s.add_argument("--refresh", action="store_true")
    s.set_defaults(func=cmd_all)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not hasattr(args, "refresh"):
        args.refresh = False
    try:
        cfg = load_analysis_config(args.config, args.collector_config)
        cfg.split(args.split)
    except AnalysisConfigError as e:
        print(e, file=sys.stderr)
        return EXIT_CONFIG
    try:
        args.func(args, cfg)
    except FileNotFoundError as e:
        print(f"필요한 파일 없음: {e}", file=sys.stderr)
        return EXIT_DATA
    except DataShortage as e:
        print(f"데이터 부족: {e}", file=sys.stderr)
        return EXIT_DATA
    return 0


if __name__ == "__main__":
    sys.exit(main())

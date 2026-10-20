"""명령줄: python -m collector <명령>  (contracts/cli.md)

종료 코드: 0 성공, 2 설정 오류, 3 외부 자원 접속 실패.
"""
from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_config

EXIT_CONFIG, EXIT_EXTERNAL = 2, 3


def _connect(args):
    from . import store
    try:
        return store.connect(args.env)
    except Exception as e:
        print(f"MySQL 접속 실패: {type(e).__name__}: {e.args[0] if e.args else ''}", file=sys.stderr)
        sys.exit(EXIT_EXTERNAL)


def cmd_check_config(args, cfg):
    print(f"OK: 공간 {len(cfg.spaces) - 1}개, 원천 {len(cfg.sources)}개, 구독 토픽 {len(cfg.all_topics())}개")


def cmd_init_db(args, cfg):
    from . import store
    prefix = cfg.storage["table_prefix"] if args.prefix is None else args.prefix
    conn = _connect(args)
    tables = store.init_db(conn, prefix)
    n = store.sync_spaces(conn, cfg, prefix)
    print("테이블:", ", ".join(tables))
    print(f"공간 {n}개 동기화")


def cmd_run(args, cfg):
    from .service import Collector
    Collector(cfg, env_path=args.env, config_path=args.config).run_forever()


def cmd_report(args, cfg):
    from . import report
    conn = _connect(args)
    print(report.main(conn, cfg, args))


def cmd_replay(args, cfg):
    from .replay import replay
    print(replay(cfg, args.file, args.limit))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m collector", description="MQTT 센서 수집기")
    p.add_argument("--config", default="config/collector.yaml")
    p.add_argument("--env", default=".env")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check-config", help="설정만 검증").set_defaults(func=cmd_check_config)

    s = sub.add_parser("init-db", help="테이블 생성·공간 동기화(멱등)")
    s.add_argument("--prefix", default=None, help="테이블 접두사(기본: 설정 storage.table_prefix)")
    s.set_defaults(func=cmd_init_db)

    sub.add_parser("run", help="상시 수집기").set_defaults(func=cmd_run)

    s = sub.add_parser("report", help="수집 현황 보고")
    s.add_argument("--hours", type=float, default=24)
    s.add_argument("--from", dest="from_", help="시작 KST 'YYYY-MM-DD HH:MM'")
    s.add_argument("--to", help="끝 KST 'YYYY-MM-DD HH:MM'")
    s.add_argument("--space", help="공간 id로 제한")
    s.add_argument("--by", choices=["device", "space"], default=None, help="한 가지 표만")
    s.add_argument("--md", help="Markdown 저장 경로")
    s.add_argument("--synthetic", action="store_true", help="합성 데이터 표시")
    s.add_argument("--capacity", action="store_true", help="저장소 크기·1년 추정")
    s.add_argument("--prefix", default=None)
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("replay", help="메시지 파일을 해석만 해 보기(저장 안 함)")
    s.add_argument("--file", required=True)
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(func=cmd_replay)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return EXIT_CONFIG
    args.func(args, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

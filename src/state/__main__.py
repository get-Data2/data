"""명령줄: state <명령> (contracts/cli.md)

종료 코드: 0 성공, 2 설정·입력 오류, 3 관찰 기록 오류, 5 선택 미고정(평가 거부).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import StateConfigError, load_state_config

EXIT_CONFIG, EXIT_OBS, EXIT_SELECTION = 2, 3, 5


def _table(df) -> str:
    from analysis.pipeline import _table as t
    return t(df.fillna("")) if len(df) else "(없음)"


def _obs_path(cfg) -> Path:
    return Path(cfg["inputs"]["observations"])


def load_door_events(cfg, start, end):
    """출입문 보고(열림·닫힘·생존)를 MySQL에서 읽는다(research R3: 점검용). 시각은 KST naive."""
    import pandas as pd
    from collector import store
    devs = {d: s for s, ds in (cfg["observe"].get("door_devices") or {}).items() for d in ds}
    if not devs:
        return pd.DataFrame(columns=["space_id", "t"])
    conn = store.connect()
    try:
        q = ("SELECT device_id, ts FROM measurement WHERE device_id IN (" + ",".join(["%s"] * len(devs)) +
             ") AND ts >= %s AND ts < %s")
        with conn.cursor() as cur:
            cur.execute(q, [*devs, start, end])
            rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows, columns=["device_id", "ts"])
    df["space_id"] = df["device_id"].map(devs)
    df["t"] = pd.to_datetime(df["ts"]).dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    return df[["space_id", "t"]]


def cmd_observe(args, cfg):
    from . import observe as ob
    path = _obs_path(cfg)
    if args.status:
        t = ob.status(ob.read(path), cfg)
        print(_table(t))
        print(t.attrs["total"] + (" — 목표 달성" if t.attrs["done"] else ""))
        return 0
    if args.check:
        df = ob.read(path)
        probs = ob.validate(df, cfg.spaces, cfg["observe"].get("live_start"))
        for p in probs:
            print(f"  줄 {p.line}: {p.reason}")
        print(f"관찰 {len(df)}줄, 문제 {len(probs)}건 ({path})")
        if args.door and len(df):
            iv = ob.intervals(df, ob.now_kst())
            from datetime import timedelta
            a = (iv["start"].min() - timedelta(hours=9)).strftime("%Y-%m-%d %H:%M")
            b = (iv["end"].max() - timedelta(hours=9)).strftime("%Y-%m-%d %H:%M")
            mm = ob.door_mismatch(df, load_door_events(cfg, a, b))
            print("문 열림과 어긋나는 '비어있음' 구간(기록 확인 대상):")
            print(_table(mm))
        return EXIT_OBS if probs else 0
    if not args.space or not args.state:
        print("사용법: state observe <공간> occupied|empty|stop [--people N] [--note ...] [--at 'YYYY-MM-DD HH:MM'] [--by 머리글자]")
        return EXIT_OBS
    try:
        at = ob.parse_time(args.at) if args.at else None
        print(ob.record(path, args.space, args.state, cfg.spaces, at=at, people=args.people,
                        observer=args.by or "", note=args.note or ""))
    except ob.ObservationError as e:
        print(f"기록 거부: {e}")
        return EXIT_OBS
    return 0


def cmd_check(args, cfg):
    from .config import validate
    _, warns = validate(cfg.raw, cfg.acfg)
    print(f"설정 해시 {cfg.digest}, 분할 {cfg['split']}, 공간 {', '.join(cfg.spaces)}")
    for w in warns:
        print(f"  경고: {w}")
    rc = 0
    for k, p in cfg["inputs"].items():
        if p is None:
            continue
        ok = Path(p).exists()
        print(f"  {k}: {'있음' if ok else '없음'}")
        if not ok and k in ("series", "norm"):
            rc = EXIT_CONFIG
    try:
        from forecast.env import pick_device
        print(f"  장치: {pick_device(args.device).type}")
    except Exception as e:      # GPU가 없어도 트리·선형 모델은 CPU로 돈다
        print(f"  장치: cpu ({e})")
    return rc


def cmd_labels(args, cfg):
    from .pipeline import run_labels
    run_labels(cfg)
    return 0


def cmd_features(args, cfg):
    from .pipeline import run_features
    run_features(cfg)
    return 0


def cmd_train(args, cfg):
    from .pipeline import run_train
    run_train(cfg, args.problem, args.device)
    return 0


def cmd_select(args, cfg):
    from .pipeline import run_select
    from .selection import SelectionMissingError
    try:
        run_select(cfg)
    except SelectionMissingError as e:
        print(f"선택 고정 불가: {e}")
        return EXIT_SELECTION
    return 0


def cmd_evaluate(args, cfg):
    from .pipeline import run_evaluate
    from .selection import SelectionMissingError
    try:
        run_evaluate(cfg, args.device)
    except SelectionMissingError as e:
        print(f"평가 거부: {e}")
        return EXIT_SELECTION
    return 0


def cmd_analyze(args, cfg):
    from .pipeline import run_analyze
    run_analyze(cfg)
    return 0


def cmd_report(args, cfg):
    from .report import write_reports
    for p in write_reports(cfg):
        print(f"→ {p}")
    return 0


def cmd_info(args, cfg):
    from . import observe as ob
    from .pipeline import ROOT
    for name in ("labels_occupancy.parquet", "labels_vent.parquet", "features.parquet", "selection.json",
                 "eval/results.json", "bundle/bundle.json"):
        print(f"  {name}: {'있음' if (ROOT / name).exists() else '없음'}")
    t = ob.status(ob.read(_obs_path(cfg)), cfg)
    print(f"  관찰: {t.attrs['total']}")
    return 0


def cmd_all(args, cfg):
    t0 = time.time()
    for f in (cmd_labels, cmd_features, cmd_train, cmd_select, cmd_evaluate, cmd_analyze, cmd_report):
        rc = f(args, cfg)
        if rc:
            return rc
    print(f"전체 {time.time() - t0:.0f}초")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="state", description="과제 5 상태 추정")
    ap.add_argument("--config", default="config/state.yaml")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("observe", help="직접 관찰 기록")
    p.add_argument("space", nargs="?")
    p.add_argument("state", nargs="?", choices=["occupied", "empty", "stop"])
    p.add_argument("--people", type=int)
    p.add_argument("--note")
    p.add_argument("--at", help="KST 'YYYY-MM-DD HH:MM' (기본: 지금)")
    p.add_argument("--by", help="관찰자 머리글자")
    p.add_argument("--status", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--door", action="store_true", help="--check와 함께: 출입문 보고와 대조(MySQL 읽기)")
    p = sub.add_parser("train")
    p.add_argument("problem", nargs="?", default="all", choices=["occupancy", "vent", "leak", "all"])
    for c in ("check", "labels", "features", "select", "evaluate", "analyze", "report", "info", "all"):
        sub.add_parser(c)
    args = ap.parse_args(argv)
    if not hasattr(args, "problem"):
        args.problem = "all"
    try:
        cfg = load_state_config(args.config)
    except StateConfigError as e:
        print(e)
        return EXIT_CONFIG
    return {"observe": cmd_observe, "check": cmd_check, "labels": cmd_labels, "features": cmd_features,
            "train": cmd_train, "select": cmd_select, "evaluate": cmd_evaluate, "analyze": cmd_analyze,
            "report": cmd_report, "info": cmd_info, "all": cmd_all}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())

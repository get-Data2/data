"""명령줄: infer <명령> (contracts/cli.md)

종료 코드: 0 성공, 2 설정 오류, 3 서비스 없음·포트 사용 중, 4 입력 부족.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .config import ServiceConfigError, load_service_config

EXIT_CONFIG, EXIT_NOSERVICE, EXIT_DATA = 2, 3, 4


def _device(args, cfg=None):
    from forecast.env import pick_device
    if args.device == "auto" and cfg is not None:
        from .bundles import prefers_cpu
        if prefers_cpu(cfg):   # 과제 7: ONNX Runtime CPU 선택 → GPU를 쓰지 않는다
            return pick_device("cpu")
    return pick_device(args.device)


def _service(args, cfg, publisher=None, writer=None):
    from .loop import Service
    from .records import Recorder
    from .window import DbSource
    return Service(cfg, DbSource(cfg), _device(args, cfg), publisher=publisher, recorder=Recorder(cfg, writer))


def cmd_once(args, cfg):
    svc = _service(args, cfg)
    now = pd.Timestamp(args.at, tz="Asia/Seoul").tz_convert("UTC") if args.at else None
    rec = svc.step(now=now)
    out = {"cycle": rec, "spaces": svc.store.spaces(), "alerts": svc.store.alerts(),
           "judgements": {k: v.availability() for k, v in svc.bundles.items()}}
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0 if rec["status"] != "failed" else EXIT_DATA


def cmd_run(args, cfg):
    from collector import store as cstore
    from .api import Api, PortInUseError
    from .publish import Publisher
    from .records import db_writer
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pub = Publisher(cfg)
    svc = _service(args, cfg, publisher=pub, writer=db_writer(cstore.connect, cfg["records"]["db_tables_prefix"]))
    api = Api(svc.store, cfg)
    try:
        port = api.start()
    except PortInUseError as e:
        logging.error("%s", e)
        svc._event("api_port_in_use", str(e))
        return EXIT_NOSERVICE
    logging.info("조회 API :%d, 발행 브로커 %s:%d, 장치 %s", port, pub.host, pub.port, svc.device.type)
    pub.start()
    svc.rec.start()
    signal.signal(signal.SIGTERM, lambda *_: svc.stop())
    signal.signal(signal.SIGINT, lambda *_: svc.stop())
    signal.signal(signal.SIGHUP, lambda *_: svc.request_reload())
    pidfile = _pidfile(cfg)
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text(str(os.getpid()))
    try:
        svc.run()
    finally:
        pidfile.unlink(missing_ok=True)
        api.stop()
        pub.stop()
        svc.rec.stop()
    return 0


def _api_get(cfg, path):
    url = f"http://127.0.0.1:{cfg['api']['port']}{path}"
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def cmd_status(args, cfg):
    try:
        h = _api_get(cfg, "/v1/health")
    except Exception as e:
        print(f"서비스 응답 없음({type(e).__name__}) — `systemctl --user status wonil-inference` 확인")
        from .records import Recorder
        cyc = Recorder(cfg).read("cycle")
        if cyc:
            print(f"마지막 주기 기록: {cyc[-1].get('scheduled_at')} {cyc[-1].get('status')} {cyc[-1].get('error') or ''}")
        return EXIT_NOSERVICE
    lc = h.get("last_cycle") or {}
    print(f"상태 {h['status']}, 장치 {h.get('device')}, 메모리 {h.get('rss_mb')}MB, MQTT 접속 {h.get('mqtt_connected')}")
    ms = lc.get("total_ms") or sum(lc.get(k) or 0 for k in ("read_ms", "prep_ms", "infer_ms", "publish_ms"))
    print(f"마지막 주기 {lc.get('scheduled_at')} {lc.get('status')} {ms}ms, 누적 {h.get('cycles')}")
    for k, v in (h.get("judgements") or {}).items():
        print(f"  {k}: {v}")
    return 0


def cmd_report(args, cfg):
    from .records import Recorder
    from .report import summarize
    a = pd.Timestamp(args.from_, tz="Asia/Seoul").tz_convert("UTC")
    b = pd.Timestamp(args.to, tz="Asia/Seoul").tz_convert("UTC") if args.to else pd.Timestamp.now(tz="UTC")
    md = summarize(Recorder(cfg), a, b)
    if args.md:
        open(args.md, "w", encoding="utf-8").write(md + "\n")
        print(f"→ {args.md}")
    else:
        print(md)
    return 0


def cmd_selftest(args, cfg):
    try:
        h = _api_get(cfg, "/v1/health")
    except Exception:
        print("서비스 응답 없음")
        return EXIT_NOSERVICE
    sm = h.get("selfmonitor") or {}
    if not sm:
        print("채점된 예측 없음(예측 묶음이 없거나 정답 칸이 아직 닫히지 않음)")
    for k, v in sorted(sm.items()):
        print(f"  {k}: 이동 MAE {v['mae']} (n={v['n']})")
    return 0


def _pidfile(cfg):
    return Path(cfg["records"]["dir"]) / "infer.pid"


def cmd_reload(args, cfg):
    """HUP은 파이썬 프로세스에만 보낸다. 서비스 전체(cgroup)에 보내면 앞단 sg 셸이 HUP으로 끝나 서비스가 재시작된다."""
    try:
        pid = int(_pidfile(cfg).read_text())
        os.kill(pid, signal.SIGHUP)
    except (OSError, ValueError) as e:
        print(f"실패: 실행 중인 서비스 없음({type(e).__name__}) — `systemctl --user restart wonil-inference`")
        return EXIT_NOSERVICE
    print(f"묶음 다시 적재 신호 보냄(pid {pid}) — 다음 주기에 적재, `infer status`로 확인")
    return 0


def cmd_fault_test(args, cfg):
    """수집기 정지 → 대기 → 재시작. 각 단계를 기록(var/service/events.jsonl)해 끊김 경보 발생·해제 시각을 잰다."""
    from .records import Recorder
    rec = Recorder(cfg)
    iso = lambda: pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z")
    if args.target != "collector":
        print("대상: collector만 지원")
        return EXIT_CONFIG
    rec.put("event", {"at": iso(), "kind": "fault_test_stop", "detail": f"wonil-collector {args.minutes}분 정지"})
    subprocess.run(["systemctl", "--user", "stop", "wonil-collector"], check=False)
    print(f"수집기 정지 — {args.minutes}분 대기")
    try:
        time.sleep(args.minutes * 60)
    finally:
        subprocess.run(["systemctl", "--user", "start", "wonil-collector"], check=False)
        rec.put("event", {"at": iso(), "kind": "fault_test_start", "detail": "wonil-collector 재시작"})
        print("수집기 재시작 — `infer report`의 경보 표에서 data_gap raised·cleared 시각 확인")
    return 0


def cmd_simulate(args, cfg):
    from .control_sim import run, to_markdown
    try:
        res = run(cfg, _device(args), days=args.days)
    except RuntimeError as e:
        print(e)
        return EXIT_DATA
    md = to_markdown(res)
    print(md)
    if args.md:
        open(args.md, "w", encoding="utf-8").write(md + "\n")
        print(f"→ {args.md}")
    if args.csv:
        out = {}
        for sp, (df, sims) in res["_frames"].items():
            w = pd.concat(sims, axis=1)
            w.columns = [f"{a}.{b}" for a, b in w.columns]
            out[sp] = w.join(df[["pred30_max", "pred60"]])
        pd.concat(out, names=["space_id"]).to_csv(args.csv)
        print(f"→ {args.csv}")
    return 0


def cmd_skew(args, cfg):
    from .skew import run, to_markdown
    df = run(cfg, _device(args), days=args.days)
    md = to_markdown(df, args.days)
    print(md)
    if args.md:
        open(args.md, "w", encoding="utf-8").write(md + "\n")
        print(f"→ {args.md}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="infer", description="과제 8 상시 추론 서비스")
    ap.add_argument("--config", default="config/service.yaml")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    p = sub.add_parser("once")
    p.add_argument("--at", help="과거 시각 재생(KST, 예: '2026-10-21 14:03:20')")
    sub.add_parser("status")
    p = sub.add_parser("report")
    p.add_argument("--from", dest="from_", required=True, help="KST")
    p.add_argument("--to", help="KST(기본 지금)")
    p.add_argument("--md")
    sub.add_parser("selftest")
    sub.add_parser("reload")
    p = sub.add_parser("simulate-control")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--md")
    p.add_argument("--csv")
    p = sub.add_parser("skew-test", help="전처리 불일치 실험(부분 칸·채우기·정규화)")
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--md")
    p = sub.add_parser("fault-test")
    p.add_argument("target")
    p.add_argument("--minutes", type=float, default=10)
    args = ap.parse_args(argv)
    try:
        cfg = load_service_config(args.config)
    except ServiceConfigError as e:
        print(e)
        return EXIT_CONFIG
    return {"run": cmd_run, "once": cmd_once, "status": cmd_status, "report": cmd_report, "selftest": cmd_selftest,
            "reload": cmd_reload, "simulate-control": cmd_simulate, "fault-test": cmd_fault_test, "skew-test": cmd_skew}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())

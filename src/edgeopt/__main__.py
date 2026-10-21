"""명령줄: edgeopt <명령> (contracts/cli.md)

종료 코드: 0 성공, 2 설정 오류, 5 측정 거부(우리 학습 실행 중), 6 판정 실패로 실행 묶음 거부.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import EdgeConfigError, load_edgeopt_config, resolve_models

ROOT = Path("var/edgeopt")
EXIT_CONFIG, EXIT_BUSY, EXIT_VERDICT = 2, 5, 6


def _specs(cfg, names):
    specs = resolve_models(cfg)
    return [s for s in specs if not names or s.name in names]


def cmd_probe(args, cfg):
    """합성 Linear·Conv1d·GRU로 ONNX → ORT CPU·CUDA → TensorRT FP16 → ORT TensorRT EP 동작·결과 일치 확인(research R2)."""
    import numpy as np
    import torch
    from torch import nn
    from .export import export_onnx
    from .models import Loaded
    from .runners import OrtRunner, TorchRunner
    from .trt import TrtRunner, build_engine, env_info

    class GRUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = nn.GRU(8, 16, batch_first=True)
            self.o = nn.Linear(16, 4)

        def forward(self, x):
            return self.o(self.g(x)[0][:, -1])

    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv1d(8, 16, 3, padding=1)
            self.o = nn.Linear(16, 4)

        def forward(self, x):
            return self.o(torch.relu(self.c(x.transpose(1, 2)))[:, :, -1])

    torch.manual_seed(0)
    nets = {"linear": (nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)), (8,)),
            "conv1d": (ConvNet(), (24, 8)), "gru": (GRUNet(), (24, 8))}
    rows = []
    pdir = ROOT / "probe"
    for name, (net, shp) in nets.items():
        ld = Loaded(name, "probe", net.eval(), ["x"], ["y"], {"x": shp})
        ex = ld.example(3)
        ref = TorchRunner(ld, "cpu").run(ex)
        row = {"model": name}
        try:
            rec = export_onnx(ld, pdir / f"{name}.onnx")
            row["onnx"] = "✅" if rec["verify"]["ok"] else f"❌ {rec['verify']['max_abs_diff']:.1e}"
        except Exception as e:
            row["onnx"] = f"❌ {type(e).__name__}: {str(e)[:80]}"
            rows.append(row)
            continue
        for rn in ("ort_cpu", "ort_cuda", "ort_trt"):
            try:
                out = OrtRunner(rn, pdir / f"{name}.onnx", ["x"]).run(ex)
                d = float(np.max(np.abs(out - ref)))
                row[rn] = f"✅ {d:.1e}" if d <= (1e-5 if rn == "ort_cpu" else 1e-2) else f"❌ {d:.1e}"
            except Exception as e:
                row[rn] = f"❌ {type(e).__name__}: {str(e)[:80]}"
        b = build_engine(pdir / f"{name}.onnx", pdir / f"{name}.plan", {"x": shp}, {"min": 1, "opt": 3, "max": 4}, 64)
        if b["ok"]:
            try:
                out = TrtRunner(pdir / f"{name}.plan", ["x"], ["y"]).run(ex)
                d = float(np.max(np.abs(out - ref)))
                row["trt_fp16"] = f"✅ {d:.1e}" if d <= 1e-2 else f"❌ {d:.1e}"
            except Exception as e:
                row["trt_fp16"] = f"❌ 실행 {type(e).__name__}: {str(e)[:80]}"
        else:
            row["trt_fp16"] = f"❌ {b.get('stage')}: {b.get('error', '')[:100]}"
        rows.append(row)
    body = {"rows": rows, "env": env_info()}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "probe.json").write_text(json.dumps(body, ensure_ascii=False, indent=1, default=str))
    from analysis.pipeline import _table
    import pandas as pd
    print(_table(pd.DataFrame(rows)))
    print(f"환경: {body['env']}")
    return 0


def cmd_export(args, cfg):
    from .export import export_onnx
    from .models import load_model
    from .trt import env_info
    conv_p = ROOT / "convert.json"
    conv = json.loads(conv_p.read_text()) if conv_p.exists() else {}
    for spec in _specs(cfg, args.models):
        if spec.status != "ready":
            print(f"  {spec.name}: {spec.status} — {spec.reason}")
            conv.setdefault(spec.name, {})["status"] = spec.status
            continue
        try:
            ld = load_model(spec)
            rec = export_onnx(ld, ROOT / "onnx" / f"{spec.name}.onnx", cfg["onnx"]["opset"])
            rec["source_weights_sha256"] = spec.extra.get("weights_sha256")
            conv.setdefault(spec.name, {})["onnx"] = rec
            print(f"  {spec.name}: ONNX {'✅' if rec['verify']['ok'] else '❌'} 최대 차이 {rec['verify']['max_abs_diff']:.2e}")
        except Exception as e:
            conv.setdefault(spec.name, {})["onnx"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  {spec.name}: 실패 — {e}")
    conv["_env"] = env_info()
    ROOT.mkdir(parents=True, exist_ok=True)
    conv_p.write_text(json.dumps(conv, ensure_ascii=False, indent=1, default=str))
    return 0


def cmd_build(args, cfg):
    from .models import load_model
    from .trt import build_engine
    probe = json.loads((ROOT / "probe.json").read_text()) if (ROOT / "probe.json").exists() else {}
    if not any(str(r.get("trt_fp16", "")).startswith("✅") for r in probe.get("rows", [])):
        print("TensorRT가 probe에서 동작하지 않음 — `edgeopt probe` 결과 확인(research R2 대안 순서)")
        return EXIT_VERDICT
    conv_p = ROOT / "convert.json"
    conv = json.loads(conv_p.read_text()) if conv_p.exists() else {}
    t = cfg["trt"]
    for spec in _specs(cfg, args.models):
        onnx = ROOT / "onnx" / f"{spec.name}.onnx"
        if spec.status != "ready" or not onnx.exists():
            continue
        ld = load_model(spec)
        kinds = ([] if args.only_extra else [("trt_fp16", False)]) + ([("trt_int8", True)] if args.int8 else []) \
            + ([("trt_fp32", False)] if args.fp32 else [])
        for k, int8 in kinds:
            calib = _int8_calibration(cfg, spec, ld) if int8 else None
            rec = build_engine(onnx, ROOT / "engines" / f"{spec.name}__{k}.plan", ld.input_shapes, t["profile"],
                               t["workspace_mb"], fp16=k != "trt_fp32", int8=int8, calib=calib,
                               fp32_last_layer=t["fp32_last_layer"] and k == "trt_fp16")
            conv.setdefault(spec.name, {})[k] = rec
            print(f"  {spec.name} {k}: {'✅' if rec['ok'] else '❌ ' + rec.get('stage', '') + ' ' + rec.get('error', '')[:100]} {rec['build_seconds']}초")
    conv_p.write_text(json.dumps(conv, ensure_ascii=False, indent=1, default=str))
    return 0


def _int8_calibration(cfg, spec, ld) -> dict:
    """INT8 보정 표본: 학습 기간 (공간, t0)에서만 무작위 N개(research R8 — 시험 기간 미사용)."""
    import numpy as np
    import torch
    n, seed = cfg["int8"]["calibration_samples"], cfg["int8"]["seed"]
    if spec.kind != "forecast":
        raise RuntimeError("INT8 보정은 예측 모델만 구현(오토인코더는 평가판이 시험 기간이라 학습 기간 창이 따로 필요)")
    from forecast.config import load_forecast_config
    from forecast.data import build_arrays, load_inputs, split_t0
    from forecast.dataset import Windows
    from forecast.experiments import exp_from_key
    fcfg = load_forecast_config()
    inp = load_inputs(fcfg)
    arr = build_arrays(fcfg, inp["series"])
    exp = exp_from_key(ld.meta["config"], [0])
    W = Windows(arr, exp, inp["norm"], fcfg["horizon_bins"], torch.device("cpu"))
    idx = split_t0(arr, fcfg, "train", W.L)
    pick = idx[np.random.default_rng(seed).choice(len(idx), min(n, len(idx)), replace=False)]
    b = W.batch(pick)
    return {"samples": {k: b[k].numpy() for k in ld.input_names}, "batch": cfg["trt"]["profile"]["opt"]}


def cmd_bench(args, cfg):
    from .bench import run_bench
    runners = args.runner or cfg["runners"]
    models = [s.name for s in _specs(cfg, args.model) if s.status == "ready"]
    try:
        run_bench(cfg, ROOT, runners, models, args.batch or cfg["batches"])
    except RuntimeError as e:
        print(f"측정 거부: {e}")
        return EXIT_BUSY
    return 0


def cmd_bench_one(args, cfg):
    from .bench import bench_one
    rec = bench_one(cfg, args.model, args.runner, args.batch, args.threads, ROOT)
    print(json.dumps(rec, ensure_ascii=False, default=str))
    return 0


def cmd_diff(args, cfg):
    from .diff import run_diff, verdict_table
    from analysis.pipeline import _table
    res = run_diff(cfg, ROOT, args.runner or (cfg["runners"] + cfg.get("optional_runners", [])), _specs(cfg, args.model))
    print(_table(verdict_table(res)))
    return 0


def cmd_colocate(args, cfg):
    from .colocate import measure_colocated
    rec = measure_colocated(cfg, ROOT, args.runner)
    print(json.dumps({k: rec[k] for k in ("runner", "models", "rss_mb", "cycle_ms_mean", "cycle_ms_p95")}, ensure_ascii=False))
    return 0


def cmd_report(args, cfg):
    from .report import write_reports
    for p in write_reports(ROOT):
        print(f"→ {p}")
    return 0


def cmd_runtime(args, cfg):
    from .runtime import write_runtime
    from .trt import env_info
    res = json.loads((ROOT / "results.json").read_text()) if (ROOT / "results.json").exists() else {}
    try:
        body = write_runtime(cfg, ROOT, args.runner, resolve_models(cfg), res, args.reason or "", env_info())
    except RuntimeError as e:
        print(f"실행 묶음 거부: {e}")
        return EXIT_VERDICT
    print(f"실행 묶음 → {ROOT / 'runtime'} ({body['runner']}, 모델 {len(body['models'])}개)")
    return 0


def cmd_info(args, cfg):
    for s in resolve_models(cfg):
        print(f"  {s.name:15s} {s.kind:11s} {s.status:11s} {s.reason}")
    for name in ("probe.json", "convert.json", "results.json", "colocate.json", "runtime/runtime.json"):
        print(f"  {name}: {'있음' if (ROOT / name).exists() else '없음'}")
    print(f"  측정 기록: {len(list((ROOT / 'bench').glob('*.json'))) if (ROOT / 'bench').exists() else 0}개")
    return 0


def cmd_all(args, cfg):
    for f in (cmd_probe, cmd_export, cmd_build, cmd_bench, cmd_diff, cmd_report):
        rc = f(args, cfg)
        print(f"[{f.__name__[4:]}] 종료 코드 {rc}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="edgeopt", description="과제 7 엣지 최적화")
    ap.add_argument("--config", default="config/edgeopt.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    p = sub.add_parser("export")
    p.add_argument("models", nargs="*")
    p = sub.add_parser("build")
    p.add_argument("models", nargs="*")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--fp32", action="store_true", help="FP32 엔진(TF32 끔)도 만든다 — FP16이 1ppm 기준을 넘을 때 비교용")
    p.add_argument("--only-extra", action="store_true", help="FP16 엔진은 다시 만들지 않는다")
    for c in ("bench", "diff"):
        p = sub.add_parser(c)
        p.add_argument("--runner", action="append")
        p.add_argument("--model", action="append")
        if c == "bench":
            p.add_argument("--batch", action="append", type=int)
    p = sub.add_parser("_bench_one")
    p.add_argument("--model", required=True)
    p.add_argument("--runner", required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--threads", type=int)
    p = sub.add_parser("colocate")
    p.add_argument("--runner", default="torch_gpu")
    sub.add_parser("report")
    p = sub.add_parser("runtime")
    p.add_argument("--runner", required=True)
    p.add_argument("--reason")
    sub.add_parser("info")
    sub.add_parser("all")
    args = ap.parse_args(argv)
    for k in ("models", "model", "runner", "batch", "int8"):
        if not hasattr(args, k):
            setattr(args, k, None)
    try:
        cfg = load_edgeopt_config(args.config)
    except EdgeConfigError as e:
        print(e)
        return EXIT_CONFIG
    fn = {"probe": cmd_probe, "export": cmd_export, "build": cmd_build, "bench": cmd_bench, "_bench_one": cmd_bench_one,
          "diff": cmd_diff, "colocate": cmd_colocate, "report": cmd_report, "runtime": cmd_runtime, "info": cmd_info,
          "all": cmd_all}[args.cmd]
    return fn(args, cfg)


if __name__ == "__main__":
    sys.exit(main())

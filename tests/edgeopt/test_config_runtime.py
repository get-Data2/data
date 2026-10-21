import json
from dataclasses import replace

import pytest

from edgeopt.colocate import llm_table
from edgeopt.config import EdgeConfig, resolve_models, validate
from edgeopt.runtime import check_env, write_runtime

pytestmark = pytest.mark.unit


def test_validation_rules(ecfg):
    assert validate(ecfg.raw) == []
    bad = json.loads(json.dumps(ecfg.raw))
    bad["runners"] = ["ort_cpu"]
    bad["batches"] = [0, 8]
    bad["bench"]["repeat"] = 10
    errs = validate(bad)
    assert any("torch_cpu" in e for e in errs) and any("0 이하" in e for e in errs) and any("repeat" in e for e in errs)


def test_resolve_with_fake_bundles(ecfg, fake_bundles):
    specs = {s.name: s for s in resolve_models(ecfg, fake_bundles)}
    assert specs["forecast_final"].status == "ready" and specs["forecast_final"].extra["weights_sha256"]
    assert specs["ae_lab"].status == "ready" and specs["ae_office"].status == "ready"
    assert specs["forecast_gru"].status == "waiting"                       # 과제 3 stages.json 없음
    assert specs["state"].status == "unsupported" and "트리" in specs["state"].reason


def test_resolve_without_bundles(ecfg, tmp_path):
    specs = resolve_models(ecfg, tmp_path)
    assert all(s.status == "waiting" for s in specs) and all(s.reason for s in specs)


def test_runtime_requires_passing_verdict(ecfg, fake_bundles, tmp_path):
    from edgeopt.export import export_onnx
    from edgeopt.models import load_model
    root = tmp_path / "edge"
    specs = [s for s in resolve_models(ecfg, fake_bundles) if s.status == "ready"]
    for s in specs:
        export_onnx(load_model(s), root / "onnx" / f"{s.name}.onnx")
    res = {s.name: {"ort_cpu": {"ok": True}} for s in specs}
    body = write_runtime(ecfg, root, "ort_cpu", specs, res, "시험", {"tensorrt": "x"})
    rt = json.loads((root / "runtime" / "runtime.json").read_text())
    assert rt["runner"] == "ort_cpu" and set(rt["models"]) == {s.name for s in specs}
    assert all((root / "runtime" / m["file"]).exists() and m["sha256"] for m in rt["models"].values())
    res["ae_lab"]["ort_cpu"]["ok"] = False
    with pytest.raises(RuntimeError, match="ae_lab"):
        write_runtime(ecfg, root, "ort_cpu", specs, res, "시험", {})
    assert check_env({"runner": "trt_fp16", "env": {"tensorrt": "11.4", "gpu": "Orin", "cc": "8.7"}},
                     {"tensorrt": "11.5", "gpu": "Orin", "cc": "8.7"}) == ["tensorrt: 11.4 → 11.5"]
    assert check_env({"runner": "ort_cpu", "env": {}}, {"tensorrt": "z"}) == []


def test_llm_table_by_hand(ecfg):
    """보드 7,486MB − 예약(1.5+0.2GB = 1,740.8MB) − 서비스 1,000MB = 4,745MB.
    LLM 필요 = 크기 × 4비트/8 + 여유 1GB: 1B 1,536MB ✅, 3B 2,560MB ✅, 7B 4,608MB ✅(여유 137MB)."""
    t = llm_table(ecfg, service_mb=1000, board_total_mb=7486)
    assert list(t["필요(MB, 가정)"]) == [1536, 2560, 4608]
    assert list(t["가능"]) == ["✅", "✅", "✅"]
    t2 = llm_table(ecfg, service_mb=1000, board_total_mb=7486, other_mb=1000)     # 다른 계정 1GB 관찰
    assert list(t2["가능"]) == ["✅", "✅", "❌"]

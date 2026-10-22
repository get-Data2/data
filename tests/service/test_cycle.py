"""한 주기 결과 형식, 묶음 없음 → unavailable, 장애 격리·회복, 신선도 (spec US1·US4, tasks T016·T020)."""
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from service.loop import Service
from tests.service.conftest import FakeSource, at

pytestmark = pytest.mark.unit
CPU = torch.device("cpu")
FIELDS = {"schema", "space_id", "space_name", "t0", "t0_kst", "computed_at", "freshness_min", "stale", "availability",
          "models", "forecast", "co2_over", "anomaly", "state", "rules", "summary_ko"}


def entry_bundle(path: Path, entry: str):
    path.mkdir(parents=True, exist_ok=True)
    (path / "bundle.json").write_text(json.dumps({"service_entry": entry, "input_spec": {"version": 1}}))


def test_no_bundles_unavailable_but_runs(scfg, raw):
    svc = Service(scfg, FakeSource(raw), CPU)
    rec = svc.step(now=at("10:03:20"))
    assert rec["status"] == "ok"
    r = svc.store.space("lab")
    assert FIELDS <= set(r)
    assert r["availability"]["forecast"] == "unavailable:no_bundle"
    assert r["availability"]["anomaly"] == "unavailable:no_bundle" and r["availability"]["rules"] == "ok"
    assert r["t0"] == "2026-03-04T01:00:00Z" and r["freshness_min"] == pytest.approx(3.3, abs=0.1)


def test_full_cycle_with_bundles(scfg, raw, fc_bundle):
    entry_bundle(Path(scfg["bundles"]["anomaly"]), "tests.service.fake_entry:anomaly")
    entry_bundle(Path(scfg["bundles"]["state"]), "tests.service.fake_entry:state")
    svc = Service(scfg, FakeSource(raw), CPU)
    svc.step(now=at("10:03:20"))
    lab = svc.store.space("lab")
    assert lab["availability"] == {"rules": "ok", "forecast": "ok", "anomaly": "ok", "state": "ok"}
    assert len(lab["forecast"]["co2"]) == 12 and lab["forecast"]["horizons_min"][-1] == 60
    assert lab["co2_over"]["threshold_ppm"] == 1000
    kinds = {(a["kind"], a["space_id"]) for a in svc.store.alerts()}
    assert ("anomaly", "lab") in kinds and ("ventilation_soon", "meeting") in kinds
    assert svc.store.health()["judgements"]["forecast"] == "ok"


def test_meeting_10min_sensor_predicts(scfg, raw, fc_bundle):
    svc = Service(scfg, FakeSource(raw), CPU)
    svc.step(now=at("10:03:20"))
    assert svc.store.space("meeting")["availability"]["forecast"] == "ok"


def test_db_failure_partial_then_recover(scfg, raw):
    src = FakeSource(raw)
    svc = Service(scfg, src, CPU)
    assert svc.step(now=at("10:01:20"))["status"] == "ok"
    src.fail = True
    rec = svc.step(now=at("10:02:20"))
    assert rec["status"] == "partial" and rec["error"].startswith("read:")
    assert svc.store.space("lab") is not None                     # 남은 창으로 계속 판단
    src.fail = False
    assert svc.step(now=at("10:03:20"))["status"] == "ok"


def test_model_error_isolated(scfg, raw, fc_bundle):
    entry_bundle(Path(scfg["bundles"]["anomaly"]), "tests.service.fake_entry:broken")
    svc = Service(scfg, FakeSource(raw), CPU)
    rec = svc.step(now=at("10:03:20"))
    lab = svc.store.space("lab")
    assert lab["availability"]["anomaly"].startswith("error:") and lab["availability"]["forecast"] == "ok"
    assert rec["status"] == "partial"
    assert any(a["kind"] == "service_degraded" for a in svc.store.alerts())


def test_data_gap_and_stop_freshness(scfg, raw, fc_bundle):
    """수집이 멈추면: 끊김 경보, 15분 넘으면 stale, 60분 넘으면 예측 중단. 다시 들어오면 3주기 뒤 해제."""
    src = FakeSource(raw[raw["ts"] < at("09:00")])
    svc = Service(scfg, src, CPU)
    svc.step(now=at("09:00:20"))
    rec = svc.step(now=at("09:20:20"))
    lab = svc.store.space("lab")
    assert lab["stale"] and lab["availability"]["forecast"] == "ok"
    assert any(a["kind"] == "data_gap" for a in svc.store.alerts())
    svc.step(now=at("10:10:20"))
    assert svc.store.space("lab")["availability"]["forecast"] == "unavailable:data_gap"
    src.raw = raw
    for m in ("10:11:20", "10:12:20", "10:13:20"):
        svc.step(now=at(m))
    assert not any(a["kind"] == "data_gap" for a in svc.store.alerts())
    ev = svc.rec.read("alert")
    gap = [e["event"] for e in ev if e["kind"] == "data_gap" and e["space_id"] == "lab"]
    assert gap.count("raised") >= 1 and gap.count("cleared") >= 1


def test_records_written(scfg, raw):
    svc = Service(scfg, FakeSource(raw), CPU)
    svc.step(now=at("10:03:20"))
    cyc = svc.rec.read("cycle")
    assert len(cyc) == 1 and {"read_ms", "prep_ms", "infer_ms", "total_ms", "rss_mb"} <= set(cyc[0])


def test_data_gap_skips_power_only_device():
    """꺼진 냉난방 실내기는 power만 매분 보내고 온도는 켤 때만 보낸다(periodic_items: [power]).
    서비스는 온도·CO₂·습도만 읽으므로 이 장치의 끊김은 판정하지 않는다(2026-10-22 사무실 실내기 오경보)."""
    from analysis.config import AnalysisConfig
    from service import rules
    acfg = AnalysisConfig("x", {}, {}, periodic_items={"idu": ("power",), "both": ("power", "temperature")})
    now = pd.Timestamp("2026-10-22T00:00:00Z")
    raw = pd.DataFrame({"ts": [now - pd.Timedelta(hours=3)] * 3, "space_id": ["office"] * 3,
                        "device_id": ["idu", "both", "sensor"], "item": ["temperature"] * 3, "value": [20.0] * 3})
    out = rules.data_gap(raw, now, acfg, {}, 10)
    assert sorted(o["device_id"] for o in out) == ["both", "sensor"]


def test_ort_selection_falls_back(tmp_path):
    """과제 7 선택(ort_cpu)이어도 ONNX가 다른 가중치에서 만들어졌거나 파일이 없으면 PyTorch로 돈다."""
    from service.bundles import ort_forecast_path, prefers_cpu
    sel = tmp_path / "runtime.json"
    cfg = {"bundles": {"edge_selection": str(sel)}}
    assert ort_forecast_path(cfg, "abc")[0] is None and not prefers_cpu(cfg)
    sel.write_text(json.dumps({"runner": "ort_cpu", "models": {"forecast_final": {
        "file": "f.onnx", "sha256": "x", "source_weights_sha256": "other"}}}))
    assert prefers_cpu(cfg)
    path, why = ort_forecast_path(cfg, "abc")
    assert path is None and "다른 가중치" in why
    path, why = ort_forecast_path(cfg, "other")
    assert path is None and "파일 없음" in why

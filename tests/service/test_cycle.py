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

"""누수(확인 기준 3)·선택 고정·평가 행 일치·재개 — 합성 데이터로 전체 흐름을 축소해 돌린다."""
import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from analysis.baseline import build as baseline_build
from forecast.config import ForecastConfig
from forecast.data import build_arrays
from forecast.evaluate import comparison_table, run_test
from forecast.experiments import (Context, SelectionMissingError, freeze_selection, load_stages, require_selection,
                                  run_stage, train_exp)
from tests.forecast.conftest import synth_norm, synth_series

pytestmark = pytest.mark.unit
NO_BOOK = pd.DataFrame(columns=["space_id", "start", "end", "source"])
CPU = torch.device("cpu")


def small_cfg(fcfg, tmp_path, series):
    """단계마다 후보 1개, 에폭 1 — 흐름만 시험. 평가 시점·기준선은 과제 2 build로 만든다."""
    raw = json.loads(json.dumps(fcfg.raw, default=str))
    raw["defaults"]["train"]["max_epochs"] = 1
    raw["stages"] = {"D1": {"lookback_min": [60]}, "D2": {"features": [["sensors", "time", "space", "schedule"]]},
                     "D3": {"output": ["direct"]}, "D4": {"target": ["delta"]},
                     "B1": {"lookback_min": [60], "model": ["gru"]}}
    cfg = ForecastConfig(fcfg.path, raw, fcfg.acfg)
    bl = baseline_build(series, cfg.split, cfg.acfg)
    bl_path = tmp_path / "baseline.parquet"
    bl.to_parquet(bl_path, index=False)
    raw["inputs"] = dict(raw["inputs"], baseline=str(bl_path))
    ev = bl[["space_id", "item", "t0", "horizon_min", "y_true", "usage_hours"]]
    return cfg, ev


def ctx_for(cfg, series, ev, root):
    inputs = {"series": series, "eval": ev, "norm": synth_norm(series), "eval_sha256": "syn"}
    return Context(cfg, CPU, inputs, build_arrays(cfg, series, bookings=NO_BOOK), root)


def test_test_period_does_not_affect_training(fcfg, tmp_path):
    """시험 기간 입력·정답을 ×10 / −999로 바꿔도 학습 기록·최고 에폭·가중치·선택이 같다."""
    s1 = synth_series()
    s2 = s1.copy()
    a, _ = fcfg.split.utc("test")
    m = s2["bin"] >= pd.Timestamp(a)
    s2.loc[m & (s2["item"] == "co2"), "value"] *= 10
    s2.loc[m & (s2["item"] == "temperature"), "value"] = -999
    raw = json.loads(json.dumps(fcfg.raw, default=str))
    raw["defaults"]["train"]["max_epochs"] = 3
    cfg = ForecastConfig(fcfg.path, raw, fcfg.acfg)
    norm = synth_norm(s1)          # 정규화 통계는 과제 2 파일(학습 기간)에서 오므로 둘 다 같은 값
    exp = cfg.defaults()
    recs = []
    for i, s in enumerate((s1, s2)):
        ctx = Context(cfg, CPU, {"series": s, "eval": None, "norm": norm}, build_arrays(cfg, s, bookings=NO_BOOK),
                      tmp_path / f"r{i}")
        recs.append(train_exp(ctx, exp, log=lambda *_: None)[0])
    assert recs[0]["history"] == recs[1]["history"]
    assert recs[0]["best_epoch"] == recs[1]["best_epoch"]
    assert recs[0]["model_sha256"] == recs[1]["model_sha256"]
    assert recs[0]["val"] == recs[1]["val"]


def test_selection_gate_and_eval_rows(fcfg, tmp_path):
    series = synth_series()
    cfg, ev = small_cfg(fcfg, tmp_path, series)
    ctx = ctx_for(cfg, series, ev, tmp_path / "fc")
    # 선택 전 시험 거부
    with pytest.raises(SelectionMissingError):
        run_test(ctx, log=lambda *_: None)
    for st in ("D0", "D1", "D2", "D3", "D4", "B1", "S", "A2", "A3"):
        run_stage(ctx, st, log=lambda *_: None)
    assert load_stages(ctx)["D2"]["picked"]["lookback_min"] == 60       # 앞 단계 선택 누적
    sel = freeze_selection(ctx, log=lambda *_: None)
    assert set(r["group"] for r in sel["runs"].values()) == {"final", "S", "A2", "A3"}
    require_selection(ctx)
    run_test(ctx, log=lambda *_: None)
    # 평가 행 일치: 예측 키 = 평가 시점 키
    rid = next(k for k, r in sel["runs"].items() if r["group"] == "final")
    p = pd.read_parquet(ctx.runs / rid / "test_pred.parquet")
    k = ["space_id", "item", "t0", "horizon_min"]
    assert len(p) == len(ev) and p[k].merge(ev[k]).shape[0] == len(ev)
    assert p["y_pred"].notna().all()
    t = comparison_table(ctx, p, "all")
    assert {"persistence", "mae", "개선율(%)"} <= set(t.columns)
    # 보고서(표·그래프) 생성
    from forecast.report import write_reports
    raw = ctx.cfg.raw
    raw["inputs"] = dict(raw["inputs"], events=str(tmp_path / "events.parquet"))
    pd.DataFrame({"space_id": ["lab"], "kind": ["B3"], "start": [pd.Timestamp("2026-03-24 03:00", tz="UTC")],
                  "end": [pd.Timestamp("2026-03-24 05:00", tz="UTC")]}).to_parquet(tmp_path / "events.parquet")
    files = write_reports(ctx, tmp_path / "deliv", tmp_path / "adv")
    assert len(files) == 6 and all(f.read_text().startswith("# 과제 3") for f in files)
    assert list((tmp_path / "deliv" / "figures").glob("*.png"))
    # 고정 뒤 변조 → 거부
    sp = ctx.root / "selection.json"
    body = json.loads(sp.read_text())
    body["final_label"] = "바꿈"
    sp.write_text(json.dumps(body, ensure_ascii=False, indent=1))
    with pytest.raises(SelectionMissingError, match="해시"):
        require_selection(ctx)


def test_model_tamper_rejected(fcfg, tmp_path):
    series = synth_series()
    cfg, ev = small_cfg(fcfg, tmp_path, series)
    ctx = ctx_for(cfg, series, ev, tmp_path / "fc")
    for st in ("D0", "D1", "D2", "D3", "D4", "B1", "S", "A2", "A3"):
        run_stage(ctx, st, log=lambda *_: None)
    sel = freeze_selection(ctx, log=lambda *_: None)
    rid = next(iter(sel["runs"]))
    m = ctx.runs / rid / "model.pt"
    m.write_bytes(m.read_bytes() + b"x")
    with pytest.raises(SelectionMissingError, match="가중치"):
        require_selection(ctx)


def test_resume_skips_done(fcfg, tmp_path):
    series = synth_series()
    raw = json.loads(json.dumps(fcfg.raw, default=str))
    raw["defaults"]["train"]["max_epochs"] = 1
    cfg = ForecastConfig(fcfg.path, raw, fcfg.acfg)
    ctx = Context(cfg, CPU, {"series": series, "eval": None, "norm": synth_norm(series)},
                  build_arrays(cfg, series, bookings=NO_BOOK), tmp_path)
    calls = []
    train_exp(ctx, cfg.defaults(), log=calls.append)
    n = len(calls)
    train_exp(ctx, cfg.defaults(), log=calls.append)
    assert len(calls) == n                     # 두 번째는 학습 로그 없음(건너뜀)


def test_failed_run_does_not_stop(fcfg, tmp_path, monkeypatch):
    series = synth_series()
    ctx = Context(fcfg, CPU, {"series": series, "eval": None, "norm": synth_norm(series)},
                  build_arrays(fcfg, series, bookings=NO_BOOK), tmp_path)
    import forecast.experiments as ex
    monkeypatch.setattr(ex, "train_one", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("메모리 부족")))
    recs = train_exp(ctx, replace(fcfg.defaults(), seeds=(0, 1)), log=lambda *_: None)
    assert [r["status"] for r in recs] == ["failed", "failed"]


def test_stage_eval_override_and_b1_includes_prior(fcfg):
    """D3은 비교만 2시간 입력으로 하고, B1 후보에는 앞 단계 선택(예: 과거 하루)이 들어간다."""
    base = replace(fcfg.defaults(), lookback_min=1440)
    d3 = fcfg.stage_candidates("D3", base)
    assert {c.lookback_min for c in d3} == {120} and {c.output for c in d3} == {"direct", "recursive"}
    b1 = fcfg.stage_candidates("B1", base)
    assert base.hash in {c.hash for c in b1} and len(b1) == 10


def test_d3_pick_keeps_prior_lookback(fcfg, tmp_path):
    series = synth_series()
    cfg, ev = small_cfg(fcfg, tmp_path, series)
    cfg.raw["stage_eval"] = {"D3": {"lookback_min": 30}}
    ctx = ctx_for(cfg, series, ev, tmp_path / "fc")
    for st in ("D0", "D1", "D2", "D3"):
        run_stage(ctx, st, log=lambda *_: None)
    s = load_stages(ctx)["D3"]
    assert s["picked"]["lookback_min"] == 60                   # 앞 단계(D1) 선택 유지
    assert all(c["key"]["lookback_min"] == 30 for c in s["candidates"])

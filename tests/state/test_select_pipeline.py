"""임계값 규칙, 선택 고정·평가 거부, 누수(시험 기간을 바꿔도 선택·임계값·모델 같음), 전체 흐름·묶음, 심화 규칙."""
import json

import numpy as np
import pandas as pd
import pytest

from state.analyze import noshow_unbooked, transition_errors
from state.pipeline import Data, load_bundle, run_analyze, run_evaluate, run_features, run_labels, run_select, run_train
from state.selection import SelectionMissingError, metrics, occ_threshold, pick, require_selection, vent_threshold
from tests.state.conftest import synth_norm, synth_obs, synth_space

pytestmark = pytest.mark.unit
QUIET = lambda *_: None


def test_vent_threshold_rule():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.75, 0.3, 0.1, 0.1, 0.05, 0.6])
    thr, ok, m = vent_threshold(y, p, 0.75)
    assert ok and m["recall"] >= 0.75
    assert thr == pytest.approx(0.7, abs=1e-9) or m["precision"] >= 0.6
    thr2, ok2, m2 = vent_threshold(y, p, 1.0)            # 재현율 100%는 0.2까지 내려야 함
    assert ok2 and m2["recall"] == 1.0 and thr2 <= 0.2


def test_metrics_and_always_negative():
    m = metrics([1, 0, 0, 0], [1, 1, 0, 0])
    assert m["recall"] == 1 and m["precision"] == 0.5 and m["false_occupied_rate"] == pytest.approx(1 / 3)
    assert m["always_negative_accuracy"] == 0.75


def test_pick_prefers_simpler_within_1sd():
    i, why = pick([{"mean": 0.80, "sd": 0.02, "simplicity": 1}, {"mean": 0.79, "sd": 0.01, "simplicity": 0}])
    assert i == 1 and "단순" in why


def _flow(scfg, series, root, obs_rows=None, norm=None):
    norm = norm or synth_norm(series)     # 실제로는 과제 2 학습 기간 통계 파일(시험 기간과 무관)
    if obs_rows is not None:
        synth_obs(scfg["inputs"]["observations"], obs_rows)
    from state.labels import load_observations
    data = Data(scfg, root, series=series, norm=norm, bookings=None, obs=load_observations(scfg))
    run_labels(scfg, data, QUIET)
    run_features(scfg, data, QUIET)
    run_train(scfg, "all", "cpu", root, QUIET)
    return data


def test_gate_flow_bundle_and_leakage(scfg, tmp_path):
    s1 = synth_space()
    d1 = _flow(scfg, s1, tmp_path / "a", obs_rows=[["lab", "2026-03-25 08:30", "2026-03-25 12:00", "occupied", "20", "KW", ""],
                                                   ["lab", "2026-03-25 12:00", "2026-03-25 13:30", "empty", "", "KW", ""]])
    with pytest.raises(SelectionMissingError):
        run_evaluate(scfg, "cpu", d1, QUIET)
    sel1 = run_select(scfg, tmp_path / "a", QUIET)
    res = run_evaluate(scfg, "cpu", d1, QUIET)
    assert [c["판정"] for c in res["criteria"]][-1] == "⏸ 판단 보류"          # 관찰 목표 미달
    assert res["occupancy"]["observation_rows"] > 0
    assert res["vent"]["forecast_model"]["status"] in ("skipped", "done")
    b = load_bundle(tmp_path / "a" / "bundle")
    assert b["input_spec_version"] == 1 and set(b["problems"]) == {"occupancy", "vent"}
    assert not [c for c in b["problems"]["occupancy"]["features"] if c.startswith(("t_", "sch_"))]
    # 과제 8 진입점: 서비스가 넘기는 공간 5분 표로 판정
    from types import SimpleNamespace
    from state.serve import load as serve_load
    svc = serve_load(tmp_path / "a" / "bundle")
    t0 = pd.Timestamp("2026-03-25 03:00", tz="UTC")
    win = s1[(s1["bin"] > t0 - pd.Timedelta(hours=2)) & (s1["bin"] <= t0)]
    out = svc.predict(SimpleNamespace(space=win, t0_bin=t0))
    assert "lab" in out and 0 <= out["lab"]["p_occupied"] <= 1 and isinstance(out["lab"]["ventilation_soon"], bool)
    run_analyze(scfg, d1, QUIET)
    from state.report import write_reports
    files = write_reports(scfg, tmp_path / "a", tmp_path / "deliv", tmp_path / "adv")
    assert all(f.read_text().startswith("# 과제 5") for f in files)
    # 누수: 시험 기간 값을 바꾸면 시험 결과는 달라져도 선택·임계값·모델은 같다
    s2 = s1.copy()
    a, _ = scfg.split.utc("test")
    m = s2["bin"] >= pd.Timestamp(a)
    s2.loc[m & (s2["item"] == "co2"), "value"] *= 3
    _flow(scfg, s2, tmp_path / "b", norm=synth_norm(s1))
    sel2 = run_select(scfg, tmp_path / "b", QUIET)
    for prob in ("occupancy", "vent"):
        assert sel1[prob]["candidate"] == sel2[prob]["candidate"]
        assert sel1[prob]["threshold"] == sel2[prob]["threshold"]
        assert sel1[prob]["model_sha256"] == sel2[prob]["model_sha256"]
    # 고정 뒤 변조 → 거부
    p = tmp_path / "a" / "selection.json"
    body = json.loads(p.read_text())
    body["vent"]["threshold"] = 0.0
    p.write_text(json.dumps(body))
    with pytest.raises(SelectionMissingError):
        require_selection(p, tmp_path / "a" / "runs")


def test_transition_errors():
    t = pd.date_range("2026-03-03 00:00", periods=8, freq="5min", tz="UTC")
    df = pd.DataFrame({"space_id": "lab", "bin": t, "y": [0, 0, 1, 1, 1, 1, 0, 0], "pred": [0, 0, 0, 1, 1, 1, 1, 0]})
    e = transition_errors(df, "y", "pred").set_index(["전환", "경과(분)"])
    assert e.loc[("입실", "0~5"), "오류율"] == 1.0
    assert e.loc[("퇴실", "0~5"), "오류율"] == 1.0


def test_noshow_and_unbooked(scfg):
    t = pd.date_range("2026-03-03 05:00", periods=12, freq="5min", tz="UTC")
    pred = pd.DataFrame({"space_id": "meeting", "bin": t, "pred": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]})
    bk = pd.DataFrame({"space_id": ["meeting"], "start": [t[0]], "end": [t[4]]})
    ns = noshow_unbooked(pred, bk, scfg)
    assert set(ns["kind"]) == {"노쇼", "예약 없는 사용"}

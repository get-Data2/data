"""합성 데이터로 inject → train → score → threshold → evaluate 전체 흐름, 임계값 고정 전 평가 거부, 임계값 누수."""
import json

import pandas as pd
import pytest
import torch

from anomaly.alarms import ThresholdMissingError
from anomaly.pipeline import Work, run_evaluate, run_inject, run_score, run_threshold, run_train

pytestmark = pytest.mark.unit
CPU = torch.device("cpu")


@pytest.fixture
def work(ncfg, sensors, tmp_path):
    raw = ncfg.raw
    raw["autoencoder"]["train"]["max_epochs"] = 2
    raw["alarms"]["q_candidates"]["n"] = 6
    ev = pd.DataFrame({"space_id": pd.Series(dtype=str), "kind": pd.Series(dtype=str),
                       "start": pd.Series(dtype="datetime64[ns, UTC]"), "end": pd.Series(dtype="datetime64[ns, UTC]")})
    ev.to_parquet(tmp_path / "events.parquet")
    pd.DataFrame({"space_id": pd.Series(dtype=str), "device_id": pd.Series(dtype=str), "item": pd.Series(dtype=str),
                  "start": pd.Series(dtype="datetime64[ns, UTC]"), "end": pd.Series(dtype="datetime64[ns, UTC]")}) \
        .to_parquet(tmp_path / "stuck.parquet")
    raw["inputs"] = dict(raw["inputs"], events=str(tmp_path / "events.parquet"), stuck=str(tmp_path / "stuck.parquet"))
    return Work(ncfg, tmp_path / "an", sensor=sensors)


def test_full_flow_and_gates(work):
    log = lambda *_: None
    faults = run_inject(work, log)
    assert len(faults) == 10
    run_train(work, CPU, log)
    run_score(work, CPU, log=log)
    with pytest.raises(ThresholdMissingError):
        run_evaluate(work, log)
    t1 = run_threshold(work, log)
    res = run_evaluate(work, log)
    assert {"model", "rules", "combined"} <= set(res["groups"])
    card = pd.DataFrame(res["groups"]["combined"]["scorecard"])
    assert list(card["kind"]) == ["F1", "F2", "F3", "F4", "F5", "전체"]
    assert res["curve"] and any(r["chosen"] for r in res["curve"])
    from anomaly.bundle import save_bundle
    from anomaly.report import write_reports
    b = save_bundle(work, log)
    assert (b / "bundle.json").exists() and list(b.glob("ae_*.pt"))
    files = write_reports(work, work.root / "deliv", work.root / "adv")
    assert len(files) >= 4 and all(f.read_text().startswith("# 과제 4") for f in files)
    # 임계값 누수 없음: 평가판 점수를 망가뜨리고 다시 정해도 같다
    p = work.root / "scores" / "ae__test.parquet"
    df = pd.read_parquet(p)
    df["score"] = df["score"] * 1000
    df.to_parquet(p, index=False)
    t2 = run_threshold(work, log)
    for d in t1["detectors"]:
        assert t1["detectors"][d]["threshold_q"] == t2["detectors"][d]["threshold_q"]

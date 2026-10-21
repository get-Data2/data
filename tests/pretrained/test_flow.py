"""가짜 모델로 전체 흐름: 검증 예측 → 선택 고정 → 시험(거부·허용) → 표. 선택은 검증만 본다."""
import json

import numpy as np
import pandas as pd
import pytest

from pretrained.pipeline import run_predict, run_select, run_test, eval_frames
from pretrained.predict import pred_path
from pretrained.selection import SelectionMissingError, choose, ensemble_weights, require

pytestmark = pytest.mark.unit
quiet = lambda *_: None


def test_flow_gate_and_selection_uses_validation_only(pcfg, fake, tmp_path):
    root = tmp_path / "pt"
    with pytest.raises(SelectionMissingError):
        run_test(pcfg, "cpu", factory=lambda n: fake, root=root, log=quiet)
    run_predict(pcfg, "cpu", "val", ["bolt_tiny", "bolt_small"], factory=lambda n: fake, root=root, log=quiet)
    s1 = run_select(pcfg, root=root, log=quiet)
    assert s1["chosen"]["model"] == "bolt_tiny" and s1["chosen"]["ctx_min"] == 60      # 같은 예측 → 작은 모델·짧은 컨텍스트
    run_test(pcfg, "cpu", factory=lambda n: fake, root=root, log=quiet)
    res = json.loads((root / "results.json").read_text())
    assert all(res["rows_check"].values())
    fr = eval_frames(pcfg, root)
    assert {"bolt_tiny", "bolt_small"} <= set(fr)
    from pretrained.report import write_reports
    files = write_reports(pcfg, tmp_path / "deliv", tmp_path / "adv", root)
    assert files and files[0].read_text().startswith("# 과제 6") and "적중률" in files[0].read_text()
    # 시험 예측을 망가뜨려도 다시 고른 선택은 같다(검증만 본다)
    p = pred_path("bolt_tiny", 60, "test", root=root)
    d = pd.read_parquet(p)
    d["y_pred"] = d["y_pred"] * 100
    d.to_parquet(p, index=False)
    s2 = run_select(pcfg, root=root, log=quiet)
    assert s1["chosen"] == s2["chosen"] and s1["grid"] == s2["grid"]
    # 고정 뒤 변조 → 거부
    sp = root / "selection.json"
    b = json.loads(sp.read_text())
    b["chosen"]["ctx_min"] = 240
    sp.write_text(json.dumps(b))
    with pytest.raises(SelectionMissingError, match="해시"):
        require(root)


def test_tie_rule(pcfg):
    rows = [{"model": "bolt_small", "ctx_min": 240, "val_norm_mae": 0.100},
            {"model": "bolt_tiny", "ctx_min": 1440, "val_norm_mae": 0.1005},
            {"model": "bolt_mini", "ctx_min": 60, "val_norm_mae": 0.12}]
    pick, why = choose(pcfg, rows)
    assert pick["model"] == "bolt_tiny" and "안" in why


def test_ensemble_weight_by_validation(pcfg):
    t0 = pd.date_range("2026-03-18", periods=200, freq="5min", tz="UTC")
    y = np.linspace(400, 800, 200)
    d = pd.DataFrame({"space_id": "lab", "t0": t0, "horizon_min": 60, "y_pred": y + 10})
    p = pd.DataFrame({"space_id": "lab", "t0": t0, "horizon_min": 60, "y_pred": y - 10, "y_true": y})
    e = ensemble_weights(pcfg, d, p)
    assert e["w_global"] == pytest.approx(0.5) and e["val_mae_global"] == pytest.approx(0.0, abs=1e-6)


def test_covariate_path(pcfg, fake, tmp_path):
    root = tmp_path / "pt"
    run_predict(pcfg, "cpu", "val", ["chronos2"], [60], cov=True, factory=lambda n: fake, root=root, log=quiet)
    assert pred_path("chronos2", 60, "val", True, root).exists()

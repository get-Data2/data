import numpy as np
import pandas as pd
import pytest

from pretrained.bench import latency, parse_tegrastats
from pretrained.context import contexts, grid, load_series, eval_keys, val_t0
from pretrained.metrics import comparison_table, coverage_table

pytestmark = pytest.mark.unit
LINE = ("10-21-2026 14:53:52 RAM 4381/7486MB (lfb 63x4MB) CPU [3%@960,50%@960,21%@960,3%@960,5%@729,4%@729] "
        "GR3D_FREQ 81% cpu@61.218C/61.218C soc2@59.562C/59.562C soc0@59.375C/59.375C gpu@61.343C/61.343C "
        "tj@61.343C/61.343C soc1@59.312C/59.312C VDD_IN 15231mW/15231mW/15231mW VDD_CPU_GPU_CV 5903mW/5903mW/5903mW "
        "VDD_SOC 4185mW/4185mW/4185mW")


def test_context_is_causal_and_ignores_interpolated(pcfg):
    s = load_series(pcfg)
    t0s = val_t0(pcfg, s, "lab")[:50]
    g = grid(s, "lab", "co2")
    c1, _ = contexts(g, t0s, 12)
    s2 = s.copy()
    s2.loc[s2["bin"] > t0s[-1], "value"] *= 10
    c2, _ = contexts(grid(s2, "lab", "co2"), t0s, 12)
    np.testing.assert_array_equal(c1, c2)
    s3 = s.copy()
    m = (s3["space_id"] == "lab") & (s3["item"] == "co2") & (s3["bin"] == t0s[0] - pd.Timedelta(minutes=5))
    s3.loc[m, "filled"] = True                                  # 과제 2 보간 칸 흉내 → 컨텍스트에서 NaN
    c3, _ = contexts(grid(s3, "lab", "co2"), t0s[:1], 12)
    assert np.isnan(c3[0, -2]) and not np.isnan(c3[0, -1])


def test_unpredictable_ratio_and_meeting_nans(pcfg):
    s = load_series(pcfg)
    t0s = val_t0(pcfg, s, "meeting")[:5]
    c, r = contexts(grid(s, "meeting", "co2"), t0s, 12)
    assert np.isnan(c).sum(1).min() >= 5 and (r > 0.4).all()
    g = grid(s, "lab", "co2")
    g2 = g.copy()
    t0 = val_t0(pcfg, s, "lab")[100]
    g2[(g2.index > t0 - pd.Timedelta(hours=1)) & (g2.index < t0)] = np.nan
    _, r2 = contexts(g2, [t0], 12)
    assert r2[0] < pcfg["min_context_obs"]


def test_coverage_by_hand():
    df = pd.DataFrame({"space_id": "lab", "horizon_min": 60, "y_true": [10, 20, 30, 40], "p10": [5, 25, 25, 25],
                       "p90": [15, 35, 35, 35], "unpredictable": [False, False, False, True], "usage_hours": True})
    c = coverage_table(df, min_points=1).iloc[0]
    assert c["n"] == 3 and c["예측 불가"] == 1
    assert c["적중률"] == pytest.approx(2 / 3) and c["아래 이탈"] == pytest.approx(1 / 3) and c["평균 폭"] == 10


def test_comparison_counts_match_baseline(pcfg, fake):
    from pretrained.pipeline import run_predict
    from pretrained.predict import pred_path
    import tempfile, pathlib
    root = pathlib.Path(tempfile.mkdtemp())
    run_predict(pcfg, "cpu", "test", ["bolt_tiny"], [60], factory=lambda n: fake, root=root, log=lambda *_: None)
    df = pd.read_parquet(pred_path("bolt_tiny", 60, "test", root=root))
    ev = pd.read_parquet(pcfg["inputs"]["eval_points"])
    ev = ev[ev["item"] == "co2"]
    assert len(df) == len(ev)                                   # 평가 시점 CO₂ 행과 행 단위 일치
    k = ["space_id", "t0", "horizon_min"]
    assert len(df[k].merge(ev[k].assign(t0=pd.DatetimeIndex(ev["t0"]).as_unit("ns")))) == len(ev)
    t = comparison_table(pcfg, {"bolt_tiny": df})
    assert (t["n"] > 0).all() and "bolt_tiny" in t.columns


def test_tegrastats_parse_and_latency():
    d = parse_tegrastats(LINE)
    assert d["ram_used_mb"] == 4381 and d["gr3d_pct"] == 81 and d["vdd_in_mw"] == 15231 and d["vdd_cpu_gpu_cv_mw"] == 5903
    n = []
    lat = latency(lambda: n.append(1), 3, 7)
    assert len(n) == 10 and lat["repeats"] == 7 and lat["p95"] >= lat["p50"]

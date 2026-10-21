from datetime import datetime

import pandas as pd
import pytest

from edgeopt.tegra import parse_line, parse_log, summarize

pytestmark = pytest.mark.unit

# research 부록: 2026-10-21 14:53 실제 출력
L1 = ("10-21-2026 14:53:52 RAM 4381/7486MB (lfb 63x4MB) CPU [3%@960,50%@960,21%@960,3%@960,5%@729,4%@729] GR3D_FREQ 81% "
      "cpu@61.218C/61.218C soc2@59.562C/59.562C soc0@59.375C/59.375C gpu@61.343C/61.343C tj@61.343C/61.343C "
      "soc1@59.312C/59.312C VDD_IN 15231mW/15231mW/15231mW VDD_CPU_GPU_CV 5903mW/5903mW/5903mW VDD_SOC 4185mW/4185mW/4185mW")
L2 = ("10-21-2026 14:53:53 RAM 4378/7486MB (lfb 63x4MB) CPU [24%@729,49%@729,1%@729,2%@729,0%@729,3%@729] GR3D_FREQ 81% "
      "cpu@60.906C/61.218C soc2@59.562C/59.562C soc0@59.5C/59.5C gpu@61.406C/61.406C tj@61.406C/61.406C "
      "soc1@59.437C/59.437C VDD_IN 15154mW/15193mW/15231mW VDD_CPU_GPU_CV 5865mW/5884mW/5903mW VDD_SOC 4185mW/4185mW/4185mW")


def test_parse_real_lines():
    r = parse_line(L1)
    assert r["t"] == datetime(2026, 10, 21, 14, 53, 52)
    assert (r["ram_used_mb"], r["ram_total_mb"], r["gr3d_pct"]) == (4381, 7486, 81)
    assert (r["vdd_in_mw"], r["vdd_cpu_gpu_cv_mw"], r["vdd_soc_mw"]) == (15231, 5903, 4185)
    assert r["temp_cpu_c"] == 61.218 and r["temp_gpu_c"] == 61.343 and r["temp_tj_c"] == 61.343
    assert r["cpu_mean_pct"] == pytest.approx((3 + 50 + 21 + 3 + 5 + 4) / 6)
    r2 = parse_line(L2)
    assert r2["vdd_in_mw"] == 15154 and r2["temp_cpu_c"] == 60.906        # 현재값(평균·최대가 아님)
    assert parse_line("garbage") is None


def test_summary_by_hand():
    """대기 10,000mW, 측정 12,000mW, 10초 동안 1,000회 → 증가분 2,000mW, 1회당 2,000 × 10 / 1000 = 20mJ."""
    t = [datetime(2026, 10, 21, 15, 0, s) for s in range(20)]
    df = pd.DataFrame({"t": t, "vdd_in_mw": [10000] * 10 + [12000] * 10, "gr3d_pct": [2] * 10 + [90] * 10,
                       "ram_used_mb": [4000] * 10 + [4300] * 10})
    s = summarize(df, (t[0], t[9]), (t[10], t[19]), 1000)
    assert s["delta_vdd_in_mw"] == 2000 and s["run_seconds"] == 9
    assert s["energy_mj_per_call"] == pytest.approx(2000 * 9 / 1000)
    assert s["idle_gr3d_mean"] == 2 and s["run_gr3d_mean"] == 90 and s["ram_run_max_mb"] == 4300
    assert len(parse_log(L1 + "\n" + L2 + "\nnoise\n")) == 2

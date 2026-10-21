"""기준선 손 계산 일치, 같은 시점 집합(SC-004), 보간 칸 제외, 사용 시간대(공휴일)."""
import numpy as np
import pandas as pd
import pytest

from analysis.baseline import build, mae_table, usage_mask

pytestmark = pytest.mark.unit


def make_ss(acfg, filled_at=None):
    sp = acfg.split("v1-provisional")
    a, b = sp.utc("test")
    bins = pd.date_range(a - pd.Timedelta(days=2), b + pd.Timedelta(hours=2), freq="5min")
    i = np.arange(len(bins))
    val = 600 + 200 * np.sin(i / 288 * 2 * np.pi) + 3 * i % 7
    ss = pd.DataFrame({"bin": bins, "space_id": "lab", "item": "co2", "value": val.astype(float),
                       "n_sensors": 2, "n_observed": 2, "filled": False})
    if filled_at is not None:
        ss.loc[ss["bin"] == filled_at, "filled"] = True
    return sp, ss


def test_hand_calculation(acfg):
    sp, ss = make_ss(acfg)
    res = build(ss, sp, acfg)
    y = ss.set_index("bin")["value"]
    r = res[(res["horizon_min"] == 30)].iloc[100]
    t0 = r["t0"]
    assert r["persistence"] == y[t0]
    assert r["mean_1h"] == pytest.approx(y[t0 - pd.Timedelta(minutes=55):t0].mean())
    assert r["yesterday"] == y[t0 + pd.Timedelta(minutes=30) - pd.Timedelta(days=1)]
    assert r["y_true"] == y[t0 + pd.Timedelta(minutes=30)]


def test_same_points_for_all_methods_and_inside_test(acfg):
    sp, ss = make_ss(acfg)
    res = build(ss, sp, acfg)
    a, b = sp.utc("test")
    assert res[["persistence", "mean_1h", "yesterday", "y_true"]].notna().all().all()   # 한 행 = 세 방법 공통 시점
    assert (res["t0"] >= a).all() and ((res["t0"] + pd.to_timedelta(res["horizon_min"], unit="min")) < b).all()
    t = mae_table(res)
    assert t["평가 시점 수"].nunique() >= 1 and set(t["지평(분)"]) == {5, 15, 30, 60}


def test_filled_bin_not_used(acfg):
    sp, ss0 = make_ss(acfg)
    a, _ = sp.utc("test")
    t_fill = a + pd.Timedelta(hours=10)
    _, ss = make_ss(acfg, filled_at=t_fill)
    res = build(ss, sp, acfg)
    five = res[res["horizon_min"] == 5]
    assert t_fill not in set(five["t0"])                                   # 입력이 보간 칸이면 시점 제외
    assert t_fill - pd.Timedelta(minutes=5) not in set(five["t0"])         # 정답이 보간 칸이면 시점 제외


def test_usage_hours_excludes_holiday():
    t = pd.DatetimeIndex(["2026-10-09 01:00", "2026-10-08 01:00", "2026-10-10 01:00", "2026-10-08 12:00"], tz="UTC")
    # KST: 10-09 10시(한글날) / 10-08 10시(목) / 10-10 10시(토) / 10-08 21시(목, 시간 밖)
    from analysis.config import load_analysis_config
    from tests.conftest import ROOT
    cfg = load_analysis_config(ROOT / "config" / "analysis.yaml", None)
    assert list(usage_mask(t, cfg)) == [False, True, False, False]

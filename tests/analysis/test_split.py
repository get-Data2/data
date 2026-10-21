"""누수 방지(SC-002): 시험 기간 값을 바꿔도 학습 기간 통계가 바뀌지 않는다."""
import numpy as np
import pandas as pd
import pytest

from analysis.split import autocorr, fit_train_stats

pytestmark = pytest.mark.unit


def synthetic(acfg):
    sp = acfg.split("v1-provisional")
    rng = np.random.default_rng(0)
    bins = pd.date_range("2025-09-01", "2026-07-31", freq="60min", tz="UTC")   # 시험용으로 1시간 칸
    ss = pd.DataFrame({"bin": bins, "space_id": "lab", "item": "co2",
                       "value": 500 + 100 * np.sin(np.arange(len(bins)) / 24 * 2 * np.pi) + rng.normal(0, 10, len(bins)),
                       "n_sensors": 2, "n_observed": 2, "filled": False})
    sb = pd.concat([ss.assign(device_id="a", value=ss.value + 5), ss.assign(device_id="b", value=ss.value - 5)])
    return sp, ss, sb


def test_test_period_changes_do_not_leak(acfg):
    sp, ss, sb = synthetic(acfg)
    base = fit_train_stats(ss, sb, sp, acfg)
    a, b = sp.utc("test")
    for df in (ss, sb):
        m = (df["bin"] >= a) & (df["bin"] < b)
        df.loc[m, "value"] = df.loc[m, "value"] * 10
    a2, b2 = sp.utc("val")
    m = (ss["bin"] >= a2) & (ss["bin"] < b2)
    ss.loc[m, "value"] = -999
    after = fit_train_stats(ss, sb, sp, acfg)
    assert base == after
    assert base["spaces"]["lab"]["co2"]["bins"] > 0 and "mad" in base["spaces"]["lab"]["co2"]


def test_train_window_excludes_other_parts(acfg):
    sp, ss, sb = synthetic(acfg)
    st = fit_train_stats(ss, sb, sp, acfg)
    a, b = sp.utc("train")
    assert st["spaces"]["lab"]["co2"]["bins"] == int(((ss.bin >= a) & (ss.bin < b)).sum())


def test_autocorr_daily_cycle():
    x = pd.Series(np.sin(np.arange(288 * 20) / 288 * 2 * np.pi))
    r = autocorr(x, (1, 144, 288))
    assert r[1] > 0.99 and r[288] > 0.99 and r[144] < -0.99

import numpy as np
import pandas as pd
import pytest

from analysis.config import KST
from anomaly.data import split_col
from anomaly.inject import inject

pytestmark = pytest.mark.unit


def test_reproducible_and_counts(ncfg, boards):
    _, f1, _ = inject(boards, ncfg)
    _, f2, _ = inject(boards, ncfg)
    pd.testing.assert_frame_equal(f1, f2)
    assert f1["kind"].value_counts().to_dict() == {k: 2 for k in ("F1", "F2", "F3", "F4", "F5")}


def test_untouched_outside_faults(ncfg, boards):
    out, faults, fid = inject(boards, ncfg)
    for s in boards:
        m = fid[s].to_numpy() == 0
        a, b = boards[s].to_numpy()[m], out[s].to_numpy()[m]
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        np.testing.assert_array_equal(a[~np.isnan(a)], b[~np.isnan(b)])
        # 결측 모양 유지
        np.testing.assert_array_equal(boards[s].isna().to_numpy(), out[s].isna().to_numpy())


def test_stuck_value_in_normal_range(ncfg, boards):
    out, faults, fid = inject(boards, ncfg)
    for f in faults[faults["kind"] == "F2"].itertuples():
        w = out[f.space_id]
        for c in [c for c in w.columns if split_col(c)[0] == f.device_id]:
            v = w.loc[fid[f.space_id][c] == f.fault_id, c]
            assert v.nunique() == 1
            orig = boards[f.space_id][c].dropna()
            assert orig.min() <= v.iloc[0] <= orig.max()


def test_equipment_faults_in_usage_hours_and_gaps(ncfg, boards):
    _, faults, _ = inject(boards, ncfg)
    for f in faults[faults["kind"].isin(["F4", "F5"])].itertuples():
        k = f.start.tz_convert(KST)
        assert k.weekday() < 5 and 9 <= k.hour < 18
    for s, g in faults.groupby("space_id"):
        g = g.sort_values("start")
        gaps = (g["start"].iloc[1:].to_numpy() - g["end"].iloc[:-1].to_numpy())
        assert (gaps >= np.timedelta64(12, "h")).all()


def test_ventilation_factor(ncfg, boards):
    out, faults, fid = inject(boards, ncfg)
    f = faults[faults["kind"] == "F5"].iloc[0]
    c = [c for c in boards[f.space_id].columns if c.endswith("/co2")][0]
    m = fid[f.space_id][c] == f.fault_id
    before, after = boards[f.space_id].loc[m, c], out[f.space_id].loc[m, c]
    assert (after >= before - 1e-9).all() and (after > before).any()

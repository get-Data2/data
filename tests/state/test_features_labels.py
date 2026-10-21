import numpy as np
import pandas as pd
import pytest

from state.config import StateConfigError, validate
from state.features import build_features, columns_for, space_features, space_grid
from state.labels import observation_labels, occupancy_schedule, vent_labels
from tests.state.conftest import synth_norm

pytestmark = pytest.mark.unit


def test_features_are_causal(scfg, space_series):
    norm = synth_norm(space_series)
    w = space_grid(space_series, "lab")
    f1 = space_features(w, "lab", norm, scfg.acfg)
    i = 3000
    w2 = w.copy()
    w2.iloc[i + 1:] = w2.iloc[i + 1:] * 3 + 100
    w2.iloc[i + 1: i + 50] = np.nan
    f2 = space_features(w2, "lab", norm, scfg.acfg)
    cols = [c for c in f1.columns if c not in ("space_id", "bin")]
    pd.testing.assert_frame_equal(f1.loc[:i, cols], f2.loc[:i, cols])


def test_occupancy_group_has_no_time_or_schedule(scfg):
    cols = columns_for(scfg["features"]["groups"]["occupancy"])
    assert not [c for c in cols if c.startswith(("t_", "sch_"))]
    assert [c for c in columns_for(scfg["features"]["groups"]["leak"]) if c.startswith("sch_")]
    raw = dict(scfg.raw, features=dict(scfg["features"], groups=dict(scfg["features"]["groups"], occupancy=["level", "time"])))
    errs, _ = validate(raw, scfg.acfg)
    assert any("금지" in e for e in errs)


def test_meeting_delta_uses_observed_bins(scfg, space_series):
    """10분 주기 공간: 빈 칸은 직전 관측으로 채워 5분 변화량이 관측 사이에서 0, 관측 칸에서 실제 변화."""
    norm = synth_norm(space_series)
    w = space_grid(space_series, "meeting")
    f = space_features(w, "meeting", norm, scfg.acfg)
    obs = w["co2"].notna().to_numpy()
    d5 = f["d5_co2"].to_numpy()
    assert np.allclose(d5[1:][~obs[1:]], 0, equal_nan=True)


def test_vent_label_rules():
    t = pd.date_range("2026-03-03 00:00", periods=12, freq="5min", tz="UTC")
    co2 = pd.Series([999, 990, 995, 1001, 1010, 980, 970, 960, 950, 940, 930, 920.0], index=t)
    v = vent_labels(co2, 1000, 30, 0.5).set_index("bin")
    assert v.loc[t[0], "label"] == "positive" and v.loc[t[0], "first_exceed"] == t[3]
    assert v.loc[t[3], "label"] == "exceeded"
    # 31분 뒤 초과는 음성: t[0]에서 7칸 뒤 값만 1000 이상인 계열
    co2b = pd.Series([900] * 7 + [1100] + [900] * 4, index=t, dtype=float)
    vb = vent_labels(co2b, 1000, 30, 0.5).set_index("bin")
    assert vb.loc[t[0], "label"] == "negative" and vb.loc[t[1], "label"] == "positive"
    # 미래 관측 부족
    co2c = co2b.copy()
    co2c.iloc[1:7] = np.nan
    assert vent_labels(co2c, 1000, 30, 0.5).set_index("bin").loc[t[0], "label"] == "undetermined"


def test_holiday_schedule_is_empty_and_meeting_none(scfg):
    t = pd.date_range("2026-03-02 01:00", periods=3, freq="5min", tz="UTC")       # 3·1절 대체공휴일 10:00 KST
    lab, src = occupancy_schedule(t, "lab", scfg, None)
    assert src == "timetable" and (lab == 0).all()
    t2 = pd.date_range("2026-03-03 01:00", periods=3, freq="5min", tz="UTC")      # 화 10:00 KST
    assert (occupancy_schedule(t2, "lab", scfg, None)[0] == 1).all()
    m, msrc = occupancy_schedule(t2, "meeting", scfg, None)
    assert msrc == "none" and np.isnan(m).all()


def test_observation_majority():
    t = pd.date_range("2026-10-22 00:00", periods=4, freq="5min", tz="UTC")       # KST 09:00~09:20
    obs = pd.DataFrame({"space_id": ["lab", "lab"],
                        "start": pd.to_datetime(["2026-10-22 09:00", "2026-10-22 09:07"]),
                        "end": pd.to_datetime(["2026-10-22 09:07", "2026-10-22 09:12"]),
                        "state": ["empty", "occupied"], "people": [None, 20], "note": ["", ""]})
    lab, ppl = observation_labels(t, "lab", obs)
    assert lab[0] == 0                      # 09:00~05 비어있음
    assert lab[1] == 1 and ppl[1] == 20     # 09:05~10: 2분 빈, 3분 사용 → 사용중
    assert lab[2] == 0 or np.isnan(lab[2])  # 09:10~15: 덮은 비율 0.4 → 판단 없음
    assert np.isnan(lab[3])


def test_build_features_split(scfg, space_series):
    f = build_features(space_series, synth_norm(space_series), scfg)
    assert set(f["split"]) >= {"train", "val", "test"}
    assert f.groupby("split")["bin"].min()["val"] > f.groupby("split")["bin"].max()["train"]

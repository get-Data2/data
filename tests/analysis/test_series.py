from datetime import timedelta

import pytest

from analysis.quality import apply_axis_a
from analysis.series import sensor_bins, space_series
from tests.analysis.conftest import T0

pytestmark = pytest.mark.unit


def rows_for(dev, space, values, item="co2", step_s=60):
    return [(T0 + timedelta(seconds=step_s * i), space, dev, item, v) for i, v in enumerate(values)]


def test_space_median_and_counts(acfg, make_raw):
    # 5분 칸 2개(10분) — 센서 3대, 한 대는 오류 코드 섞임, 복도 센서는 대표값에서 제외
    raw = make_raw(rows_for("a", "lab", [500] * 10) + rows_for("b", "lab", [600] * 10)
                   + rows_for("c", "lab", [700] * 5 + [65535] * 5)
                   + rows_for("24e124136d151523", "lab", [9999] * 10))
    q = apply_axis_a(raw, acfg)
    sb = sensor_bins(raw, q, acfg)
    ss = space_series(sb, acfg)
    first, second = ss.sort_values("bin").to_dict("records")
    assert first["value"] == 600 and first["n_sensors"] == 3            # 중앙값(500, 600, 700), 복도 제외
    assert second["value"] == 550 and second["n_sensors"] == 2          # c의 65535는 error라 빠짐
    assert not first["filled"]


def test_saturated_kept(acfg, make_raw):
    raw = make_raw(rows_for("a", "lab", [10000] * 5))
    q = apply_axis_a(raw, acfg)
    sb = sensor_bins(raw, q, acfg)
    assert sb["value"].iloc[0] == 10000 and bool(sb["saturated"].iloc[0])


def test_filled_flag_only_when_no_observation(acfg, make_raw):
    # a는 계속, b는 칸 1개 빔(보간) → 공간 칸은 a가 관측했으므로 filled=False
    a = rows_for("a", "lab", [500] * 20)
    b = [r for i, r in enumerate(rows_for("b", "lab", [600] * 20)) if not 5 <= i < 10]
    raw = make_raw(a + b)
    sb = sensor_bins(raw, apply_axis_a(raw, acfg), acfg)
    assert sb[(sb.device_id == "b")]["filled"].sum() == 1
    ss = space_series(sb, acfg)
    assert not ss["filled"].any()

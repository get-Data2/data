import numpy as np
import pandas as pd
import pytest

from forecast.data import build_arrays, input_obs_ratio, norm_table, split_t0, eval_samples

pytestmark = pytest.mark.unit
NO_BOOK = pd.DataFrame(columns=["space_id", "start", "end", "source"])


def arrays(fcfg, series):
    return build_arrays(fcfg, series, bookings=NO_BOOK)


def test_inputs_are_causal(fcfg, series):
    """칸 i 이후 값을 바꿔도 i까지의 입력(값·관측 표시·경과 시간)은 같다."""
    a = arrays(fcfg, series)
    i = 4000
    s2 = series.copy()
    cut = a.t[i]
    s2.loc[s2["bin"] > cut, "value"] *= 10
    s2.loc[(s2["bin"] > cut) & (s2["space_id"] == "lab"), "filled"] = True
    b = arrays(fcfg, s2)
    for f in ("raw_ffill", "obs", "age_h"):
        np.testing.assert_array_equal(getattr(a, f)[:, :i + 1], getattr(b, f)[:, :i + 1])


def test_interpolated_bins_not_used(fcfg, series):
    a0 = arrays(fcfg, series)
    g0 = a0.t[3000]
    from tests.forecast.conftest import synth_series
    s = synth_series(gaps=[("lab", g0, 3)])
    a = arrays(fcfg, s)
    si = a.spaces.index("lab")
    assert a.obs[si, 3000:3003, 0].sum() == 0
    assert np.isnan(a.y[si, 3000:3003]).all()
    # 앞으로 채운 값은 결측 직전 관측값
    assert (a.raw_ffill[si, 3000:3003, 0] == a.raw_ffill[si, 2999, 0]).all()
    assert a.age_h[si, 3002, 0] == pytest.approx(15 / 60)


def test_meeting_obs_alternates(fcfg, series):
    a = arrays(fcfg, series)
    si = a.spaces.index("meeting")
    o = a.obs[si, 500:510, 0]
    assert o.sum() == 5 and (np.diff(o.astype(int)) != 0).all()


def test_split_boundaries(fcfg, series):
    a = arrays(fcfg, series)
    for kind in ("train", "val"):
        idx = split_t0(a, fcfg, kind, 12)
        lo, hi = fcfg.split.utc(kind)
        t0 = a.t[idx[:, 1]]
        assert (t0 >= lo).all() and (t0 + pd.Timedelta(minutes=60) < hi).all()
        # 현재값 관측
        assert not np.isnan(a.y[idx[:, 0], idx[:, 1]]).any()


def test_unpredictable_when_input_missing(fcfg, series):
    a = arrays(fcfg, series)
    si = a.spaces.index("office")
    a.obs[si, 6000:6100] = 0
    ev = pd.DataFrame({"space_id": ["office", "office"], "item": "co2", "t0": [a.t[6050], a.t[7000]],
                       "horizon_min": 60})
    _, unpred, _ = eval_samples(a, ev, fcfg, 12)
    assert list(unpred) == [True, False]
    assert input_obs_ratio(a, si, 12)[6050] == 0


def test_pooled_norm_ignores_held_out(norm):
    spaces = ("lab", "office", "meeting")
    p = norm_table(norm, spaces, "pooled", ("lab", "office"))
    norm2 = {"spaces": dict(norm["spaces"], meeting={k: dict(v, mean=v["mean"] * 100) for k, v in norm["spaces"]["meeting"].items()})}
    np.testing.assert_array_equal(p, norm_table(norm2, spaces, "pooled", ("lab", "office")))
    assert (p[0] == p[2]).all()

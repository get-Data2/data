import numpy as np
import pytest

from edgeopt.diff import ae_diff, ae_scores, class_flip, forecast_diff, to_q

pytestmark = pytest.mark.unit


def test_forecast_judgement_in_ppm():
    """정규화 출력 차이 × σ = 원 단위 차이. σ_CO₂ = 400ppm에서 0.5ppm은 통과, 1.5ppm은 실패."""
    n = 10
    ref = np.zeros((n, 12, 2), np.float32)
    sd = np.tile([[400.0, 2.0]], (n, 1))
    ok = ref.copy()
    ok[3, 5, 0] = 0.5 / 400
    bad = ref.copy()
    bad[7, 11, 0] = 1.5 / 400
    r1 = forecast_diff(ref, ok, sd)
    r2 = forecast_diff(ref, bad, sd)
    assert r1["ok"] and r1["co2_max_ppm"] == pytest.approx(0.5, abs=1e-4)
    assert not r2["ok"] and r2["co2_max_ppm"] == pytest.approx(1.5, abs=1e-4) and r2["worst_index"] == 7


def test_forecast_mae_change():
    ref = np.zeros((2, 12, 2))
    out = np.full((2, 12, 2), 0.01)
    sd = np.ones((2, 2)) * 100
    base = np.full((2, 2), 500.0)
    y = np.full((2, 12, 2), 500.0)
    y[0, :, 1] = np.nan
    r = forecast_diff(ref, out, sd, y, base)
    assert r["co2_mae_ref"] == 0 and r["co2_mae_out"] == pytest.approx(1.0)
    assert r["temp_mae_out"] == pytest.approx(1.0)               # 결측 정답은 빼고 계산


def test_ae_alarm_flip_rate():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, (1000, 12, 3))
    ref = z.copy()
    out = z.copy()
    out[:5, -1, 0] += 50                                         # 5개 창만 크게 바뀜
    q = np.sort(ae_scores(rng.normal(0, 0.5, (1000, 12, 3)) + z, z, np.ones(3), 3))
    r = ae_diff(ref, out, z, np.ones(3), 3, list(np.quantile(q, np.linspace(0, 1, 1001))), 0.999)
    assert r["alarm_flip_rate"] == pytest.approx(0.005) and r["ok"] is False
    r_same = ae_diff(ref, ref, z, np.ones(3), 3, list(q), 0.999)
    assert r_same["alarm_flip_rate"] == 0 and r_same["ok"]


def test_class_flip_and_quantile_interp():
    a = np.array([[2.0, 1.0]] * 999 + [[1.0, 2.0]])
    b = a.copy()
    assert class_flip(a, b)["ok"]
    b[0] = [0.0, 3.0]
    b[1] = [0.0, 3.0]
    r = class_flip(a, b)
    assert r["flip_rate"] == pytest.approx(0.002) and not r["ok"]
    assert list(to_q(np.array([-1, 5, 20]), [0, 10])) == [0.0, 0.5, 1.0]

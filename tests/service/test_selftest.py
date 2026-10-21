"""자기 성능 감시(심화 2, tasks T036)와 값 고정 규칙."""
import pandas as pd
import pytest

from service.rules import stuck
from service.selftest import SelfMonitor
from tests.service.conftest import at

pytestmark = pytest.mark.unit


def test_score_and_drift(scfg):
    m = SelfMonitor(scfg)
    t0 = at("10:00")
    fc = {"horizons_min": list(range(5, 65, 5)), "co2": [1000.0] * 12, "temperature": [21.0] * 12}
    for k in range(40):                                    # 40번 예측, 정답은 모두 800 → 오차 200
        m.log(t0 + pd.Timedelta(minutes=5 * k), "lab", fc)
    bins = pd.date_range(t0, periods=60, freq="5min")
    sb = pd.DataFrame({"space_id": "lab", "item": "co2", "bin": bins, "value": 800.0, "filled": False})
    m.score(sb, bins[-1])
    roll = m.rolling()
    assert roll["lab/co2/60"]["mae"] == 200.0 and roll["lab/co2/60"]["n"] >= 30
    d = m.drift({"lab/co2/60": 80.0})
    assert d and d[0]["kind"] == "model_drift" and d[0]["evidence"]["test_mae"] == 80.0
    assert m.drift({"lab/co2/60": 150.0}) == []            # 200 < 150 × 2


def test_stuck_rule():
    t = pd.date_range(at("09:00"), periods=20, freq="5min")
    rows = []
    for i, b in enumerate(t):
        rows.append(("A", "lab", "co2", b, 600.0, False))              # 고정
        rows.append(("B", "lab", "co2", b, 600.0 + 10 * i, False))     # 이웃은 190ppm 변함
    sb = pd.DataFrame(rows, columns=["device_id", "space_id", "item", "bin", "value", "filled"])
    out = stuck(sb, {"min_hours": 1.0, "others_min_change": {"co2": 50}})
    assert [o["device_id"] for o in out] == ["A"] and out[0]["hours"] >= 1

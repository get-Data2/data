"""경보 상태 기계 (spec US6, tasks T029)."""
import pandas as pd
import pytest

from service.alerts import AlertManager
from tests.service.conftest import at

pytestmark = pytest.mark.unit
F = [{"kind": "data_gap", "space_id": "lab", "device_id": "D1", "evidence": {}}]


def minutes(n, start="10:00"):
    t = at(start)
    return [t + pd.Timedelta(minutes=i) for i in range(n)]


def test_raise_once_renotify_every_30min(scfg, tmp_path):
    m = AlertManager(scfg, tmp_path / "s.json")
    ev = [e["event"] for t in minutes(61) for e in m.update(t, F)]
    assert ev == ["raised", "renotified", "renotified"]           # 0분, 30분, 60분


def test_clear_after_3_ok_cycles_and_no_flapping(scfg, tmp_path):
    m = AlertManager(scfg, tmp_path / "s.json")
    ts = minutes(10)
    seq = [F, [], F, [], [], [], F]                                 # 정상 2주기 뒤 다시 → 해제 안 됨
    ev = [(i, e["event"]) for i, (t, f) in enumerate(zip(ts, seq)) for e in m.update(t, f)]
    assert ev == [(0, "raised"), (5, "cleared"), (6, "raised")]


def test_restart_does_not_reraise(scfg, tmp_path):
    p = tmp_path / "s.json"
    m = AlertManager(scfg, p)
    m.update(at("10:00"), F)
    m2 = AlertManager(scfg, p)                                      # 재시작
    assert m2.update(at("10:01"), F) == []
    assert m2.active_list("lab")[0]["notify_role"] == "센서 관리 담당"

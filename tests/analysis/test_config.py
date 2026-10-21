import copy

import pytest
import yaml

from analysis.config import AnalysisConfigError, load_analysis_config
from tests.conftest import ROOT

pytestmark = pytest.mark.unit


def test_load(acfg):
    sp = acfg.split()
    assert sp.name == "v1-provisional"
    a, b = sp.utc("test")
    assert a.isoformat() == "2026-05-26T15:00:00+00:00" and b.isoformat() == "2026-07-22T15:00:00+00:00"
    assert acfg.period_s("24e124126d152590") == 120
    assert acfg.period_s("unknown") == 60
    assert acfg.is_periodic("24e124141d189196") is False
    assert len(acfg.digest) == 12


def write(tmp_path, mutate):
    raw = yaml.safe_load((ROOT / "config" / "analysis.yaml").read_text(encoding="utf-8"))
    raw = copy.deepcopy(raw)
    mutate(raw)
    p = tmp_path / "a.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return p


@pytest.mark.parametrize("mutate, msg", [
    (lambda r: r["split"]["versions"]["v1-provisional"].update({"val": ["2026-03-01", "2026-05-09"]}), "겹치거나"),
    (lambda r: r["split"]["versions"]["v1-provisional"].update({"val": ["2026-04-01", "2026-05-26"]}), "간격이 1일 미만"),
    (lambda r: r["split"].update({"active": "v9"}), "없는 분할 버전"),
    (lambda r: r["schedule"]["rows"][0].update({"days": ["monday"]}), "알 수 없는 요일"),
])
def test_invalid(tmp_path, mutate, msg):
    with pytest.raises(AnalysisConfigError) as e:
        load_analysis_config(write(tmp_path, mutate), None)
    assert msg in str(e.value)

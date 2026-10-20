import copy

import pytest
import yaml

from collector.config import UNMAPPED, ConfigError, build_config, load_config
from tests.conftest import CONFIG_PATH

pytestmark = pytest.mark.unit


@pytest.fixture
def raw():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_project_config_is_valid():
    cfg = load_config(CONFIG_PATH)
    assert {"lab", "office", "ceo_room", "plant", UNMAPPED} <= set(cfg.spaces)
    assert [s.name for s in cfg.sources] == ["chirpstack", "xflow"]
    assert "gateway/+/state/#" in cfg.all_topics()


def test_resolve_space_order(config):
    assert config.resolve_space("24e124785c389818", "엉뚱한 곳") == ("lab", "device")
    assert config.resolve_space("other", "실습실") == ("lab", "alias")
    assert config.resolve_space("other", " 사무실 밖 ") == ("office_out", "alias")
    assert config.resolve_space("other", "controller") == ("plant", "alias")
    assert config.resolve_space("other", "보일러실") == (UNMAPPED, "none")
    assert config.resolve_space("other", None) == (UNMAPPED, "none")


def test_item_aliases(config):
    assert config.item_name("current_temperature") == "temperature"
    assert config.item_name("co2") == "co2"
    assert config.item_name("tvoc") == "tvoc"
    assert config.item_name("LAeq") == "LAeq"
    assert config.item_unit("co2") == "ppm"


def bad(raw, mutate):
    r = copy.deepcopy(raw)
    mutate(r)
    with pytest.raises(ConfigError) as e:
        build_config(r, "x.yaml")
    return "\n".join(e.value.errors)


def test_bad_space_id(raw):
    msg = bad(raw, lambda r: r["spaces"].append({"id": "Lab 2", "name": "x", "kind": "office"}))
    assert "x.yaml: spaces[9].id" in msg


def test_duplicate_alias(raw):
    msg = bad(raw, lambda r: r["spaces"][1]["aliases"].append("실습실"))
    assert "'실습실'가 공간 'lab'와 겹침" in msg


def test_device_unknown_space(raw):
    msg = bad(raw, lambda r: r["devices"].update({"abc": {"space": "nowhere"}}))
    assert "devices.abc.space: 없는 공간 id 'nowhere'" in msg


def test_bad_topic_and_expr(raw):
    def m(r):
        r["sources"][1]["topic"] = "xflow/#/status"
        r["sources"][0]["time"] = ["max2(ts)"]
    msg = bad(raw, m)
    assert "sources[1].topic" in msg
    assert "sources[0].time[0]: 알 수 없는 함수 'max2'" in msg


def test_missing_required(raw):
    msg = bad(raw, lambda r: r["sources"][0].pop("measurements"))
    assert "sources[0].measurements: 필수" in msg


def test_unknown_keys(raw):
    msg = bad(raw, lambda r: (r["mqtt"].update({"kepalive": 3}), r.update({"spacs": []})))
    assert "mqtt: 알 수 없는 키" in msg and "최상위 키" in msg


def test_reports_all_errors_at_once(raw):
    def m(r):
        r["version"] = 2
        r["devices"]["abc"] = {"space": "nowhere"}
    assert bad(raw, m).count("x.yaml:") == 2

import pytest

from collector.backfill import convert, infer_table
from collector.influx import parse_csv

pytestmark = pytest.mark.unit


def row(**kw):
    base = {"_time": "2025-09-10T01:02:03.456Z", "_value": "812", "_measurement": "co2",
            "device_id": "24e124725d081175", "location": "사무실", "point": "업무 공간 안쪽", "spot": "업무 공간 안쪽"}
    return {**base, **kw}


def test_convert_tagged(config):
    m, info, why = convert(row(), config, {})
    assert why is None
    assert m["space_id"] == "office" and m["item"] == "co2" and m["value"] == 812.0 and m["unit"] == "ppm"
    assert m["ts"] == "2025-09-10 01:02:03.456" and m["source"] == "influx" and m["frame_cnt"] is None
    assert info["mapped_by"] == "alias" and info["raw_location"] == "사무실"


def test_convert_infers_untagged(config):
    infer = infer_table([row(), row(device_id="24e124136d151485", location="창고", point="서버 옆")])
    m, info, why = convert(row(device_id=""), config, infer)
    assert why is None and m["device_id"] == "24e124725d081175"
    m, _, _ = convert(row(device_id="", location="창고", point="서버 옆", _measurement="temperature"), config, infer)
    assert m["device_id"] == "24e124136d151485" and m["space_id"] == "storage"


def test_infer_ambiguous_dropped():
    t = infer_table([row(device_id="a"), row(device_id="b"), row(device_id="c", point="창가")])
    assert ("사무실", "업무 공간 안쪽") not in t and t[("사무실", "창가")] == "c"


@pytest.mark.parametrize("kw, why", [
    ({"device_id": "", "point": "모름"}, "장치 특정 불가"),
    ({"device_id": "dev001", "location": "lab"}, "수집 제외 장치"),
    ({"_value": "abc"}, "숫자 아님"),
    ({"_value": "NaN"}, "숫자 아님"),
])
def test_convert_skips(config, kw, why):
    assert convert(row(**kw), config, {})[2] == why


def test_unknown_location_unmapped(config):
    m, info, _ = convert(row(location="강의실"), config, {})
    assert m["space_id"] == "unmapped" and info["mapped_by"] == "none"


def test_parse_annotated_csv_multiple_tables():
    text = ("#datatype,string,long,dateTime:RFC3339,double\n#group,false,false,false,false\n#default,_result,,,\n"
            ",result,table,_time,_value\n,,0,2025-09-10T00:00:00Z,1\n,,0,2025-09-10T00:01:00Z,2\n\n"
            "#datatype,string,long,dateTime:RFC3339,double,string\n#group,false,false,false,false,true\n#default,_result,,,,\n"
            ",result,table,_time,_value,device_id\n,,1,2025-09-10T00:00:00Z,3,d\n\n")
    rows = parse_csv(text)
    assert [r["_value"] for r in rows] == ["1", "2", "3"] and rows[2]["device_id"] == "d"

"""맥 내보내기 스크립트·CSV 파일 적재(2026-10-21 유선망 단절 대응)."""
import json
import re
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from collector.backfill import chunk_file, flux_chunk, flux_infer, plan_chunks
from collector.influx import FileReader
from collector.influx_export import make_script

pytestmark = pytest.mark.unit
UTC = timezone.utc
CSV = "#datatype,string,long,dateTime:RFC3339,double,string,string,string,string,string\n" \
      ",result,table,_time,_value,_measurement,device_id,location,point,spot\n" \
      ",_result,0,2026-07-23T00:01:00Z,612,co2,dev1,실습실,,\n\n"


def test_plan_skips_done(config):
    a = datetime(2026, 7, 22, tzinfo=UTC)
    end = datetime(2026, 7, 25, 7, 33, tzinfo=UTC)
    prog = {"done": {a.isoformat(): {"stop": (a + timedelta(days=1)).isoformat()}}}
    ch = plan_chunks(config, prog, a, end)
    assert [c[0].day for c in ch] == [23, 24, 25] and ch[-1][1] == end


def test_script_same_flux_no_token(config):
    a = datetime(2026, 7, 23, tzinfo=UTC)
    ch = [(a, a + timedelta(days=1))]
    s = make_script(config, "http://influx.example:8086", "iot-bucket", ch, a + timedelta(days=1))
    assert "Token $INFLUX_TOKEN" in s and "read -r -s" in s
    assert not re.search(r"INFLUX_TOKEN=\S", s)                      # 토큰 값이 들어가지 않음
    bodies = re.findall(r"<<'Q\d+'\n(.*?)\nQ\d+\n", s, re.S)
    assert [json.loads(b)["query"] for b in bodies] == [flux_infer("iot-bucket", config, a + timedelta(days=1)),
                                                        flux_chunk("iot-bucket", config, *ch[0])]
    assert chunk_file(a) in s
    r = subprocess.run(["bash", "-n"], input=s, text=True, capture_output=True)
    assert r.returncode == 0, r.stderr


def test_file_reader_maps_queries(config, tmp_path):
    a = datetime(2026, 7, 23, tzinfo=UTC)
    (tmp_path / chunk_file(a)).write_text(CSV, encoding="utf-8")
    (tmp_path / "infer.csv").write_text(CSV, encoding="utf-8")
    fr = FileReader(tmp_path, "iot-bucket")
    rows = fr.query(flux_chunk("iot-bucket", config, a, a + timedelta(days=1)))
    assert rows[0]["_value"] == "612" and rows[0]["device_id"] == "dev1"
    assert fr.query(flux_infer("iot-bucket", config, a))[0]["location"] == "실습실"
    with pytest.raises(RuntimeError, match="내보낸 파일이 없음"):
        fr.query(flux_chunk("iot-bucket", config, a + timedelta(days=1), a + timedelta(days=2)))

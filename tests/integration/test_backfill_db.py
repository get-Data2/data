"""과거 데이터 적재를 가짜 InfluxDB 응답과 실 MySQL(test_ 테이블)로 검증한다."""
from datetime import datetime, timezone

import pytest

from collector import store
from collector.backfill import Backfill

pytestmark = pytest.mark.mysql


class FakeReader:
    bucket = "iot-bucket"

    def __init__(self, rows):
        self.rows = rows

    def query(self, flux):
        if "|> first()" in flux:
            return [min(self.rows, key=lambda r: r["_time"])]
        if "|> last()" in flux:          # 장치 추론표용
            return [r for r in self.rows if r.get("device_id")]
        return list(self.rows)


ROWS = [
    {"_time": "2025-09-10T01:00:00Z", "_value": "800", "_measurement": "co2", "device_id": "dA",
     "location": "사무실", "point": "안쪽", "spot": "안쪽"},
    {"_time": "2025-09-10T01:01:00Z", "_value": "24.5", "_measurement": "temperature", "device_id": "dA",
     "location": "사무실", "point": "안쪽", "spot": "안쪽"},
    {"_time": "2025-09-10T01:02:00Z", "_value": "810", "_measurement": "co2", "device_id": "",
     "location": "사무실", "point": "안쪽"},                                # 태그 없음 → dA로 복원
    {"_time": "2025-09-10T01:03:00Z", "_value": "1", "_measurement": "co2", "device_id": "dev001",
     "location": "lab", "point": "x"},                                      # 수집 제외
]


def count(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()[0]


def test_backfill_idempotent_and_keeps_current_space(mysql_conn, config, tmp_path):
    conn, p = mysql_conn
    # 수집기가 이미 등록한 장치: 지금은 실습실(lab)에 있음
    store.write_batch(conn, p, [("device", {
        "device_id": "dA", "source": "chirpstack", "name": "AM103", "model": "AM103", "space_id": "lab",
        "spot": "창가", "raw_location": "실습실", "mapped_by": "alias", "expected_period_s": 60,
        "first_seen": "2026-10-20 07:32:52.000", "last_seen": "2026-10-20 08:00:00.000"})])
    end = datetime(2025, 9, 11, tzinfo=timezone.utc)
    bf = Backfill(config, FakeReader(ROWS), conn, p, progress_path=str(tmp_path / "prog.json"))
    out = bf.run(end=end)
    assert "적재 3" in out and "수집 제외 장치 1" in out
    assert count(conn, f"SELECT COUNT(*) FROM `{p}measurement` WHERE source='influx'") == 3
    assert count(conn, f"SELECT COUNT(*) FROM `{p}measurement` WHERE space_id='office'") == 3   # 과거 공간은 그때 위치

    with conn.cursor() as cur:
        cur.execute(f"SELECT space_id, first_seen, last_seen FROM `{p}device` WHERE device_id='dA'")
        space, first, last = cur.fetchone()
    assert space == "lab"                                         # 현재 공간은 바꾸지 않음
    assert str(first).startswith("2025-09-10 01:00") and str(last).startswith("2026-10-20 08:00")

    # 진행 파일이 있으면 건너뛰고, 진행 파일을 지우고 다시 해도 행이 늘지 않는다
    assert "남은 0개" in Backfill(config, FakeReader(ROWS), conn, p, str(tmp_path / "prog.json")).run(end=end)
    Backfill(config, FakeReader(ROWS), conn, p, str(tmp_path / "other.json")).run(end=end)
    assert count(conn, f"SELECT COUNT(*) FROM `{p}measurement`") == 3


def _live(dev, ts, v):
    return ("measurement", {"ts": ts, "space_id": "office", "device_id": dev, "item": "co2", "value": v, "unit": "ppm",
                            "source": "chirpstack", "received_at": ts, "source_ts": ts, "ts_corrected": 0, "frame_cnt": None})


def test_fill_gaps_only_inside_gap_without_near_duplicates(mysql_conn, config):
    """수집기 정지 빈틈만 InfluxDB로 채운다. 경계의 같은 측정(InfluxDB 시각이 0.2초 늦음)은 중복으로 넣지 않는다."""
    from collector.backfill import fill_gaps, find_gaps
    conn, p = mysql_conn
    store.write_batch(conn, p, [_live("dA", "2026-10-21 10:50:00.000", 700), _live("dA", "2026-10-21 10:51:00.000", 710),
                                _live("dA", "2026-10-21 11:01:00.000", 760), _live("dA", "2026-10-21 11:02:00.000", 770)])
    rows = [{"_time": f"2026-10-21T{t}Z", "_value": v, "_measurement": "co2", "device_id": "dA",
             "location": "사무실", "point": "안쪽", "spot": "안쪽"}
            for t, v in [("10:51:00.230", "710"), ("10:53:00.200", "720"), ("10:57:00.200", "740"), ("11:01:00.210", "760")]]
    a, b = datetime(2026, 10, 21, 10, 0, tzinfo=timezone.utc), datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
    assert [(g[0].minute, g[1].minute) for g in find_gaps(conn, p, a, b)] == [(51, 1)]
    bf = Backfill(config, FakeReader(rows), conn, p)
    res = fill_gaps(bf, a, b)
    assert res[0]["written"] == 2                        # 10:53, 10:57 — 경계 두 행(10:51·11:01 쌍둥이)은 제외
    assert count(conn, f"SELECT COUNT(*) FROM `{p}measurement` WHERE source='influx'") == 2
    assert fill_gaps(bf, a, b)[0]["written"] == 0        # 다시 돌려도 그대로

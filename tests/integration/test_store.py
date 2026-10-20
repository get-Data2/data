import pytest

from collector import store

pytestmark = pytest.mark.mysql


def meas(ts="2026-10-20 06:42:04.228", value=879.0, device="d1"):
    return ("measurement", {"ts": ts, "space_id": "lab", "device_id": device, "item": "co2", "value": value,
                            "unit": "ppm", "source": "chirpstack", "received_at": ts, "source_ts": None,
                            "ts_corrected": 0, "frame_cnt": 5})


def dev(last, first=None):
    return ("device", {"device_id": "d1", "source": "chirpstack", "name": "x", "model": "EM500-CO2",
                       "space_id": "lab", "spot": None, "raw_location": "실습실", "mapped_by": "alias",
                       "expected_period_s": 60, "first_seen": first or last, "last_seen": last})


def one(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()


def test_idempotent_measurement(mysql_conn):
    conn, p = mysql_conn
    store.write_batch(conn, p, [meas()])
    store.write_batch(conn, p, [meas()])          # 재처리
    assert one(conn, f"SELECT COUNT(*) FROM `{p}measurement`")[0] == 1


def test_broken_value_stored(mysql_conn):
    conn, p = mysql_conn
    store.write_batch(conn, p, [meas(value=0.0)])
    assert one(conn, f"SELECT value FROM `{p}measurement`")[0] == 0.0


def test_device_last_seen_only_grows(mysql_conn):
    conn, p = mysql_conn
    store.write_batch(conn, p, [dev("2026-10-20 06:45:00.000")])
    store.write_batch(conn, p, [dev("2026-10-20 06:40:00.000")])   # 늦게 도착한 옛 행
    first, last = one(conn, f"SELECT first_seen, last_seen FROM `{p}device`")
    assert str(last).startswith("2026-10-20 06:45") and str(first).startswith("2026-10-20 06:40")


def test_stats_merge_and_events(mysql_conn):
    conn, p = mysql_conn
    row = {"minute": "2026-10-20 06:42:00", "host": "h", "messages": 3, "stored": 10, "ignored": 1,
           "failed": 0, "duplicate": 0, "max_lag_s": 2.5}
    store.write_batch(conn, p, [("stats", row), ("stats", {**row, "max_lag_s": 4.0}),
                                ("event", {"ts": "2026-10-20 06:42:01.000", "kind": "start", "detail": None,
                                           "host": "h"})])
    assert one(conn, f"SELECT messages, `stored`, max_lag_s FROM `{p}ingest_stats`") == (6, 20, 4.0)
    assert one(conn, f"SELECT kind FROM `{p}collector_event`")[0] == "start"


def test_sync_spaces(mysql_conn, config):
    conn, p = mysql_conn
    store.sync_spaces(conn, config, p)
    assert one(conn, f"SELECT name FROM `{p}space` WHERE space_id='lab'")[0] == "실습실"

"""현황 보고를 실 DB(test_ 테이블)의 합성 측정값으로 검증한다."""
import argparse
from datetime import datetime, timedelta, timezone

import pytest

from collector import report, store

pytestmark = pytest.mark.mysql
T0 = datetime(2026, 10, 20, 0, 0, tzinfo=timezone.utc)


def fmt(t):
    return t.strftime("%Y-%m-%d %H:%M:%S.000")


def test_report_on_synthetic_rows(mysql_conn, config, tmp_path):
    conn, p = mysql_conn
    store.sync_spaces(conn, config, p)
    rows = []
    for i in range(60):
        t = T0 + timedelta(minutes=i, seconds=10)
        for dev, missing in (("a", range(20, 25)), ("b", range(40, 44))):
            if i in missing or 50 <= i < 53:
                continue
            rows.append(("measurement", {"ts": fmt(t), "space_id": "lab", "device_id": dev, "item": "co2",
                                         "value": 500.0, "unit": "ppm", "source": "t", "received_at": fmt(t),
                                         "source_ts": None, "ts_corrected": 0, "frame_cnt": None}))
        if not 50 <= i < 53:        # 50~52분: 수집기 미가동
            rows.append(("stats", {"minute": (T0 + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:00"),
                                   "host": "h", "messages": 2, "stored": 2, "ignored": 0, "failed": 0,
                                   "duplicate": 0, "max_lag_s": 0.1}))
    for dev in ("a", "b"):
        rows.append(("device", {"device_id": dev, "source": "t", "name": None, "model": "EM500-CO2",
                                "space_id": "lab", "spot": None, "raw_location": "실습실", "mapped_by": "alias",
                                "expected_period_s": 60, "first_seen": fmt(T0), "last_seen": fmt(T0 + timedelta(minutes=59))}))
    rows.append(("event", {"ts": fmt(T0 + timedelta(minutes=50, seconds=1)), "kind": "stop", "detail": None,
                           "host": "h"}))
    store.write_batch(conn, p, rows)

    md = tmp_path / "r.md"
    args = argparse.Namespace(prefix=p, capacity=False, to="2026-10-20 10:00", from_="2026-10-20 09:00",
                              hours=1, space=None, synthetic=True, by=None, md=str(md))
    text = report.main(conn, config, args)
    saved = md.read_text(encoding="utf-8")
    assert saved.startswith("# 수집 현황 보고") and "합성 데이터" in saved
    assert "2026-10-20 09:00 ~ 2026-10-20 10:00 KST" in saved
    # 공간 합산: a·b 중 하나라도 있으면 채움 → 비는 칸은 50~52분 3칸 → 57/60
    assert "| 실습실(lab) | co2 | 2 |" in text and "95.0%" in text
    assert "| (전체) | — | 10-20 09:50 | 10-20 09:53 | 3분 | 수집기 미가동" in text   # 전체 결측의 원인 단서
    assert "stop 09:50" in text
    assert "| a | 실습실(lab) | 10-20 09:20 | 10-20 09:25 | 5분 | 해당 장치만 결측" in text
    assert "| b | 실습실(lab) | 10-20 09:40 | 10-20 09:44 | 4분 |" in text

"""실 브로커 연기 시험: 수집기를 하위 프로세스로 몇 분 돌려 실데이터가 쌓이는지 본다.

실행: .venv/bin/python -m pytest -m live   (기본 180초, LIVE_SECONDS로 조정)
"""
import os
import signal
import subprocess
import sys
import time
import uuid

import pytest
import yaml

from tests.conftest import CONFIG_PATH, ROOT

pytestmark = pytest.mark.live


def test_live_collects_real_data(mysql_conn, tmp_path):
    conn, prefix = mysql_conn
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["storage"]["table_prefix"] = prefix
    raw["mqtt"]["client_id"] = f"team2-collector-test-{uuid.uuid4().hex[:6]}"
    raw["spool"]["path"] = str(tmp_path / "spool.db")
    cfg = tmp_path / "collector.yaml"
    cfg.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    seconds = int(os.environ.get("LIVE_SECONDS", "180"))
    proc = subprocess.Popen([sys.executable, "-m", "collector", "--config", str(cfg),
                             "--env", str(ROOT / ".env"), "run"], cwd=ROOT)
    try:
        time.sleep(seconds)
    finally:
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 0

    with conn.cursor() as cur:
        cur.execute(f"SELECT item, COUNT(*) FROM `{prefix}measurement` GROUP BY item")
        items = dict(cur.fetchall())
        cur.execute(f"SELECT COUNT(*) FROM `{prefix}measurement` WHERE space_id='unmapped'")
        unmapped = cur.fetchone()[0]
        cur.execute(f"SELECT MAX(ABS(TIMESTAMPDIFF(MICROSECOND, ts, received_at)))/1e6 "
                    f"FROM `{prefix}measurement` WHERE ts_corrected=0")
        max_skew = float(cur.fetchone()[0])
        cur.execute(f"SELECT kind FROM `{prefix}collector_event` ORDER BY id")
        kinds = [r[0] for r in cur.fetchall()]
        cur.execute(f"SELECT COUNT(*) FROM `{prefix}ingest_stats`")
        stat_rows = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM `{prefix}device`")
        devices = cur.fetchone()[0]

    print(f"\n항목별 건수 {items}\n장치 {devices}대, 최대 |ts-수신| {max_skew:.1f}초, 사건 {kinds}")
    assert items.get("co2", 0) > 0 and items.get("temperature", 0) > 0
    assert unmapped == 0
    assert max_skew <= 120
    assert kinds[0] == "start" and "mqtt_connected" in kinds and kinds[-1] == "stop"
    assert stat_rows >= seconds // 60 - 1
    assert devices >= 10

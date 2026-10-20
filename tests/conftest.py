"""공통 시험 고정물."""
import json
import os
import pathlib
import uuid
from datetime import datetime, timezone

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
CONFIG_PATH = ROOT / "config" / "collector.yaml"


@pytest.fixture
def fixture_messages():
    """2026-10-20 실 브로커에서 3분간 받은 메시지 표본."""
    with open(FIXTURES / "messages.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture
def config():
    from collector.config import load_config
    return load_config(CONFIG_PATH)


@pytest.fixture
def now_utc():
    return datetime(2026, 10, 20, 6, 42, 30, tzinfo=timezone.utc)


@pytest.fixture
def mysql_conn():
    """실 MySQL 연결과 시험용 테이블 접두사. 접속할 수 없으면 건너뛴다. 시험 뒤 테이블을 지운다."""
    from collector import store
    try:
        conn = store.connect(ROOT / ".env")
    except Exception as e:  # 접속 불가 환경
        pytest.skip(f"MySQL 접속 불가: {type(e).__name__}")
    prefix = f"test_{uuid.uuid4().hex[:6]}_"
    store.init_db(conn, prefix)
    yield conn, prefix
    with conn.cursor() as cur:
        for t in store.TABLES:
            cur.execute(f"DROP TABLE IF EXISTS `{prefix}{t}`")
    conn.commit()
    conn.close()

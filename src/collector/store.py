"""MySQL 저장소: 연결, DDL, 일괄 기록."""
from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

import pymysql
from dotenv import load_dotenv

TABLES = ("space", "device", "measurement", "collector_event", "ingest_stats")


def connect(env_path: str | Path = ".env") -> pymysql.connections.Connection:
    """`.env`의 MYSQL_* 로 접속한다. 세션 시간대는 UTC."""
    load_dotenv(env_path, override=False)
    return pymysql.connect(
        host=os.environ["MYSQL_HOST"],
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASS"],
        database=os.environ["MYSQL_DB"],
        charset="utf8mb4",
        autocommit=False,
        connect_timeout=10,
        read_timeout=30,
        write_timeout=30,
        init_command="SET time_zone='+00:00'",
    )


def init_db(conn, prefix: str = "") -> list[str]:
    """테이블을 만든다(이미 있으면 그대로). 만든·확인한 테이블 이름을 돌려준다."""
    ddl = resources.files("collector").joinpath("schema.sql").read_text(encoding="utf-8")
    lines = [ln for ln in ddl.splitlines() if not ln.strip().startswith("--")]
    with conn.cursor() as cur:
        for stmt in "\n".join(lines).split(";"):
            if stmt.strip():
                cur.execute(stmt.replace("{p}", prefix))
    conn.commit()
    return [prefix + t for t in TABLES]


def sync_spaces(conn, config, prefix: str = "") -> int:
    rows = [(s.id, s.name, s.kind, s.area_m2, s.note) for s in config.spaces.values()]
    with conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO `{prefix}space` (space_id, name, kind, area_m2, note) VALUES (%s,%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE name=VALUES(name), kind=VALUES(kind), "
            "area_m2=VALUES(area_m2), note=VALUES(note)",
            rows,
        )
    conn.commit()
    return len(rows)


_MEAS_COLS = ("ts", "space_id", "device_id", "item", "value", "unit", "source", "received_at",
              "source_ts", "ts_corrected", "frame_cnt")
_DEV_COLS = ("device_id", "source", "name", "model", "space_id", "spot", "raw_location", "mapped_by",
             "expected_period_s", "first_seen", "last_seen")
_STAT_COLS = ("minute", "host", "messages", "stored", "ignored", "failed", "duplicate", "max_lag_s")


def write_batch(conn, prefix: str, batch: list[tuple[str, dict]]) -> None:
    """스풀에서 꺼낸 (kind, row) 묶음을 한 트랜잭션으로 기록한다. 같은 묶음을 다시 써도 결과가 같다(멱등)."""
    by_kind: dict[str, list[dict]] = {}
    for kind, row in batch:
        by_kind.setdefault(kind, []).append(row)
    try:
        with conn.cursor() as cur:
            if rows := by_kind.get("measurement"):
                cols = ",".join(_MEAS_COLS)
                cur.executemany(
                    f"INSERT INTO `{prefix}measurement` ({cols}) VALUES ({','.join(['%s'] * len(_MEAS_COLS))}) "
                    "ON DUPLICATE KEY UPDATE id=id",
                    [tuple(r.get(c) for c in _MEAS_COLS) for r in rows],
                )
            if rows := by_kind.get("device"):
                cols = ",".join(_DEV_COLS)
                cur.executemany(
                    f"INSERT INTO `{prefix}device` ({cols}) VALUES ({','.join(['%s'] * len(_DEV_COLS))}) "
                    "ON DUPLICATE KEY UPDATE source=VALUES(source), "
                    "name=COALESCE(VALUES(name), name), model=COALESCE(VALUES(model), model), "
                    "space_id=VALUES(space_id), spot=COALESCE(VALUES(spot), spot), "
                    "raw_location=VALUES(raw_location), mapped_by=VALUES(mapped_by), "
                    "expected_period_s=VALUES(expected_period_s), "
                    "first_seen=LEAST(first_seen, VALUES(first_seen)), "
                    "last_seen=GREATEST(last_seen, VALUES(last_seen))",
                    [tuple(r.get(c) for c in _DEV_COLS) for r in rows],
                )
            if rows := by_kind.get("event"):
                cur.executemany(
                    f"INSERT INTO `{prefix}collector_event` (ts, kind, detail, host) VALUES (%s,%s,%s,%s)",
                    [(r["ts"], r["kind"], (r.get("detail") or "")[:512] or None, r["host"]) for r in rows],
                )
            if rows := by_kind.get("stats"):
                cols = ",".join(f"`{c}`" for c in _STAT_COLS)
                cur.executemany(
                    f"INSERT INTO `{prefix}ingest_stats` ({cols}) VALUES ({','.join(['%s'] * len(_STAT_COLS))}) "
                    "ON DUPLICATE KEY UPDATE messages=messages+VALUES(messages), `stored`=`stored`+VALUES(`stored`), "
                    "ignored=ignored+VALUES(ignored), failed=failed+VALUES(failed), "
                    "duplicate=duplicate+VALUES(duplicate), max_lag_s=GREATEST(COALESCE(max_lag_s,0), "
                    "COALESCE(VALUES(max_lag_s),0))",
                    [tuple(r.get(c) for c in _STAT_COLS) for r in rows],
                )
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise

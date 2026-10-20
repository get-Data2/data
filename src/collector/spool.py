"""SQLite 스풀: 받은 측정값을 DB에 기록하기 전에 보드 안에 먼저 남긴다(무손실, research R2)."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path


class Spool:
    def __init__(self, path: str | Path, max_rows: int = 400_000):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.max_rows = max_rows
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS outbox (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
                         "kind TEXT NOT NULL, payload TEXT NOT NULL)")
        self._count = self._db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]

    def put_many(self, rows: list[tuple[str, dict]]) -> dict | None:
        """(kind, row) 목록을 넣는다. 한도를 넘겨 오래된 측정값을 지웠으면 그 내역을 돌려준다."""
        if not rows:
            return None
        with self._lock:
            self._db.execute("BEGIN")
            self._db.executemany("INSERT INTO outbox (kind, payload) VALUES (?, ?)",
                                 [(k, json.dumps(r, ensure_ascii=False)) for k, r in rows])
            self._db.execute("COMMIT")
            self._count += len(rows)
            if self._count <= self.max_rows:
                return None
            n = self._count - self.max_rows
            old = self._db.execute(
                "SELECT seq, payload FROM outbox WHERE kind='measurement' ORDER BY seq LIMIT ?", (n,)
            ).fetchall()
            if not old:
                return None
            self._db.execute("DELETE FROM outbox WHERE kind='measurement' AND seq <= ?", (old[-1][0],))
            self._count -= len(old)
            return {"dropped": len(old), "from": json.loads(old[0][1]).get("ts"),
                    "to": json.loads(old[-1][1]).get("ts")}

    def peek(self, limit: int) -> list[tuple[int, str, dict]]:
        with self._lock:
            cur = self._db.execute("SELECT seq, kind, payload FROM outbox ORDER BY seq LIMIT ?", (limit,))
            return [(s, k, json.loads(p)) for s, k, p in cur.fetchall()]

    def ack(self, max_seq: int) -> None:
        with self._lock:
            n = self._db.execute("DELETE FROM outbox WHERE seq <= ?", (max_seq,)).rowcount
            self._count -= n

    def count(self) -> int:
        return self._count

    def close(self) -> None:
        with self._lock:
            self._db.close()

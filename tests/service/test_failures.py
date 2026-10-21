"""장애 관련: DB 읽기 연결 설정 (tasks T020 보강)."""
import pytest

pytestmark = pytest.mark.unit


def test_db_source_autocommit(monkeypatch):
    """읽기 연결이 자동 커밋이어야 주기마다 새 스냅샷을 본다(REPEATABLE READ 스냅샷 고정 방지)."""
    from collector import store
    from service.window import DbSource

    class Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *a): pass

    class Conn:
        auto = None
        def autocommit(self, v): Conn.auto = v
        def cursor(self): return Cur()

    monkeypatch.setattr(store, "connect", lambda *a, **k: Conn())
    DbSource({"cycle": {"db_timeout_s": 5}})._connect()
    assert Conn.auto is True

"""프로세스가 갑자기 죽어도 스풀에 남은 측정값은 다음 시작 때 기록된다."""
import pytest

from tests.integration.conftest import wait_for

pytestmark = pytest.mark.broker


def test_unflushed_rows_written_after_crash(make_collector, writer, publisher):
    c1 = make_collector(spool_name="shared.db")
    c1.start()
    assert c1.connected.wait(10)
    writer.fail = True
    for _ in range(20):
        publisher.publish_one().wait_for_publish(5)
    assert wait_for(lambda: c1.spool.count() >= 20, 15)
    left = c1.spool.count()
    # SIGKILL 흉내: 정리 없이 MQTT·기록 스레드만 멈춘다
    c1._stop.set()
    c1.client.loop_stop()
    c1._writer_thread.join(5)

    writer.fail = False
    c2 = make_collector(spool_name="shared.db")
    assert c2.spool.count() >= left
    c2.start()
    assert wait_for(lambda: c2.spool.count() == 0, 30)
    assert len(writer.measurements()) >= 20

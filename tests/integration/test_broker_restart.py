"""확인 기준: 브로커를 껐다 켜도 1분 안에 수집이 다시 시작된다 (SC-001)."""
import time

import pytest

from tests.integration.conftest import wait_for

pytestmark = pytest.mark.broker


def test_resume_within_one_minute_after_broker_restart(broker, make_collector, writer, publisher):
    c = make_collector()
    c.start()
    assert c.connected.wait(10), "처음 접속 실패"
    publisher.publish_one()
    assert wait_for(lambda: writer.measurements(), 10), "브로커 정지 전 수집 안 됨"

    broker.stop()
    assert wait_for(lambda: not c.connected.is_set(), 10), "끊김 감지 못함"
    time.sleep(10)                        # 브로커가 10초 동안 꺼져 있음
    before = len(writer.measurements())
    broker.start()
    restarted = time.monotonic()

    def resumed():
        publisher.publish_one()
        return len(writer.measurements()) > before
    assert wait_for(resumed, 60, step=1), "브로커 재시작 후 60초 안에 수집이 재개되지 않음"
    elapsed = time.monotonic() - restarted
    print(f"\n브로커 재시작 → 첫 저장 {elapsed:.1f}초")
    assert elapsed <= 60

    kinds = [e["kind"] for e in writer.kinds("event")]
    assert "mqtt_disconnected" in kinds and kinds.count("mqtt_connected") >= 2

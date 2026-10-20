"""DB가 끊겨도 받은 측정값을 잃지 않는다 (SC-008), 스풀 한도 초과 시 오래된 것부터 버리고 기록한다."""
import pytest

from collector.spool import Spool
from tests.integration.conftest import wait_for

pytestmark = pytest.mark.broker


def test_db_outage_lossless(make_collector, writer, publisher):
    c = make_collector()
    c.start()
    assert c.connected.wait(10)
    writer.fail = True
    sent = 0
    for _ in range(600):
        publisher.publish_one().wait_for_publish(5)
        sent += 1
    # 수집기가 600건을 모두 받아 스풀에 쌓을 때까지(메시지 1건 → 측정값 여러 건)
    assert wait_for(lambda: c.spool.count() >= sent, 30), c.spool.count()
    assert not writer.measurements()
    writer.fail = False
    assert wait_for(lambda: c.spool.count() == 0, 60), f"스풀 잔여 {c.spool.count()}"

    meas = writer.measurements()
    keys = {(m["device_id"], m["item"], m["ts"]) for m in meas}
    assert len(keys) == len(meas)
    # 메시지 600건 → 측정값은 그보다 많다. 모든 메시지가 1건 이상 남았는지 확인
    assert len({(m["device_id"], m["frame_cnt"], m["ts"]) for m in meas}) >= 600 - 5
    kinds = [e["kind"] for e in writer.kinds("event")]
    assert "db_down" in kinds and "db_up" in kinds


def test_spool_cap_drops_oldest(tmp_path):
    s = Spool(tmp_path / "s.db", max_rows=10)
    s.put_many([("event", {"kind": "start"})])
    s.put_many([("measurement", {"ts": f"t{i}"}) for i in range(9)])
    dropped = s.put_many([("measurement", {"ts": f"t{i}"}) for i in range(9, 12)])
    assert dropped == {"dropped": 3, "from": "t0", "to": "t2"}
    assert s.count() == 10
    kinds = [k for _, k, _ in s.peek(100)]
    assert kinds[0] == "event"            # 사건은 버리지 않는다
    assert [r["ts"] for _, k, r in s.peek(100) if k == "measurement"][0] == "t3"


def test_spool_survives_reopen(tmp_path):
    s = Spool(tmp_path / "s.db")
    s.put_many([("measurement", {"ts": "a"}), ("measurement", {"ts": "b"})])
    seq = s.peek(1)[0][0]
    s.ack(seq)
    s.close()
    s2 = Spool(tmp_path / "s.db")
    assert s2.count() == 1 and s2.peek(10)[0][2]["ts"] == "b"


class PoisonError(Exception):
    pass


def test_bad_row_does_not_block_spool(make_collector, writer):
    """DB가 거부하는 행 하나가 스풀 전체를 막지 않는다: 그 행만 버리고 사건을 남긴다."""
    def strict(batch):
        if any(r.get("value") == -999 for _, r in batch):
            raise PoisonError("거부")
        writer(batch)
    strict.is_data_error = lambda e: isinstance(e, PoisonError)
    c = make_collector()
    c.writer = strict
    good = {"ts": "2026-10-20 06:00:00.000", "space_id": "lab", "device_id": "d", "item": "co2",
            "value": 500.0, "unit": "ppm", "source": "t", "received_at": "2026-10-20 06:00:00.000",
            "source_ts": None, "ts_corrected": 0, "frame_cnt": None}
    c.spool.put_many([("measurement", good), ("measurement", {**good, "value": -999}),
                      ("measurement", {**good, "item": "temperature", "value": 24.0})])
    assert c._flush_once() == 3
    assert c.spool.count() == 1                              # 남은 것은 bad_row 사건뿐
    assert [m["value"] for m in writer.measurements()] == [500.0, 24.0]
    c._flush_once()
    assert writer.kinds("event")[0]["kind"] == "bad_row"

import json
from datetime import datetime, timezone

import pytest

from collector.dedup import FrameDedup
from collector.parser import parse

pytestmark = pytest.mark.unit


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_same_frame_twice():
    d = FrameDedup()
    assert not d.is_duplicate(("dev", 7))
    assert d.is_duplicate(("dev", 7))
    assert not d.is_duplicate(("dev", 8))
    assert not d.is_duplicate(("other", 7))
    assert not d.is_duplicate(None)


def test_window_expiry_and_reset():
    c = Clock()
    d = FrameDedup(window_s=7200, clock=c)
    d.is_duplicate(("dev", 7))
    c.t = 7000
    assert d.is_duplicate(("dev", 7))
    c.t = 14300                      # 처음 본 뒤 2시간 넘음 → 재가입 뒤 같은 fCnt도 허용
    assert not d.is_duplicate(("dev", 7))
    assert not d.is_duplicate(("dev", 0))       # fCnt 리셋(작아짐)은 새 프레임


def test_real_chirpstack_double_publish(fixture_messages, config):
    """실측: 24e124126d152862의 fCnt 104222가 deduplicationId만 다른 두 메시지로 왔다."""
    d = FrameDedup()
    kept, dup = 0, 0
    for m in fixture_messages:
        if "24e124126d152862" not in m["topic"]:
            continue
        r = parse(m["topic"], m["payload"].encode(), datetime.fromtimestamp(m["t"], timezone.utc), config)
        if d.is_duplicate(r.frame_key):
            dup += 1
        else:
            kept += 1
    assert (kept, dup) == (2, 2)

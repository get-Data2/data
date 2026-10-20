"""LoRaWAN 재전송·중복 발행 걸러내기 (research R8).

실측(2026-10-20): ChirpStack이 같은 업링크(같은 fCnt)를 게이트웨이 조합만 다른 두 메시지로 발행했다.
deduplicationId와 nsTime(1ms 차이)이 달라 DB 유일키로는 못 막으므로 (장치, fCnt)를 시간 창 안에서 기억한다.
fCnt는 장치 재가입 때 0으로 돌아가므로 영구 키로 쓰지 않는다.
"""
from __future__ import annotations

import time
from collections import OrderedDict


class FrameDedup:
    def __init__(self, window_s: float = 7200, clock=time.monotonic):
        self.window_s = window_s
        self.clock = clock
        self._seen: OrderedDict[tuple, float] = OrderedDict()

    def is_duplicate(self, key: tuple | None) -> bool:
        """처음 보는 키면 기억하고 False, 창 안에서 이미 본 키면 True."""
        if key is None:
            return False
        now = self.clock()
        while self._seen:
            k, t = next(iter(self._seen.items()))
            if now - t <= self.window_s:
                break
            self._seen.popitem(last=False)
        if key in self._seen:
            return True
        self._seen[key] = now
        return False

    def __len__(self):
        return len(self._seen)

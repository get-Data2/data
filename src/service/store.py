"""최신 결과 저장소: 계산 루프가 쓰고 API·발행이 읽는다(잠금 보호)."""
from __future__ import annotations

import copy
import threading


class Store:
    def __init__(self):
        self._lock = threading.Lock()
        self._spaces: dict[str, dict] = {}
        self._health: dict = {"status": "starting", "last_cycle": None}
        self._alerts: list[dict] = []

    def set_cycle(self, spaces: dict[str, dict], alerts: list[dict], health: dict):
        with self._lock:
            self._spaces = copy.deepcopy(spaces)
            self._alerts = copy.deepcopy(alerts)
            self._health = copy.deepcopy(health)

    def set_health(self, **kw):
        with self._lock:
            self._health.update(kw)

    def space(self, s: str) -> dict | None:
        with self._lock:
            return copy.deepcopy(self._spaces.get(s))

    def spaces(self) -> dict[str, dict]:
        with self._lock:
            return copy.deepcopy(self._spaces)

    def alerts(self) -> list[dict]:
        with self._lock:
            return copy.deepcopy(self._alerts)

    def health(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._health)

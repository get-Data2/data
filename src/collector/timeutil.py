"""시각 해석·표시 도구. 저장은 모두 UTC, 표시는 KST."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
KST = timezone(timedelta(hours=9), "KST")

_FRAC = re.compile(r"\.(\d+)")


def parse_time(v) -> datetime | None:
    """ISO 문자열(나노초 포함), epoch 초/밀리초 숫자를 UTC datetime으로. 해석 불가면 None."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        x = float(v)
        if x > 1e14:      # 마이크로초
            x /= 1e6
        elif x > 1e11:    # 밀리초
            x /= 1e3
        if not 1e9 <= x <= 1e10:   # 2001~2286년 밖은 시각이 아님
            return None
        return datetime.fromtimestamp(x, UTC)
    if isinstance(v, str):
        s = v.strip()
        if not s or not s[0].isdigit():
            return None
        if s.isdigit():
            return parse_time(int(s))
        s = s.replace("Z", "+00:00")
        s = _FRAC.sub(lambda m: "." + m.group(1)[:6].ljust(6, "0"), s, count=1)
        try:
            t = datetime.fromisoformat(s)
        except ValueError:
            return None
        return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)
    return None


def to_naive_utc(t: datetime) -> datetime:
    """MySQL DATETIME 저장용(세션 time_zone '+00:00')."""
    return t.astimezone(UTC).replace(tzinfo=None)


def kst(t: datetime | None, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    if t is None:
        return "—"
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    return t.astimezone(KST).strftime(fmt)

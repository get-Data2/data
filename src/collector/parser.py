"""MQTT 메시지 → 측정값. 원천 규칙(설정)만으로 해석한다."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime

from paho.mqtt.client import topic_matches_sub

from .config import Config, Source
from .timeutil import UTC, parse_time

DT_FMT = "%Y-%m-%d %H:%M:%S.%f"


def dbtime(t: datetime) -> str:
    """UTC aware datetime → MySQL DATETIME(3) 문자열(JSON 스풀에도 그대로 들어간다)."""
    return t.astimezone(UTC).strftime(DT_FMT)[:-3]


@dataclass
class ParseResult:
    source: str | None = None
    measurements: list[dict] = field(default_factory=list)
    devices: list[dict] = field(default_factory=list)
    ignored: int = 0              # 숫자·참거짓이 아닌 항목 수
    ignored_topic: bool = False   # 측정값이 아닌 메시지(게이트웨이 상태, 수집 제외 장치)
    failed: str | None = None     # 해석 실패 사유
    lag_s: float | None = None    # 수신 시각 − 기준 시각
    frame_key: tuple | None = None


def match_source(topic: str, cfg: Config) -> Source | None:
    for s in cfg.sources:
        if any(topic_matches_sub(t, topic) for t in s.topics):
            return s
    return None


def _str(expr, payload, topic):
    if expr is None:
        return None
    v = expr.eval(payload, topic)
    if isinstance(v, list):
        v = v[0] if v else None
    return None if v is None else str(v)


def _value(v):
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    return None


def parse(topic: str, payload: bytes, received_at: datetime, cfg: Config) -> ParseResult:
    if any(topic_matches_sub(t, topic) for t in cfg.ignore_topics):
        return ParseResult(ignored_topic=True)
    src = match_source(topic, cfg)
    if src is None:
        return ParseResult(ignored_topic=True)
    res = ParseResult(source=src.name)
    try:
        data = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        res.failed = "JSON 아님"
        return res
    if not isinstance(data, dict):
        res.failed = "JSON 객체 아님"
        return res

    device_id = _str(src.device_id, data, topic)
    if not device_id:
        res.failed = "device_id 없음"
        return res
    ov = cfg.devices.get(device_id)
    if ov and not ov.enabled:
        return ParseResult(source=src.name, ignored_topic=True)

    # 기준 시각: 설정의 우선순위 → 허용 범위 밖이면 수신 시각
    ts = None
    for e in src.time:
        ts = parse_time(e.eval(data, topic))
        if ts is not None:
            break
    corrected = 0
    if ts is None:
        ts = received_at
    else:
        diff = (ts - received_at).total_seconds()
        if diff > cfg.timing["max_future_s"] or -diff > cfg.timing["max_past_s"]:
            ts, corrected = received_at, 1
    res.lag_s = (received_at - ts).total_seconds()

    src_ts = parse_time(src.source_time.eval(data, topic)) if src.source_time else None
    fcnt = src.frame_cnt.eval(data, topic) if src.frame_cnt else None
    fcnt = fcnt if isinstance(fcnt, int) and not isinstance(fcnt, bool) and fcnt >= 0 else None
    if fcnt is not None:
        res.frame_key = (device_id, fcnt)

    raw_loc = _str(src.space, data, topic)
    space_id, mapped_by = cfg.resolve_space(device_id, raw_loc)

    groups = src.measurements.eval_items(data, topic)
    if not groups or not all(isinstance(g, dict) for g, _ in groups):
        res.failed = "측정값 위치에 사전이 없음"
        return res

    common = {
        "ts": dbtime(ts), "space_id": space_id, "device_id": device_id, "source": src.name,
        "received_at": dbtime(received_at), "source_ts": dbtime(src_ts) if src_ts else None,
        "ts_corrected": corrected, "frame_cnt": fcnt,
    }
    seen = set()
    for g, _key in groups:
        for fname, raw in g.items():
            v = _value(raw)
            if v is None:
                res.ignored += 1
                continue
            item = cfg.item_name(str(fname))
            if item in seen:          # 별칭 두 개가 같은 항목으로 모이면 첫 값만
                continue
            seen.add(item)
            res.measurements.append({**common, "item": item, "value": v, "unit": cfg.item_unit(item)})

    res.devices.append({
        "device_id": device_id, "source": src.name,
        "name": _str(src.device_name, data, topic), "model": _str(src.model, data, topic),
        "space_id": space_id, "spot": _str(src.spot, data, topic), "raw_location": raw_loc,
        "mapped_by": mapped_by, "expected_period_s": cfg.expected_period_s(device_id),
        "first_seen": dbtime(received_at), "last_seen": dbtime(received_at),
    })
    return res

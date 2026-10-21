"""수집기 설정(`config/collector.yaml`) 읽기·검증과 공간 매핑.

형식은 specs/001-data-collection/contracts/config.md를 따른다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .pathexpr import Expr, PathExprError, compile_expr

UNMAPPED = "unmapped"
_SPACE_ID = re.compile(r"^[a-z][a-z0-9_]{1,31}$")

DEFAULTS: dict[str, dict[str, Any]] = {
    "timing": {"max_future_s": 120, "max_past_s": 3600},
    "mqtt": {"keepalive_s": 30, "reconnect_max_s": 30, "stale_reconnect_s": 300, "qos": 1,
             "client_id": "team2-collector"},
    "storage": {"table_prefix": "", "batch_rows": 1000, "flush_interval_s": 2},
    "spool": {"path": "var/spool.db", "max_rows": 400000},
    "report": {"stale_after_s": 600, "gap_min_minutes": 2},
    "backfill": {"bucket": "iot-bucket", "measurements": [], "chunk_hours": 24,
                 "progress_path": "var/backfill_progress.json", "batch_rows": 5000},
}


class ConfigError(ValueError):
    def __init__(self, path, errors: list[str]):
        self.errors = [f"{path}: {e}" for e in errors]
        super().__init__("설정 오류\n  " + "\n  ".join(self.errors))


@dataclass(frozen=True)
class Space:
    id: str
    name: str
    kind: str
    aliases: tuple[str, ...] = ()
    area_m2: float | None = None
    note: str | None = None


@dataclass(frozen=True)
class DeviceOverride:
    space: str | None = None
    expected_period_s: int | None = None
    enabled: bool = True
    periodic: bool = True          # False: 사건이 있을 때만 보내는 센서(문 열림 등) — 커버율·결측 판정에서 뺀다


@dataclass(frozen=True)
class ItemDef:
    name: str
    unit: str | None
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class Source:
    name: str
    topics: tuple[str, ...]
    device_id: Expr
    measurements: Expr
    time: tuple[Expr, ...] = ()
    device_name: Expr | None = None
    model: Expr | None = None
    space: Expr | None = None
    spot: Expr | None = None
    source_time: Expr | None = None
    frame_cnt: Expr | None = None


@dataclass(frozen=True)
class Config:
    path: str
    spaces: dict[str, Space]
    devices: dict[str, DeviceOverride]
    items: dict[str, ItemDef]
    sources: tuple[Source, ...]
    ignore_topics: tuple[str, ...]
    timing: dict[str, Any]
    mqtt: dict[str, Any]
    storage: dict[str, Any]
    spool: dict[str, Any]
    report: dict[str, Any]
    backfill: dict[str, Any]
    _alias_space: dict[str, str] = field(default_factory=dict, repr=False)
    _alias_item: dict[str, str] = field(default_factory=dict, repr=False)

    def resolve_space(self, device_id: str | None, raw_location: str | None) -> tuple[str, str]:
        """공간 결정: ① 장치 지정 → ② 원천 표기의 이름·별칭 → ③ unmapped. (space_id, mapped_by)."""
        ov = self.devices.get(device_id or "")
        if ov and ov.space:
            return ov.space, "device"
        if raw_location is not None:
            sid = self._alias_space.get(str(raw_location).strip())
            if sid:
                return sid, "alias"
        return UNMAPPED, "none"

    def item_name(self, field_name: str) -> str:
        """원천 필드 이름 → 표준 항목 이름. 대응표에 없으면 원래 이름."""
        return self._alias_item.get(field_name, field_name)

    def item_unit(self, item: str) -> str | None:
        d = self.items.get(item)
        return d.unit if d else None

    def expected_period_s(self, device_id: str) -> int:
        ov = self.devices.get(device_id)
        return (ov.expected_period_s if ov and ov.expected_period_s else 60)

    def is_periodic(self, device_id: str) -> bool:
        ov = self.devices.get(device_id)
        return ov.periodic if ov else True

    def all_topics(self) -> list[str]:
        out = []
        for s in self.sources:
            out.extend(s.topics)
        out.extend(self.ignore_topics)
        return list(dict.fromkeys(out))


def valid_topic_filter(t: str) -> bool:
    if not isinstance(t, str) or not t or len(t) > 65535:
        return False
    levels = t.split("/")
    for i, lv in enumerate(levels):
        if "#" in lv and (lv != "#" or i != len(levels) - 1):
            return False
        if "+" in lv and lv != "+":
            return False
    return True


def load_config(path: str | Path) -> Config:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(path, ["파일이 없음"])
    except yaml.YAMLError as e:
        raise ConfigError(path, [f"YAML 구문 오류: {e}"])
    return build_config(raw, str(path))


def build_config(raw: Any, path: str = "<설정>") -> Config:
    errs: list[str] = []
    if not isinstance(raw, dict):
        raise ConfigError(path, ["최상위는 사전이어야 함"])
    if raw.get("version") != 1:
        errs.append("version: 1이어야 함")

    # spaces
    spaces: dict[str, Space] = {}
    alias_space: dict[str, str] = {}
    raw_spaces = raw.get("spaces")
    if not isinstance(raw_spaces, list) or not raw_spaces:
        errs.append("spaces: 공간 목록이 필요함")
        raw_spaces = []
    for i, s in enumerate(raw_spaces):
        where = f"spaces[{i}]"
        if not isinstance(s, dict):
            errs.append(f"{where}: 사전이어야 함")
            continue
        sid, name, kind = s.get("id"), s.get("name"), s.get("kind")
        if not isinstance(sid, str) or not _SPACE_ID.match(sid):
            errs.append(f"{where}.id: '{sid}'는 ^[a-z][a-z0-9_]{{1,31}}$ 형식이어야 함")
            continue
        if sid in spaces or sid == UNMAPPED:
            errs.append(f"{where}.id: '{sid}' 중복 또는 예약어")
            continue
        if not name or not kind:
            errs.append(f"{where}: name과 kind가 필요함")
        aliases = tuple(str(a).strip() for a in (s.get("aliases") or []))
        for a in dict.fromkeys((str(name).strip(), *aliases)):
            if a in alias_space and alias_space[a] != sid:
                errs.append(f"{where}.aliases: '{a}'가 공간 '{alias_space[a]}'와 겹침")
            alias_space[a] = sid
        spaces[sid] = Space(sid, str(name), str(kind), aliases, s.get("area_m2"), s.get("note"))
    spaces[UNMAPPED] = Space(UNMAPPED, "미지정", "unmapped")

    # devices
    devices: dict[str, DeviceOverride] = {}
    for did, d in (raw.get("devices") or {}).items():
        where = f"devices.{did}"
        d = d or {}
        if not isinstance(d, dict):
            errs.append(f"{where}: 사전이어야 함")
            continue
        sp = d.get("space")
        if sp is not None and sp not in spaces:
            errs.append(f"{where}.space: 없는 공간 id '{sp}'")
        per = d.get("expected_period_s")
        if per is not None and (not isinstance(per, int) or per <= 0):
            errs.append(f"{where}.expected_period_s: 양의 정수여야 함")
        unknown = set(d) - {"space", "expected_period_s", "enabled", "periodic"}
        if unknown:
            errs.append(f"{where}: 알 수 없는 키 {sorted(unknown)}")
        devices[str(did)] = DeviceOverride(sp, per, bool(d.get("enabled", True)), bool(d.get("periodic", True)))

    # items
    items: dict[str, ItemDef] = {}
    alias_item: dict[str, str] = {}
    for name, d in (raw.get("items") or {}).items():
        d = d or {}
        aliases = tuple(str(a) for a in (d.get("aliases") or []))
        items[name] = ItemDef(name, d.get("unit"), aliases)
        for a in aliases:
            if a in alias_item and alias_item[a] != name:
                errs.append(f"items.{name}.aliases: '{a}'가 항목 '{alias_item[a]}'와 겹침")
            alias_item[a] = name

    # sources
    def expr(where, v, required=False):
        if v is None:
            if required:
                errs.append(f"{where}: 필수")
            return None
        try:
            return compile_expr(str(v))
        except PathExprError as e:
            errs.append(f"{where}: {e}")
            return None

    sources: list[Source] = []
    names = set()
    raw_sources = raw.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        errs.append("sources: 원천 규칙 목록이 필요함")
        raw_sources = []
    for i, s in enumerate(raw_sources):
        where = f"sources[{i}]"
        if not isinstance(s, dict):
            errs.append(f"{where}: 사전이어야 함")
            continue
        name = s.get("name")
        if not name or name in names:
            errs.append(f"{where}.name: 비었거나 중복")
        names.add(name)
        topics = [s.get("topic"), *(s.get("also_topics") or [])]
        for t in topics:
            if not valid_topic_filter(t):
                errs.append(f"{where}.topic: 올바른 MQTT 구독 패턴이 아님 '{t}'")
        times = s.get("time") or []
        if isinstance(times, str):
            times = [times]
        texprs = [expr(f"{where}.time[{j}]", t) for j, t in enumerate(times)]
        src = Source(
            name=str(name),
            topics=tuple(t for t in topics if isinstance(t, str)),
            device_id=expr(f"{where}.device_id", s.get("device_id"), True),
            measurements=expr(f"{where}.measurements", s.get("measurements"), True),
            time=tuple(t for t in texprs if t is not None),
            device_name=expr(f"{where}.device_name", s.get("device_name")),
            model=expr(f"{where}.model", s.get("model")),
            space=expr(f"{where}.space", s.get("space")),
            spot=expr(f"{where}.spot", s.get("spot")),
            source_time=expr(f"{where}.source_time", s.get("source_time")),
            frame_cnt=expr(f"{where}.frame_cnt", s.get("frame_cnt")),
        )
        sources.append(src)

    ignore = tuple(raw.get("ignore_topics") or [])
    for t in ignore:
        if not valid_topic_filter(t):
            errs.append(f"ignore_topics: 올바른 MQTT 구독 패턴이 아님 '{t}'")

    sections = {}
    for sec, dflt in DEFAULTS.items():
        val = raw.get(sec) or {}
        if not isinstance(val, dict):
            errs.append(f"{sec}: 사전이어야 함")
            val = {}
        unknown = set(val) - set(dflt)
        if unknown:
            errs.append(f"{sec}: 알 수 없는 키 {sorted(unknown)}")
        sections[sec] = {**dflt, **val}
    if not 0 <= sections["mqtt"]["qos"] <= 2:
        errs.append("mqtt.qos: 0~2")

    known = {"version", "spaces", "devices", "items", "sources", "ignore_topics", *DEFAULTS}
    unknown_top = set(raw) - known
    if unknown_top:
        errs.append(f"알 수 없는 최상위 키 {sorted(unknown_top)}")

    if errs:
        raise ConfigError(path, errs)
    return Config(path, spaces, devices, items, tuple(sources), ignore, _alias_space=alias_space,
                  _alias_item=alias_item, **sections)

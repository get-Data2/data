"""config/service.yaml 읽기·검증 (contracts/config.md)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from analysis.config import load_analysis_config

SPACE_NAME = {"lab": "실습실", "office": "사무실", "meeting": "회의실"}


class ServiceConfigError(ValueError):
    def __init__(self, path, errors: list[str]):
        self.errors = errors
        super().__init__(f"{path}: " + "; ".join(errors))


@dataclass(frozen=True)
class ServiceConfig:
    path: str
    raw: dict[str, Any]
    acfg: Any

    def __getitem__(self, k):
        return self.raw[k]

    @property
    def spaces(self) -> list[str]:
        return list(self.raw["spaces"])


def load_service_config(path: str | Path = "config/service.yaml") -> ServiceConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    errs = []
    acfg = load_analysis_config(raw.get("analysis_config", "config/analysis.yaml"),
                                raw.get("collector_config", "config/collector.yaml"))
    known = set(acfg["space"]["targets"]) | {"lab", "office", "meeting"}
    for s in raw.get("spaces") or []:
        if s not in known:
            errs.append(f"spaces: 분석 설정에 없는 공간 '{s}'")
    f = raw.get("freshness", {})
    if f.get("stale_min", 0) >= f.get("stop_min", 0):
        errs.append("freshness: stale_min은 stop_min보다 작아야 함")
    if raw.get("mqtt", {}).get("target") not in ("team", "local"):
        errs.append("mqtt.target: team 또는 local")
    if int(raw.get("api", {}).get("port", 0)) < 1024:
        errs.append("api.port: 1024 이상(sudo 없이 열 수 있는 포트)")
    if not 0 <= int(raw.get("cycle", {}).get("offset_s", -1)) <= 50:
        errs.append("cycle.offset_s: 0~50")
    if errs:
        raise ServiceConfigError(path, errs)
    return ServiceConfig(str(path), raw, acfg)

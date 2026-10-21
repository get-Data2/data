"""config/anomaly.yaml 읽기·검증 (contracts/config.md)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from analysis.config import load_analysis_config

KINDS = ("F1", "F2", "F3", "F4", "F5")
KIND_NAME = {"F1": "튄 값", "F2": "값 고정", "F3": "영점 틀어짐", "F4": "냉방 정지", "F5": "환기 정지"}


class AnomalyConfigError(ValueError):
    def __init__(self, path, errors: list[str]):
        self.errors = errors
        super().__init__(f"{path}: " + "; ".join(errors))


@dataclass(frozen=True)
class AnomalyConfig:
    path: str
    raw: dict[str, Any]
    acfg: Any

    def __getitem__(self, k):
        return self.raw[k]

    @property
    def split(self):
        return self.acfg.split(self.raw["split"])

    @property
    def digest(self) -> str:
        r = {k: v for k, v in self.raw.items() if k != "inputs"}
        return hashlib.sha256(json.dumps(r, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]

    @property
    def spaces(self) -> list[str]:
        return list(self.raw["spaces"])

    def test_days(self) -> int:
        a, b = self.split.test
        return (b - a).days + 1

    def val_days(self) -> int:
        a, b = self.split.val
        return (b - a).days + 1


def _ranges(d, where, errs):
    if isinstance(d, list) and len(d) == 2 and all(isinstance(x, (int, float)) for x in d):
        if d[0] > d[1]:
            errs.append(f"{where}: [하한, 상한] 순서가 아님")
    elif isinstance(d, dict):
        for k, v in d.items():
            _ranges(v, f"{where}.{k}", errs)


def load_anomaly_config(path: str | Path = "config/anomaly.yaml") -> AnomalyConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    errs = []
    acfg = load_analysis_config(raw.get("analysis_config", "config/analysis.yaml"))
    if raw.get("split") not in acfg.splits:
        errs.append(f"split: 분석 설정에 없는 분할 '{raw.get('split')}'")
    if int(raw.get("inject", {}).get("per_kind", 0)) < 1:
        errs.append("inject.per_kind: 1 이상이어야 함")
    if float(raw.get("alarms", {}).get("target_per_day", 0)) <= 0:
        errs.append("alarms.target_per_day: 0보다 커야 함")
    for k in KINDS:
        _ranges(raw.get("inject", {}).get(k, {}), f"inject.{k}", errs)
    if errs:
        raise AnomalyConfigError(path, errs)
    return AnomalyConfig(str(path), raw, acfg)

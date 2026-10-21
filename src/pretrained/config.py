"""config/pretrained.yaml 읽기·검증 (contracts/config.md)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from analysis.config import load_analysis_config

ROOT = Path("var/pretrained")


class PretrainedConfigError(ValueError):
    def __init__(self, path, errors: list[str]):
        self.errors = errors
        super().__init__(f"{path}: " + "; ".join(errors))


@dataclass(frozen=True)
class PretrainedConfig:
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
    def horizons(self) -> list[int]:
        return [5, 15, 30, 60]

    def size_rank(self, name: str) -> int:
        order = self.raw.get("size_order") or list(self.raw["models"])
        return order.index(name) if name in order else len(order)


def load_pretrained_config(path: str | Path = "config/pretrained.yaml") -> PretrainedConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    errs = []
    acfg = load_analysis_config(raw.get("analysis_config", "config/analysis.yaml"))
    if raw.get("split") not in acfg.splits:
        errs.append(f"split: 분석 설정에 없는 분할 '{raw.get('split')}'")
    if not {0.1, 0.5, 0.9} <= set(raw.get("quantiles") or []):
        errs.append("quantiles: 0.1·0.5·0.9가 필요함")
    if any(int(x) % 5 or int(x) <= 0 for x in raw.get("context_min") or [0]):
        errs.append("context_min: 5분의 배수(양수)여야 함")
    if raw.get("missing") not in ("nan", "ffill"):
        errs.append("missing: nan 또는 ffill")
    if not raw.get("models"):
        errs.append("models: 비어 있음")
    if errs:
        raise PretrainedConfigError(path, errs)
    return PretrainedConfig(str(path), raw, acfg)

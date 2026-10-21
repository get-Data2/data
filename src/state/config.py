"""config/state.yaml 읽기·검증 (contracts/config.md)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from analysis.config import load_analysis_config

FEATURE_GROUPS = ("level", "delta", "shape", "obs", "time", "schedule")
FORBIDDEN_OCCUPANCY = ("time", "schedule")       # 사용 여부 정답이 일정에서 나오므로 입력 금지(README 과제 5)


class StateConfigError(ValueError):
    def __init__(self, path, errors: list[str]):
        self.errors = errors
        super().__init__(f"{path}: " + "; ".join(errors))


@dataclass(frozen=True)
class StateConfig:
    path: str
    raw: dict[str, Any]
    acfg: Any

    def __getitem__(self, k):
        return self.raw[k]

    @property
    def split(self):
        return self.acfg.split(self.raw["split"])

    @property
    def spaces(self) -> list[str]:
        return list(self.raw["spaces"])

    @property
    def digest(self) -> str:
        r = {k: v for k, v in self.raw.items() if k != "inputs"}
        return hashlib.sha256(json.dumps(r, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]


def validate(raw: dict, acfg) -> tuple[list[str], list[str]]:
    errs, warns = [], []
    if raw.get("split") not in acfg.splits:
        errs.append(f"split: 분석 설정에 없는 분할 '{raw.get('split')}'")
    groups = (raw.get("features") or {}).get("groups") or {}
    occ = groups.get("occupancy") or []
    bad = [g for g in occ if g in FORBIDDEN_OCCUPANCY]
    if bad:
        errs.append(f"features.groups.occupancy: {bad} 금지 — 사용 여부 입력에 일정 정보(시각·일정) 금지, 누수 실험은 leak 묶음만")
    for name, gs in groups.items():
        unk = [g for g in gs if g not in FEATURE_GROUPS]
        if unk:
            errs.append(f"features.groups.{name}: 알 수 없는 묶음 {unk}")
    lab = (raw.get("labels") or {}).get("vent") or {}
    rule = (raw.get("rules") or {}).get("vent") or {}
    if lab.get("threshold_ppm") is not None and rule.get("co2_ppm") is not None and lab["threshold_ppm"] <= rule["co2_ppm"]:
        warns.append("labels.vent.threshold_ppm ≤ rules.vent.co2_ppm — 규칙이 이미 초과 상태를 보게 됨")
    mr = (((raw.get("selection") or {}).get("vent") or {}).get("threshold") or {}).get("min_recall")
    if mr is not None and not (0 < float(mr) < 1):
        errs.append("selection.vent.threshold.min_recall: 0과 1 사이여야 함")
    bf = (raw.get("inputs") or {}).get("booking_file")
    if bf:
        import pandas as pd
        cols = set(pd.read_csv(bf, nrows=0).columns)
        if not {"space_id", "start", "end"} <= cols:
            errs.append(f"inputs.booking_file: 열 space_id·start·end가 필요함({bf})")
    return errs, warns


def load_state_config(path: str | Path = "config/state.yaml") -> StateConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    acfg = load_analysis_config(raw.get("analysis_config", "config/analysis.yaml"))
    errs, _ = validate(raw, acfg)
    if errs:
        raise StateConfigError(path, errs)
    return StateConfig(str(path), raw, acfg)

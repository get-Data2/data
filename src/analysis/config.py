"""분석 설정(config/analysis.yaml) 읽기·검증. 장치별 기대 주기는 과제 1 설정에서 읽는다."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

KST = timezone(timedelta(hours=9))
WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


class AnalysisConfigError(ValueError):
    def __init__(self, path, errors: list[str]):
        self.errors = [f"{path}: {e}" for e in errors]
        super().__init__("설정 오류\n  " + "\n  ".join(self.errors))


@dataclass(frozen=True)
class Split:
    name: str
    train: tuple[date, date]
    val: tuple[date, date]
    test: tuple[date, date]

    def utc(self, part: str) -> tuple[datetime, datetime]:
        """KST 날짜 [시작, 끝](끝 포함) → UTC [시작, 끝) 구간."""
        a, b = getattr(self, part)
        start = datetime.combine(a, time(0), KST).astimezone(timezone.utc)
        end = datetime.combine(b + timedelta(days=1), time(0), KST).astimezone(timezone.utc)
        return start, end


@dataclass(frozen=True)
class AnalysisConfig:
    path: str
    raw: dict[str, Any]
    splits: dict[str, Split]
    periods: dict[str, int] = field(default_factory=dict)      # 장치 → 기대 주기(초), 과제 1 설정
    periodic: dict[str, bool] = field(default_factory=dict)    # 장치 → 주기형 여부
    digest: str = ""

    def __getitem__(self, k):
        return self.raw[k]

    def split(self, name: str | None = None) -> Split:
        name = name or self.raw["split"]["active"]
        if name not in self.splits:
            raise AnalysisConfigError(self.path, [f"split: 없는 분할 버전 '{name}'"])
        return self.splits[name]

    def period_s(self, device_id: str) -> int:
        return self.periods.get(device_id, 60)

    def is_periodic(self, device_id: str) -> bool:
        return self.periodic.get(device_id, True)


def _d(s, where, errs):
    try:
        return date.fromisoformat(str(s))
    except ValueError:
        errs.append(f"{where}: 날짜 형식 YYYY-MM-DD가 아님 '{s}'")
        return None


def load_analysis_config(path: str | Path = "config/analysis.yaml",
                         collector_config: str | Path | None = "config/collector.yaml") -> AnalysisConfig:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise AnalysisConfigError(path, ["파일이 없음"])
    errs: list[str] = []
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise AnalysisConfigError(path, ["version: 1인 사전이어야 함"])
    for sec in ("extract", "axis_a", "axis_b", "space", "gaps", "interval", "split", "baseline", "schedule", "report"):
        if not isinstance(raw.get(sec), dict):
            errs.append(f"{sec}: 필수 섹션")
    if errs:
        raise AnalysisConfigError(path, errs)

    splits = {}
    for name, v in (raw["split"].get("versions") or {}).items():
        parts = {}
        for part in ("train", "val", "test"):
            rng = v.get(part)
            if not (isinstance(rng, list) and len(rng) == 2):
                errs.append(f"split.versions.{name}.{part}: [시작, 끝] 필요")
                continue
            a, b = _d(rng[0], f"split.versions.{name}.{part}[0]", errs), _d(rng[1], f"split.versions.{name}.{part}[1]", errs)
            if a and b and a > b:
                errs.append(f"split.versions.{name}.{part}: 시작이 끝보다 늦음")
            parts[part] = (a, b)
        if len(parts) == 3 and all(x[0] and x[1] for x in parts.values()):
            for p, q in (("train", "val"), ("val", "test")):
                if parts[p][1] >= parts[q][0]:
                    errs.append(f"split.versions.{name}: {p}와 {q}가 겹치거나 순서가 틀림")
                elif (parts[q][0] - parts[p][1]).days < 2:
                    errs.append(f"split.versions.{name}: {p}와 {q} 사이 간격이 1일 미만(1시간 평균·어제 같은 시각이 경계를 넘음)")
            splits[name] = Split(name, parts["train"], parts["val"], parts["test"])
    if raw["split"].get("active") not in splits:
        errs.append(f"split.active: 없는 분할 버전 '{raw['split'].get('active')}'")

    for i, row in enumerate(raw["schedule"].get("rows") or []):
        bad = set(row.get("days", [])) - set(WEEKDAYS)
        if bad:
            errs.append(f"schedule.rows[{i}].days: 알 수 없는 요일 {sorted(bad)}")
    if raw["interval"]["minutes"] not in raw["interval"]["candidates"]:
        errs.append("interval.minutes: candidates 안에 있어야 함")
    if errs:
        raise AnalysisConfigError(path, errs)

    periods, periodic = {}, {}
    if collector_config and Path(collector_config).exists():
        cc = yaml.safe_load(Path(collector_config).read_text(encoding="utf-8")) or {}
        for dev, d in (cc.get("devices") or {}).items():
            d = d or {}
            if d.get("expected_period_s"):
                periods[str(dev)] = int(d["expected_period_s"])
            if d.get("periodic") is False:
                periodic[str(dev)] = False
    # 보고서 위치·글꼴(report)은 결과 값에 영향이 없으므로 규칙 버전(해시)에서 뺀다
    rules = {k: v for k, v in raw.items() if k != "report"}
    digest = hashlib.sha256(json.dumps(rules, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]
    return AnalysisConfig(str(path), raw, splits, periods, periodic, digest)

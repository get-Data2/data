"""config/forecast.yaml 읽기·검증, 실험 설정과 해시 (contracts/config.md, data-model "실험 설정")."""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from analysis.config import load_analysis_config

FEATURES = ("sensors", "time", "space", "schedule", "booking")
MODELS = ("gru", "lstm", "tcn")
STAGE_ORDER = ("D0", "D1", "D2", "D3", "D4", "B1", "S", "A2", "A3")


class ForecastConfigError(ValueError):
    def __init__(self, path, errors: list[str]):
        self.errors = errors
        super().__init__(f"{path}: " + "; ".join(errors))


@dataclass(frozen=True)
class ExperimentConfig:
    model: str = "gru"
    size: dict = field(default_factory=dict)
    lookback_min: int = 120
    features: tuple = ("sensors", "time", "space", "schedule")
    output: str = "direct"
    target: str = "delta"
    train_spaces: tuple = ("lab", "office", "meeting")
    norm: str = "per_space"               # per_space | pooled(학습 공간 합친 통계, 처음 보는 공간 실험)
    train: dict = field(default_factory=dict)
    seeds: tuple = (0, 1, 2)

    def key(self) -> dict:
        """해시 대상: 씨앗을 뺀 실험 설정."""
        d = asdict(self)
        d.pop("seeds")
        d["features"] = sorted(d["features"])
        return d

    @property
    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.key(), sort_keys=True).encode()).hexdigest()[:12]

    def label(self) -> str:
        f = "+".join(x for x in FEATURES if x in self.features)
        lb = f"{self.lookback_min // 60}h" if self.lookback_min % 60 == 0 else f"{self.lookback_min}m"
        s = "" if set(self.train_spaces) == {"lab", "office", "meeting"} else " 학습=" + ",".join(self.train_spaces)
        return f"{self.model} {lb} [{f}] {self.output} {self.target}{s}"

    def simplicity(self) -> tuple:
        """동률일 때 고를 순서: 짧은 과거, 적은 입력, direct, delta, gru<lstm<tcn."""
        return (self.lookback_min, len(self.features), self.output != "direct", self.target != "delta",
                MODELS.index(self.model))


@dataclass(frozen=True)
class ForecastConfig:
    path: str
    raw: dict[str, Any]
    acfg: Any

    def __getitem__(self, k):
        return self.raw[k]

    @property
    def split(self):
        return self.acfg.split(self.raw["split"])

    def defaults(self) -> ExperimentConfig:
        d = self.raw["defaults"]
        return ExperimentConfig(model=d["model"], size=dict(d["size"]), lookback_min=int(d["lookback_min"]),
                                features=tuple(d["features"]), output=d["output"], target=d["target"],
                                train_spaces=tuple(d["train_spaces"]), train=dict(d["train"]),
                                seeds=tuple(d["seeds"]))

    def stage_candidates(self, stage: str, base: ExperimentConfig) -> list[ExperimentConfig]:
        """앞 단계 선택(base)에 이 단계의 후보 값을 하나씩 적용한다. B1은 곱집합."""
        if stage == "D0":
            return [base]
        if stage == "S":       # 일정 가치 쌍(US5) — 선택에 쓰지 않음
            return [base, replace(base, features=tuple(f for f in base.features if f != "schedule"))]
        if stage == "A2":      # 미래 예약 입력(심화 2)
            return [base, replace(base, features=tuple(base.features) + ("booking",))]
        if stage == "A3":      # 처음 보는 공간(심화 3): space 끔, 합친 통계
            nf = tuple(f for f in base.features if f != "space")
            spaces = tuple(self.raw["spaces"])
            ref = replace(base, features=nf, norm="pooled", train_spaces=spaces)
            return [ref] + [replace(ref, train_spaces=tuple(s for s in spaces if s != out)) for out in spaces]
        grid = self.raw["stages"][stage]
        keys = list(grid)
        out = []
        for combo in itertools.product(*(grid[k] for k in keys)):
            kw = {}
            for k, v in zip(keys, combo):
                kw[k] = tuple(v) if isinstance(v, list) else v
            out.append(replace(base, **kw))
        return out


def load_forecast_config(path: str | Path = "config/forecast.yaml") -> ForecastConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    errs = []
    acfg = load_analysis_config(raw.get("analysis_config", "config/analysis.yaml"))
    if raw.get("split") not in acfg.splits:
        errs.append(f"split: 분석 설정에 없는 분할 '{raw.get('split')}'")
    d = raw.get("defaults") or {}
    if d.get("model") not in MODELS:
        errs.append(f"defaults.model: {MODELS} 중 하나여야 함")
    feats = [d.get("features") or []] + list((raw.get("stages", {}).get("D2", {}) or {}).get("features", []))
    for fs in feats:
        bad = [f for f in fs if f not in FEATURES]
        if bad:
            errs.append(f"features: 알 수 없는 이름 {bad}")
        if "sensors" not in fs:
            errs.append(f"features: sensors가 없음 {fs}")
    lbs = [d.get("lookback_min", 0)] + [x for st in (raw.get("stages") or {}).values() for x in st.get("lookback_min", [])]
    if any(int(x) % 5 or int(x) <= 0 for x in lbs):
        errs.append("lookback_min: 5분의 배수(양수)여야 함")
    if not d.get("seeds"):
        errs.append("defaults.seeds: 비어 있음")
    bf = (raw.get("booking") or {}).get("file")
    if bf:
        import pandas as pd
        cols = set(pd.read_csv(bf, nrows=0).columns)
        if not {"space_id", "start", "end"} <= cols:
            errs.append(f"booking.file: 열 space_id·start·end가 필요함({bf})")
    if errs:
        raise ForecastConfigError(path, errs)
    return ForecastConfig(str(path), raw, acfg)

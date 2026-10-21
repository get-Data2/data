"""과제 2 시험 고정물: 합성 원본 사본 만들기."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from tests.conftest import ROOT

UTC = timezone.utc


@pytest.fixture
def acfg():
    from analysis.config import load_analysis_config
    return load_analysis_config(ROOT / "config" / "analysis.yaml", ROOT / "config" / "collector.yaml")


def raw_frame(rows):
    """rows: (ts, space_id, device_id, item, value) 목록 → 원본 사본 형식 DataFrame."""
    df = pd.DataFrame(rows, columns=["ts", "space_id", "device_id", "item", "value"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["source"] = "influx"
    return df


@pytest.fixture
def make_raw():
    return raw_frame


def minutes(start, n, step_s=60):
    return [start + timedelta(seconds=step_s * i) for i in range(n)]


T0 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)

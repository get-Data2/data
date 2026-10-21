"""MySQL → 달별 Parquet 사본(research R1). '달별'은 저장 단위일 뿐, 분석은 사본 전체를 합쳐 쓴다.

닫힌 달(이번 달보다 앞)은 한 번 뽑으면 다시 읽지 않는다(--refresh로 강제). 수집 쓰기와 오래 겹치지 않게 한 달씩 읽는다.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from .config import KST, AnalysisConfig

RAW_COLS = ["ts", "space_id", "device_id", "item", "value", "source"]


def _months(start: date, today: date):
    y, m = start.year, start.month
    while (y, m) <= (today.year, today.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _bounds_utc(y: int, m: int) -> tuple[datetime, datetime]:
    a = datetime(y, m, 1, tzinfo=KST)
    b = datetime(y + (m == 12), 1 if m == 12 else m + 1, 1, tzinfo=KST)
    return a.astimezone(timezone.utc).replace(tzinfo=None), b.astimezone(timezone.utc).replace(tzinfo=None)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_dir(cfg: AnalysisConfig) -> Path:
    return Path(cfg["extract"]["cache_dir"])


def load_manifest(cfg) -> dict:
    p = cache_dir(cfg) / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {"months": {}}


def extract(cfg: AnalysisConfig, conn, refresh: bool = False, today: date | None = None, log=print) -> dict:
    """달마다 측정값을 읽어 사본을 만든다. 장치 목록도 함께 저장한다. 결과: manifest."""
    today = today or datetime.now(KST).date()
    root = cache_dir(cfg)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    man = load_manifest(cfg)
    items = cfg["extract"]["items"]
    ph = ",".join(["%s"] * len(items))
    start = date.fromisoformat(cfg["extract"]["start"])
    for y, m in _months(start, today):
        key = f"{y:04d}-{m:02d}"
        closed = (y, m) < (today.year, today.month)
        info = man["months"].get(key)
        path = root / "raw" / f"{key}.parquet"
        a, b = _bounds_utc(y, m)
        if info and info.get("closed") and path.exists() and not refresh:
            # 닫힌 달이라도 과거 적재(InfluxDB)로 행이 늘었으면 다시 읽는다 — 날짜만으로 판단하면 놓친다
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM measurement WHERE item IN ({ph}) AND ts >= %s AND ts < %s",
                            [*items, a, b])
                if cur.fetchone()[0] == info["rows"]:
                    continue
            log(f"  {key}: DB 행 수가 사본과 달라 다시 읽음")
        with conn.cursor() as cur:
            cur.execute(f"SELECT ts, space_id, device_id, item, value, source FROM measurement "
                        f"WHERE item IN ({ph}) AND ts >= %s AND ts < %s", [*items, a, b])
            rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=RAW_COLS)
        df["ts"] = pd.to_datetime(df["ts"], utc=True).astype("datetime64[ms, UTC]")
        df["value"] = df["value"].astype("float64")
        for c in ("space_id", "device_id", "item", "source"):
            df[c] = df[c].astype("category")
        df = df.sort_values(["device_id", "item", "ts"], kind="stable").reset_index(drop=True)
        df.to_parquet(path, index=False)
        man["months"][key] = {"rows": len(df), "sha256": _sha(path), "closed": closed,
                              "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        log(f"  {key}: {len(df):,}행{' (진행 중인 달)' if not closed else ''}")
    with conn.cursor() as cur:
        cur.execute("SELECT device_id, model, space_id, spot, source FROM device")
        dev = pd.DataFrame(cur.fetchall(), columns=["device_id", "model", "space_id", "spot", "source"])
    dev.to_parquet(root / "devices.parquet", index=False)
    (root / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1))
    return man


def load_raw(cfg: AnalysisConfig, start_utc=None, end_utc=None) -> pd.DataFrame:
    """사본 전체(또는 기간)를 합쳐 읽는다."""
    files = sorted((cache_dir(cfg) / "raw").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("사본이 없음 — 먼저 `analysis extract`")
    filters = None
    if start_utc is not None and end_utc is not None:
        filters = [("ts", ">=", pd.Timestamp(start_utc)), ("ts", "<", pd.Timestamp(end_utc))]
    parts = [pd.read_parquet(f, filters=filters) for f in files]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    for c in ("space_id", "device_id", "item", "source"):
        df[c] = df[c].astype("category")
    return df


def load_devices(cfg) -> pd.DataFrame:
    return pd.read_parquet(cache_dir(cfg) / "devices.parquet")

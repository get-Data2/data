"""InfluxDB 과거 데이터 → MySQL 적재 (US6, FR-026~031).

- InfluxDB는 조회만 한다. 끝 시각은 수집기의 첫 수신 시각(겹치면 0.2초 차이로 중복이 되므로).
- 공간 매핑·항목 별칭·수집 제외 장치는 수집기 설정을 그대로 쓴다.
- 초기 한 달(2025-08-08~09-03)은 device_id 태그가 없다 → (location, point)로 장치를 하나로 특정할 수 있으면 복원.
- 구간(기본 1일)마다 진행 파일에 기록해 끊겨도 이어서 하고, 유일키 덕분에 다시 해도 행이 늘지 않는다.
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from . import store
from .parser import dbtime
from .timeutil import UTC, kst, parse_time

log = logging.getLogger("collector.backfill")
SOURCE = "influx"


def _flt(cfg) -> str:
    ms = cfg.backfill["measurements"]
    return " or ".join(f'r._measurement=="{m}"' for m in ms)


def infer_table(rows: list[dict]) -> dict[tuple[str, str], str]:
    """(location, point) → device_id. 둘 이상의 장치가 같은 조합을 쓰면 특정할 수 없으므로 뺀다."""
    seen: dict[tuple[str, str], set[str]] = {}
    for r in rows:
        dev, loc, pt = r.get("device_id"), r.get("location"), r.get("point")
        if dev and loc and pt:
            seen.setdefault((loc, pt), set()).add(dev)
    return {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}


def convert(r: dict, cfg, infer: dict) -> tuple[dict | None, dict | None, str | None]:
    """InfluxDB 행 하나 → (측정값 행, 장치 정보, 건너뛴 사유)."""
    dev = r.get("device_id") or None
    loc, pt, spot = r.get("location") or None, r.get("point") or None, r.get("spot") or None
    if not dev:
        dev = infer.get((loc, pt)) if loc and pt else None
        if not dev:
            return None, None, "장치 특정 불가"
    ov = cfg.devices.get(dev)
    if ov and not ov.enabled:
        return None, None, "수집 제외 장치"
    try:
        v = float(r["_value"])
    except (KeyError, ValueError):
        return None, None, "숫자 아님"
    if not math.isfinite(v):
        return None, None, "숫자 아님"
    t = parse_time(r.get("_time"))
    if t is None:
        return None, None, "시각 없음"
    item = cfg.item_name(r["_measurement"])
    space_id, mapped_by = cfg.resolve_space(dev, loc)
    ts = dbtime(t)
    meas = {"ts": ts, "space_id": space_id, "device_id": dev, "item": item, "value": v,
            "unit": cfg.item_unit(item), "source": SOURCE, "received_at": ts, "source_ts": ts,
            "ts_corrected": 0, "frame_cnt": None}
    info = {"device_id": dev, "space_id": space_id, "mapped_by": mapped_by, "raw_location": loc,
            "spot": spot or pt, "t": t}
    return meas, info, None


class Backfill:
    def __init__(self, cfg, reader, conn, prefix: str = "", progress_path: str | None = None):
        self.cfg, self.reader, self.conn, self.prefix = cfg, reader, conn, prefix
        self.progress_path = Path(progress_path or cfg.backfill["progress_path"])
        self.progress = self._load()

    # ---------- 진행 기록 ----------

    def _load(self) -> dict:
        try:
            return json.loads(self.progress_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return {"done": {}, "skipped": {}}

    def _save(self):
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.progress_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.progress, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.progress_path)

    # ---------- 범위 ----------

    def cutoff(self) -> datetime:
        """수집기의 첫 수신 시각(UTC). 수집기 데이터가 없으면 지금."""
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT MIN(received_at) FROM `{self.prefix}measurement` WHERE source<>%s", [SOURCE])
            v = cur.fetchone()[0]
        return v.replace(tzinfo=UTC) if v else datetime.now(UTC)

    def first_time(self) -> datetime:
        b = self.reader.bucket
        rows = self.reader.query(f'from(bucket:"{b}") |> range(start: 0) |> filter(fn:(r)=> ({_flt(self.cfg)}) '
                                 f'and r._field=="value") |> group() |> first()')
        if not rows:
            raise RuntimeError("InfluxDB에 적재할 데이터가 없음")
        return parse_time(rows[0]["_time"])

    def build_infer(self, stop: datetime) -> dict:
        b = self.reader.bucket
        rows = self.reader.query(
            f'from(bucket:"{b}") |> range(start: 0, stop: {stop.strftime("%Y-%m-%dT%H:%M:%SZ")}) '
            f'|> filter(fn:(r)=> ({_flt(self.cfg)}) and r._field=="value" and exists r.device_id) '
            f'|> group(columns:["device_id","location","point"]) |> last() '
            f'|> keep(columns:["device_id","location","point"]) |> group()')
        return infer_table(rows)

    # ---------- 적재 ----------

    def load_chunk(self, start: datetime, stop: datetime, infer: dict) -> dict:
        b = self.reader.bucket
        rows = self.reader.query(
            f'from(bucket:"{b}") |> range(start: {start.strftime("%Y-%m-%dT%H:%M:%SZ")}, '
            f'stop: {stop.strftime("%Y-%m-%dT%H:%M:%S.%fZ")}) '
            f'|> filter(fn:(r)=> ({_flt(self.cfg)}) and r._field=="value") '
            f'|> keep(columns:["_time","_value","_measurement","device_id","location","point","spot"]) |> group()')
        skipped: Counter = Counter()
        meas, devs = [], {}
        for r in rows:
            m, info, why = convert(r, self.cfg, infer)
            if why:
                skipped[why] += 1
                continue
            meas.append(("measurement", m))
            d = devs.get(info["device_id"])
            if d is None or info["t"] > d["t_last"]:
                devs[info["device_id"]] = {**info, "t_first": d["t_first"] if d else info["t"], "t_last": info["t"]}
            elif info["t"] < d["t_first"]:
                d["t_first"] = info["t"]
        dev_rows = [("device_seen", {
            "device_id": dev, "source": SOURCE, "name": None, "model": None, "space_id": d["space_id"],
            "spot": d["spot"], "raw_location": d["raw_location"], "mapped_by": d["mapped_by"],
            "expected_period_s": self.cfg.expected_period_s(dev),
            "first_seen": dbtime(d["t_first"]), "last_seen": dbtime(d["t_last"])}) for dev, d in devs.items()]
        n = self.cfg.backfill["batch_rows"]
        for i in range(0, len(meas), n):
            self._write(meas[i:i + n])
        if dev_rows:
            self._write(dev_rows)
        return {"read": len(rows), "written": len(meas), "devices": len(devs), "skipped": dict(skipped)}

    def _with_network_retry(self, fn, give_up_s: float = 3600):
        """InfluxDB 망이 끊기면(2026-10-20 실제 발생: 유선망 경로 사라짐) 간격을 늘려 가며 최대 give_up_s 기다린다."""
        import urllib.error
        waited, delay = 0.0, 10.0
        while True:
            try:
                return fn()
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                if waited >= give_up_s:
                    raise
                log.warning("InfluxDB 접속 실패(%s) — %.0f초 뒤 재시도(누적 대기 %.0f분)", type(e).__name__, delay, waited / 60)
                time.sleep(delay)
                waited += delay
                delay = min(delay * 2, 300)

    def _write(self, batch, tries: int = 3):
        for k in range(tries):
            try:
                store.write_batch(self.conn, self.prefix, batch)
                return
            except Exception as e:
                if k == tries - 1:
                    raise
                log.warning("기록 실패(%s) — 다시 접속해 재시도", type(e).__name__)
                time.sleep(2 * (k + 1))
                try:
                    self.conn.ping(reconnect=True)
                except Exception:
                    pass

    def run(self, start: datetime | None = None, end: datetime | None = None, max_chunks: int | None = None,
            dry_run: bool = False) -> str:
        end = end or self.cutoff()
        start = start or self.first_time()
        step = timedelta(hours=self.cfg.backfill["chunk_hours"])
        t = start.replace(hour=0, minute=0, second=0, microsecond=0)
        chunks = []
        while t < end:
            chunks.append((t, min(t + step, end)))
            t += step
        todo = [(a, b) for a, b in chunks if self.progress["done"].get(a.isoformat(), {}).get("stop") != b.isoformat()]
        head = (f"적재 범위 {kst(start, '%Y-%m-%d %H:%M')} ~ {kst(end, '%Y-%m-%d %H:%M:%S')} KST, "
                f"구간 {len(chunks)}개 중 남은 {len(todo)}개")
        log.info(head)
        if dry_run:
            return head
        infer = self.build_infer(end)
        log.info("장치 추론표 %d개 (location, point) 조합", len(infer))
        total, skipped = Counter(), Counter()
        t0 = time.monotonic()
        for i, (a, b) in enumerate(todo[:max_chunks] if max_chunks else todo, 1):
            res = self._with_network_retry(lambda: self.load_chunk(a, b, infer))
            self.progress["done"][a.isoformat()] = {"stop": b.isoformat(), **res}
            self._save()
            total.update({"read": res["read"], "written": res["written"]})
            skipped.update(res["skipped"])
            rate = total["written"] / max(time.monotonic() - t0, 1e-6)
            log.info("[%d/%d] %s %s행 적재, 건너뜀 %s (누적 %s행, %.0f행/초)", i, len(todo),
                     kst(a, "%Y-%m-%d"), f"{res['written']:,}", res["skipped"] or 0, f"{total['written']:,}", rate)
        remaining = [a for a, b in chunks
                     if self.progress["done"].get(a.isoformat(), {}).get("stop") != b.isoformat()]
        if not remaining:
            self.progress_path.with_name("backfill_done").write_text(datetime.now(UTC).isoformat())
            log.info("모든 구간 적재 완료 — 완료 표시 파일을 남김(타이머가 더 돌지 않음)")
        return summarize(self.progress, head)


def summarize(progress: dict, head: str = "") -> str:
    done = progress["done"]
    written = sum(v["written"] for v in done.values())
    read = sum(v["read"] for v in done.values())
    skipped: Counter = Counter()
    for v in done.values():
        skipped.update(v["skipped"])
    lines = [head, f"완료 구간 {len(done)}개, 읽은 행 {read:,}, 적재 {written:,}"]
    if skipped:
        lines.append("건너뜀: " + ", ".join(f"{k} {v:,}" for k, v in skipped.most_common()))
    return "\n".join(x for x in lines if x)

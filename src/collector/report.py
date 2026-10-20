"""수집 현황 보고: 공간·항목별 건수, 마지막 수신, 커버율, 빠진 구간과 원인 단서 (FR-013~017).

커버율 = 기간 안 1분 칸 중 값이 하나 이상 있는 칸의 비율(칸은 기준 시각 ts로 나눈다).
기대 주기가 1분보다 긴 장치는 주기 크기의 칸으로 센다. 공간·항목 합산은 장치 중 하나라도 값이 있으면 채운 칸이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .timeutil import KST, UTC, kst

# ---------- 계산(순수 함수) ----------


def _bucket_range(start_min: int, end_min: int, k: int) -> range:
    return range(start_min // k, (end_min - 1) // k + 1)


def coverage(minutes: set[int], start_min: int, end_min: int, period_s: int = 60) -> float:
    """[start_min, end_min) 안의 칸 중 값이 있는 칸 비율. 분 번호는 epoch 분."""
    k = max(1, period_s // 60)
    expected = _bucket_range(start_min, end_min, k)
    if not len(expected):
        return 0.0
    have = {m // k for m in minutes if start_min <= m < end_min}
    return len(have) / len(expected)


def gaps(minutes: set[int], start_min: int, end_min: int, period_s: int = 60,
         min_minutes: int = 2) -> list[tuple[int, int]]:
    """연속으로 빈 칸 구간 [시작분, 끝분). 길이가 min_minutes(또는 주기) 이상인 것만."""
    k = max(1, period_s // 60)
    have = {m // k for m in minutes}
    out, run = [], None
    for b in _bucket_range(start_min, end_min, k):
        if b not in have:
            run = b if run is None else run
        elif run is not None:
            out.append((run, b))
            run = None
    if run is not None:
        out.append((run, (end_min - 1) // k + 1))
    res = []
    for a, b in out:
        s, e = max(a * k, start_min), min(b * k, end_min)
        if e - s >= max(min_minutes, k):
            res.append((s, e))
    return res


@dataclass
class Event:
    minute: float      # epoch 분(소수)
    kind: str
    detail: str | None = None


def cause_hint(gap: tuple[int, int], device_id: str | None, device_minutes: dict[str, set[int]],
               stats_minutes: set[int], events: list[Event], battery: float | None = None) -> str:
    """빠진 구간 하나의 원인 단서."""
    s, e = gap
    span = range(s, e)
    n = max(1, len(span))
    hints = []
    no_stats = sum(1 for m in span if m not in stats_minutes) / n
    near = [ev for ev in events if s - 2 <= ev.minute <= e + 2
            and ev.kind in ("start", "stop", "mqtt_disconnected", "mqtt_connected", "stale_reconnect",
                            "spool_drop", "db_down")]
    ev_txt = ", ".join(f"{ev.kind} {kst(_dt(ev.minute), '%H:%M')}" for ev in near[:4])

    others = [d for d in device_minutes if d != device_id]
    active = [d for d in others if device_minutes[d]]
    if active:
        missing_share = sum(
            sum(1 for m in span if m not in device_minutes[d]) / n for d in active) / len(active)
    else:
        missing_share = 1.0

    if no_stats > 0.5:
        hints.append("수집기 미가동(분 통계 없음)")
    elif device_id is None or missing_share >= 0.8:
        if any(ev.kind in ("mqtt_disconnected", "stale_reconnect") for ev in near):
            hints.append("전 장치 동시 결측 + MQTT 끊김 → 브로커·네트워크")
        else:
            hints.append("전 장치 동시 결측, 수집기 정상 → 원천 측(게이트웨이·브로커)")
    else:
        hints.append("해당 장치만 결측 → 센서·배터리·전파")
        if battery is not None and battery <= 20:
            hints.append(f"배터리 {battery:.0f}%")
    if ev_txt:
        hints.append(f"사건: {ev_txt}")
    return "; ".join(hints)


def _dt(epoch_min: float) -> datetime:
    return datetime.fromtimestamp(epoch_min * 60, UTC)


def _min(t: datetime) -> int:
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    return int(t.timestamp() // 60)


# ---------- 조회 ----------

@dataclass
class Data:
    start: datetime
    end: datetime
    spaces: dict[str, tuple[str, str]] = field(default_factory=dict)        # id → (이름, 종류)
    devices: dict[str, dict] = field(default_factory=dict)
    series: dict[tuple[str, str], dict] = field(default_factory=dict)       # (space, item) → 통계
    space_item_minutes: dict[tuple[str, str], set[int]] = field(default_factory=dict)
    space_item_devices: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    device_minutes: dict[str, set[int]] = field(default_factory=dict)
    device_counts: dict[str, int] = field(default_factory=dict)
    stats_minutes: set[int] = field(default_factory=set)
    events: list[Event] = field(default_factory=list)
    battery: dict[str, float] = field(default_factory=dict)


def fetch(conn, prefix: str, start: datetime, end: datetime, space: str | None = None) -> Data:
    p = prefix
    s, e = start.astimezone(UTC).replace(tzinfo=None), end.astimezone(UTC).replace(tzinfo=None)
    d = Data(start, end)
    sp_where, sp_arg = (" AND space_id=%s", [space]) if space else ("", [])
    with conn.cursor() as cur:
        cur.execute(f"SELECT space_id, name, kind FROM `{p}space`")
        d.spaces = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
        cur.execute(f"SELECT device_id, source, name, model, space_id, raw_location, mapped_by, "
                    f"expected_period_s, last_seen FROM `{p}device`" + (" WHERE space_id=%s" if space else ""),
                    sp_arg)
        cols = ("device_id", "source", "name", "model", "space_id", "raw_location", "mapped_by",
                "expected_period_s", "last_seen")
        d.devices = {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}

        cur.execute(f"SELECT space_id, item, COUNT(*), MAX(received_at), COUNT(DISTINCT device_id) "
                    f"FROM `{p}measurement` WHERE ts>=%s AND ts<%s{sp_where} GROUP BY space_id, item",
                    [s, e, *sp_arg])
        for sid, item, n, last, ndev in cur.fetchall():
            d.series[(sid, item)] = {"count": n, "last": last, "devices": ndev}

        cur.execute(f"SELECT space_id, item, FLOOR(UNIX_TIMESTAMP(ts)/60) FROM `{p}measurement` "
                    f"WHERE ts>=%s AND ts<%s{sp_where} GROUP BY 1, 2, 3", [s, e, *sp_arg])
        for sid, item, m in cur.fetchall():
            d.space_item_minutes.setdefault((sid, item), set()).add(int(m))

        cur.execute(f"SELECT device_id, space_id, item FROM `{p}measurement` WHERE ts>=%s AND ts<%s{sp_where} "
                    f"GROUP BY 1, 2, 3", [s, e, *sp_arg])
        for dev, sid, item in cur.fetchall():
            d.space_item_devices.setdefault((sid, item), set()).add(dev)

        cur.execute(f"SELECT device_id, FLOOR(UNIX_TIMESTAMP(ts)/60), COUNT(*) FROM `{p}measurement` "
                    f"WHERE ts>=%s AND ts<%s{sp_where} GROUP BY 1, 2", [s, e, *sp_arg])
        for dev, m, n in cur.fetchall():
            d.device_minutes.setdefault(dev, set()).add(int(m))
            d.device_counts[dev] = d.device_counts.get(dev, 0) + n

        cur.execute(f"SELECT FLOOR(UNIX_TIMESTAMP(minute)/60) FROM `{p}ingest_stats` WHERE minute>=%s AND minute<%s",
                    [s - timedelta(minutes=1), e])
        d.stats_minutes = {int(r[0]) for r in cur.fetchall()}

        cur.execute(f"SELECT UNIX_TIMESTAMP(ts)/60, kind, detail FROM `{p}collector_event` "
                    f"WHERE ts>=%s AND ts<%s ORDER BY ts", [s - timedelta(minutes=5), e])
        d.events = [Event(float(m), k, det) for m, k, det in cur.fetchall()]

        cur.execute(f"SELECT m.device_id, m.value FROM `{p}measurement` m JOIN ("
                    f"SELECT device_id, MAX(ts) ts FROM `{p}measurement` WHERE item='battery' AND ts>=%s AND ts<%s "
                    f"GROUP BY device_id) x ON m.device_id=x.device_id AND m.ts=x.ts AND m.item='battery'", [s, e])
        d.battery = {dev: float(v) for dev, v in cur.fetchall()}
    for dev in d.devices:
        d.device_minutes.setdefault(dev, set())
    return d


# ---------- 표 만들기 ----------

def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _table(head: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join("" if c is None else str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _space_label(d: Data, sid: str) -> str:
    name = d.spaces.get(sid, (sid, ""))[0]
    return f"{name}({sid})"


def build_report(d: Data, cfg, synthetic: bool = False, by: str | None = None,
                 now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    s_min, e_min = _min(d.start), _min(d.end)
    hours = (d.end - d.start).total_seconds() / 3600
    gap_min = cfg.report["gap_min_minutes"]
    parts = [
        "# 수집 현황 보고",
        f"- 기간: {kst(d.start, '%Y-%m-%d %H:%M')} ~ {kst(d.end, '%Y-%m-%d %H:%M')} KST ({hours:g}h) · "
        f"{'합성 데이터' if synthetic else '실데이터'} · 생성 {kst(now, '%Y-%m-%d %H:%M')} KST",
        "- 커버율: 1분 칸(기준 시각) 중 값이 있는 칸 비율. 공간·항목은 장치 중 하나라도 있으면 채움. "
        "주기가 긴 장치는 주기 칸으로 계산",
    ]

    # 요약
    covs = {k: coverage(m, s_min, e_min) for k, m in d.space_item_minutes.items()}
    core = {k: v for k, v in covs.items() if k[1] in ("co2", "temperature", "humidity")}
    if core:
        low = sorted((v, k) for k, v in core.items() if v < 0.95)
        parts.append(f"- 핵심 항목(co2·temperature·humidity) 공간별 커버율 최저 {_pct(min(core.values()))}, "
                     f"95% 미만 {len(low)}개" + (": " + ", ".join(
                         f"{_space_label(d, k[0])} {k[1]} {_pct(v)}" for v, k in low[:6]) if low else ""))

    # ① 미매핑·끊김 장치
    stale_s = cfg.report["stale_after_s"]
    live_end = abs((now - d.end).total_seconds()) < 120
    rows = []
    for dev, info in sorted(d.devices.items()):
        problems = []
        if info["mapped_by"] == "none" or info["space_id"] == "unmapped":
            problems.append(f"공간 미지정(원천 표기 '{info['raw_location']}')")
        last = info["last_seen"]
        if live_end and last is not None:
            age = (d.end - last.replace(tzinfo=UTC)).total_seconds()
            if age > stale_s:
                problems.append(f"끊김: {age / 60:.0f}분째 수신 없음")
        if problems:
            rows.append([dev, info["model"], _space_label(d, info["space_id"]), kst(last), "; ".join(problems)])
    parts += ["", "## ① 확인 필요 장치(미매핑·끊김)",
              _table(["장치", "기종", "공간", "마지막 수신", "문제"], rows) if rows else "없음"]

    # ② 공간·항목
    if by in (None, "space"):
        rows = []
        for (sid, item), st in sorted(d.series.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            rows.append([_space_label(d, sid), item, st["devices"], f"{st['count']:,}", kst(st["last"]),
                         _pct(covs.get((sid, item), 0.0))])
        parts += ["", "## ② 공간·항목별", _table(["공간", "항목", "장치 수", "건수", "마지막 수신", "커버율"], rows)]

    # ③ 장치별
    if by in (None, "device"):
        rows = []
        for dev, info in sorted(d.devices.items(), key=lambda kv: (kv[1]["space_id"], kv[0])):
            per = info["expected_period_s"] or 60
            rows.append([dev, info["model"], _space_label(d, info["space_id"]), per,
                         f"{d.device_counts.get(dev, 0):,}", kst(info["last_seen"]),
                         _pct(coverage(d.device_minutes.get(dev, set()), s_min, e_min, per))])
        parts += ["", "## ③ 장치별",
                  _table(["장치", "기종", "공간", "주기(초)", "건수", "마지막 수신", "커버율"], rows)]

    # ④ 빠진 구간
    all_minutes = set().union(*d.device_minutes.values()) if d.device_minutes else set()
    global_gaps = gaps(all_minutes, s_min, e_min, 60, gap_min)
    global_set = {m for a, b in global_gaps for m in range(a, b)}
    rows = []
    for g in global_gaps:
        rows.append(["(전체)", "—", kst(_dt(g[0]), "%m-%d %H:%M"), kst(_dt(g[1]), "%m-%d %H:%M"),
                     f"{g[1] - g[0]}분", cause_hint(g, None, d.device_minutes, d.stats_minutes, d.events)])
    for dev, info in sorted(d.devices.items()):
        per = info["expected_period_s"] or 60
        mins = d.device_minutes.get(dev, set()) | global_set     # 전체 결측은 위에서 이미 보고
        for g in gaps(mins, s_min, e_min, per, gap_min):
            rows.append([dev, _space_label(d, info["space_id"]), kst(_dt(g[0]), "%m-%d %H:%M"),
                         kst(_dt(g[1]), "%m-%d %H:%M"), f"{g[1] - g[0]}분",
                         cause_hint(g, dev, d.device_minutes, d.stats_minutes, d.events, d.battery.get(dev))])
    parts += ["", f"## ④ 빠진 구간({gap_min}분 이상)",
              _table(["장치", "공간", "시작", "끝", "길이", "원인 단서"], rows) if rows else "없음"]
    return "\n".join(parts) + "\n"


# ---------- 명령 ----------

def _parse_kst(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=KST).astimezone(UTC)


def main(conn, cfg, args) -> str:
    prefix = cfg.storage["table_prefix"] if getattr(args, "prefix", None) is None else args.prefix
    if getattr(args, "capacity", False):
        from .capacity import capacity_report
        return capacity_report(conn, prefix)
    now = datetime.now(UTC)
    end = _parse_kst(args.to) if args.to else now
    start = _parse_kst(args.from_) if args.from_ else end - timedelta(hours=args.hours)
    d = fetch(conn, prefix, start, end, args.space)
    text = build_report(d, cfg, synthetic=args.synthetic, by=args.by, now=now)
    if args.md:
        from pathlib import Path
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(text, encoding="utf-8")
        text += f"\n(저장: {args.md})"
    return text

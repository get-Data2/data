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
        # 결측 = 자기 주기의 2배 이상 빈 경우(보고 한 번 손실은 결측으로 세지 않음 — 과제 2 spec과 같은 정의)
        if e - s >= max(min_minutes, 2 * k):
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

    mqtt_down = any(ev.kind in ("mqtt_disconnected", "stale_reconnect") and ev.detail != "Normal disconnection"
                    for ev in near)
    restarted = any(ev.kind in ("stop", "start") for ev in near)
    if no_stats > 0.5:
        hints.append("수집기 미가동(분 통계 없음)")
    elif mqtt_down:
        hints.append("MQTT 끊김(수집기 측) → 브로커·네트워크")
    elif restarted and e - s <= 5:
        hints.append("수집기 재시작 중 결측")
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


def observed_period_min(minutes: set[int]) -> float | None:
    """값이 있는 분 칸 사이 간격의 중앙값(분). 기대 주기 설정의 근거로 쓴다. 칸이 3개 미만이면 None."""
    ms = sorted(minutes)
    if len(ms) < 3:
        return None
    diffs = sorted(b - a for a, b in zip(ms, ms[1:]))
    return float(diffs[len(diffs) // 2])


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
    live_since: datetime | None = None    # 수집기 실시간 첫 수신
    influx_end: datetime | None = None    # 과거 적재 마지막 시각


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
        cur.execute(f"SELECT MIN(received_at) FROM `{p}measurement` WHERE source<>'influx'")
        v = cur.fetchone()[0]
        d.live_since = v.replace(tzinfo=UTC) if v else None
        cur.execute(f"SELECT MAX(ts) FROM `{p}measurement` WHERE source='influx'")
        v = cur.fetchone()[0]
        d.influx_end = v.replace(tzinfo=UTC) if v else None
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
        "- 커버율: 기대 주기 칸(기준 시각) 중 값이 있는 칸 비율. 공간·항목은 장치 중 하나라도 있으면 채움. "
        "빠진 구간은 자기 주기의 2배 이상 빈 경우만 셈",
    ]

    # 요약 — 공간·항목 커버율은 그 항목을 보내는 장치 중 가장 짧은 기대 주기의 칸으로 센다(10분 장치만 있는 회의실 등)
    def period(dev):
        if dev in cfg.devices and cfg.devices[dev].expected_period_s:
            return cfg.devices[dev].expected_period_s
        return (d.devices.get(dev) or {}).get("expected_period_s") or 60

    def series_period(key):
        pers = [period(x) for x in d.space_item_devices.get(key, ()) if cfg.is_periodic(x, key[1])]
        return min(pers) if pers else 60
    covs = {k: coverage(m, s_min, e_min, series_period(k)) for k, m in d.space_item_minutes.items()}
    # 주기적으로 보내는 장치가 하나도 없는 공간·항목(사건형·조건부)은 커버율 판정에서 뺀다
    judged = {k for k in covs if any(cfg.is_periodic(x, k[1]) for x in d.space_item_devices.get(k, ()))}
    core = {k: v for k, v in covs.items() if k[1] in ("co2", "temperature", "humidity") and k in judged}
    if core:
        low = sorted((v, k) for k, v in core.items() if v < 0.95)
        parts.append(f"- 핵심 항목(co2·temperature·humidity) 공간별 커버율 최저 {_pct(min(core.values()))}, "
                     f"95% 미만 {len(low)}개" + (": " + ", ".join(
                         f"{_space_label(d, k[0])} {k[1]} {_pct(v)}" for v, k in low[:6]) if low else ""))

    # ① 미매핑·끊김 장치 — 끊김 기준은 장치마다 max(stale_after_s, 기대 주기 × 2)
    #   (문 센서처럼 18시간마다 보고하는 장치를 10분 기준으로 판정하면 오판한다. 2026-10-21 실제 발생)
    stale_s = cfg.report["stale_after_s"]
    rows = []
    for dev, info in sorted(d.devices.items()):
        problems = []
        if info["mapped_by"] == "none" or info["space_id"] == "unmapped":
            problems.append(f"공간 미지정(원천 표기 '{info['raw_location']}')")
        last = info["last_seen"]
        last_utc = last.replace(tzinfo=UTC) if last is not None else None
        limit = max(stale_s, 2 * period(dev))
        age = (d.end - last_utc).total_seconds() if last_utc is not None else None
        if age is not None and age > limit:
            observed = (d.end - d.live_since).total_seconds() if d.live_since else 0
            near_load_end = (info.get("source") == "influx" and d.influx_end is not None
                             and abs((last_utc - d.influx_end).total_seconds()) < 2 * period(dev))
            if near_load_end:
                problems.append(f"판단 보류: 마지막 기록이 과거 적재 끝({kst(d.influx_end, '%m-%d')})과 겹침 — 이후 데이터 미적재")
            elif observed < limit and info.get("source") == "influx":
                problems.append(f"판단 보류: 실시간 관찰 {observed / 3600:.0f}시간 < 판정 기준 {limit / 3600:.0f}시간"
                                f"(마지막 기록 {age / 86400:.0f}일 전)")
            elif age >= 86400:
                problems.append(f"수신 중단: {age / 86400:.0f}일째 없음(기준 {limit / 3600:.1f}시간)")
            else:
                problems.append(f"끊김: {age / 60:.0f}분째 수신 없음(기준 {limit / 60:.0f}분)")
        bat = d.battery.get(dev)
        if bat is not None and bat <= cfg.report["battery_warn_pct"]:
            problems.append(f"배터리 {bat:.0f}% — 교체 필요")
        if problems:
            rows.append([dev, info["model"], _space_label(d, info["space_id"]), kst(last), "; ".join(problems)])
    parts += ["", "## ① 확인 필요 장치(미매핑·끊김)",
              _table(["장치", "기종", "공간", "마지막 수신", "문제"], rows) if rows else "없음"]

    # ② 공간·항목
    if by in (None, "space"):
        rows = []
        for (sid, item), st in sorted(d.series.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            cov = _pct(covs.get((sid, item), 0.0))
            rows.append([_space_label(d, sid), item, st["devices"], f"{st['count']:,}", kst(st["last"]),
                         cov if (sid, item) in judged else f"{cov}(조건부·판정 제외)"])
        parts += ["", "## ② 공간·항목별", _table(["공간", "항목", "장치 수", "건수", "마지막 수신", "커버율"], rows)]

    # ③ 장치별
    if by in (None, "device"):
        rows, mismatch = [], []
        for dev, info in sorted(d.devices.items(), key=lambda kv: (kv[1]["space_id"], kv[0])):
            per = period(dev)
            obs = observed_period_min(d.device_minutes.get(dev, set()))
            if not cfg.is_periodic(dev):
                rows.append([dev, info["model"], _space_label(d, info["space_id"]), "사건형", "—",
                             f"{d.device_counts.get(dev, 0):,}", kst(info["last_seen"]), "—"])
                continue
            if obs is not None and obs * 60 >= per * 2:
                mismatch.append(f"`{dev}: {{expected_period_s: {int(obs * 60)}}}`")
            rows.append([dev, info["model"], _space_label(d, info["space_id"]), per,
                         "—" if obs is None else f"{obs:g}분",
                         f"{d.device_counts.get(dev, 0):,}", kst(info["last_seen"]),
                         _pct(coverage(d.device_minutes.get(dev, set()), s_min, e_min, per))])
        parts += ["", "## ③ 장치별",
                  _table(["장치", "기종", "공간", "기대 주기(초)", "관측 주기", "건수", "마지막 수신", "커버율"], rows)]
        if mismatch:
            parts.append("\n관측 주기가 기대 주기의 2배 이상인 장치 — 실제 주기가 길다면 `config/collector.yaml`의 "
                         "`devices`에 추가: " + ", ".join(mismatch))

    # ④ 빠진 구간
    all_minutes = set().union(*d.device_minutes.values()) if d.device_minutes else set()
    global_gaps = gaps(all_minutes, s_min, e_min, 60, gap_min)
    global_set = {m for a, b in global_gaps for m in range(a, b)}
    rows = []
    for g in global_gaps:
        rows.append(["(전체)", "—", kst(_dt(g[0]), "%m-%d %H:%M"), kst(_dt(g[1]), "%m-%d %H:%M"),
                     f"{g[1] - g[0]}분", cause_hint(g, None, d.device_minutes, d.stats_minutes, d.events)])
    for dev, info in sorted(d.devices.items()):
        if not d.device_minutes.get(dev) or not cfg.is_periodic(dev):
            continue                                         # 측정값 없는 장치는 ①번 표, 사건형 센서는 결측 판정 안 함
        per = period(dev)
        mins = d.device_minutes.get(dev, set()) | global_set     # 전체 결측은 위에서 이미 보고
        for g in gaps(mins, s_min, e_min, per, gap_min):
            rows.append([dev, _space_label(d, info["space_id"]), kst(_dt(g[0]), "%m-%d %H:%M"),
                         kst(_dt(g[1]), "%m-%d %H:%M"), f"{g[1] - g[0]}분",
                         cause_hint(g, dev, d.device_minutes, d.stats_minutes, d.events, d.battery.get(dev))])
    parts += ["", f"## ④ 빠진 구간(자기 주기의 2배 이상, 최소 {gap_min}분)",
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

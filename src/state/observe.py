"""직접 관찰 기록 (spec US2, research R2·R3).

data/observations.csv에 상태 전환을 한 줄씩 남긴다. 직전 열린 구간은 자동으로 닫힌다. 덮어쓰지 않고 덧붙인다
(열린 구간을 닫을 때만 그 줄의 end를 채운다). 시각은 KST `YYYY-MM-DD HH:MM`.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

KST = timezone(timedelta(hours=9))
COLUMNS = ["space_id", "start", "end", "state", "people", "observer", "note"]
STATES = ("occupied", "empty")
FMT = "%Y-%m-%d %H:%M"


class ObservationError(ValueError):
    pass


@dataclass
class Problem:
    line: int          # CSV 줄 번호(머리글 = 1)
    reason: str


def now_kst() -> datetime:
    return datetime.now(KST).replace(second=0, microsecond=0, tzinfo=None)


def parse_time(s: str) -> datetime:
    try:
        return datetime.strptime(str(s).strip(), FMT)
    except ValueError as e:
        raise ObservationError(f"시각 형식은 'YYYY-MM-DD HH:MM'(KST): {s!r}") from e


def read(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[COLUMNS]


def write(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df[COLUMNS].to_csv(tmp, index=False, quoting=csv.QUOTE_MINIMAL)
    tmp.replace(path)


def validate(df: pd.DataFrame, spaces: list[str], live_start: str | None = None) -> list[Problem]:
    """겹침·모르는 공간·상태·시각 형식·끝 ≤ 시작·실시간 수집 이전을 줄 번호와 함께 찾는다."""
    probs, rows = [], []
    ls = datetime.strptime(live_start, "%Y-%m-%d") if live_start else None
    for i, r in enumerate(df.itertuples(index=False), start=2):
        if r.space_id not in spaces:
            probs.append(Problem(i, f"모르는 공간 '{r.space_id}' (가능: {', '.join(spaces)})"))
            continue
        if r.state not in STATES:
            probs.append(Problem(i, f"상태는 occupied·empty 중 하나: '{r.state}'"))
        try:
            a = parse_time(r.start)
            b = parse_time(r.end) if str(r.end).strip() else None
        except ObservationError as e:
            probs.append(Problem(i, str(e)))
            continue
        if b is not None and b <= a:
            probs.append(Problem(i, f"끝({r.end})이 시작({r.start})보다 늦어야 함"))
        if ls and a < ls:
            probs.append(Problem(i, f"실시간 수집 시작({live_start}) 이전 시각"))
        if str(r.people).strip() and not str(r.people).strip().isdigit():
            probs.append(Problem(i, f"인원은 0 이상의 정수: '{r.people}'"))
        rows.append((i, r.space_id, a, b))
    open_per_space = {}
    for i, s, a, b in rows:
        if b is None:
            if s in open_per_space:
                probs.append(Problem(i, f"{s}에 열린 구간이 이미 있음(줄 {open_per_space[s]})"))
            open_per_space[s] = i
    for s in {r[1] for r in rows}:
        iv = sorted((a, b or datetime.max, i) for i, sp, a, b in rows if sp == s)
        for (a1, b1, i1), (a2, b2, i2) in zip(iv, iv[1:]):
            if a2 < b1:
                probs.append(Problem(i2, f"{s} 구간이 줄 {i1}과 겹침"))
    return sorted(probs, key=lambda p: p.line)


def record(path: Path, space: str, state: str, spaces: list[str], at: datetime | None = None,
           people: int | None = None, observer: str = "", note: str = "") -> str:
    """상태 전환: 그 공간의 열린 구간을 at에 닫고 새 구간을 연다. state='stop'이면 닫기만."""
    if space not in spaces:
        raise ObservationError(f"모르는 공간 '{space}' (가능: {', '.join(spaces)})")
    if state not in STATES + ("stop",):
        raise ObservationError("상태는 occupied·empty·stop 중 하나")
    at = at or now_kst()
    df = read(path)
    open_idx = df.index[(df["space_id"] == space) & (df["end"].str.strip() == "")]
    msg = []
    if len(open_idx):
        i = open_idx[-1]
        if parse_time(df.at[i, "start"]) >= at:
            raise ObservationError(f"{space}의 열린 구간 시작({df.at[i, 'start']})보다 늦은 시각이어야 함")
        df.at[i, "end"] = at.strftime(FMT)
        msg.append(f"닫음: {space} {df.at[i, 'state']} {df.at[i, 'start']} ~ {df.at[i, 'end']}")
    elif state == "stop":
        raise ObservationError(f"{space}에 열린 구간이 없음")
    else:
        # 닫힌 구간과 겹치지 않는지(뒤늦은 --at 기록)
        same = df[df["space_id"] == space]
        for r in same.itertuples(index=False):
            if parse_time(r.start) <= at < parse_time(r.end):
                raise ObservationError(f"{space} 기존 구간 {r.start} ~ {r.end}과 겹침")
    if state != "stop":
        if observer and len(observer) > 4:
            raise ObservationError("관찰자는 머리글자(4자 이하)로 적는다(공개 저장소 밖이지만 개인정보 최소화)")
        new = {"space_id": space, "start": at.strftime(FMT), "end": "", "state": state,
               "people": "" if people is None else str(int(people)), "observer": observer, "note": note}
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        msg.append(f"시작: {space} {state} {new['start']}~" + (f" ({people}명)" if people is not None else ""))
    write(df, path)
    return " / ".join(msg)


def intervals(df: pd.DataFrame, close_open_at: datetime | None = None) -> pd.DataFrame:
    """닫힌 구간(열린 구간은 close_open_at로 닫아서 포함, 없으면 뺌). 시각은 KST naive."""
    out = []
    for r in df.itertuples(index=False):
        a = parse_time(r.start)
        b = parse_time(r.end) if str(r.end).strip() else close_open_at
        if b is None or b <= a:
            continue
        out.append({"space_id": r.space_id, "start": a, "end": b, "state": r.state,
                    "people": int(r.people) if str(r.people).strip().isdigit() else None, "note": r.note})
    return pd.DataFrame(out, columns=["space_id", "start", "end", "state", "people", "note"])


def _is_class_day(d, acfg) -> bool:
    import holidays
    from analysis.schedule import is_class_day
    return is_class_day(d, acfg, holidays.KR(years=d.year))


def status(df: pd.DataFrame, cfg, now: datetime | None = None) -> pd.DataFrame:
    """공간별 관찰 시간·상태 비율·수업일 수·경계 포함 횟수와 목표 대비 진행률."""
    o = cfg["observe"]
    tgt = o["target"]
    win = timedelta(minutes=o["boundary_window_min"])
    bounds = [datetime.strptime(b, "%H:%M").time() for b in o["boundaries"]]
    iv = intervals(df, now or now_kst())
    rows = []
    for space in cfg.spaces:
        g = iv[iv["space_id"] == space]
        hours = float(((g["end"] - g["start"]).dt.total_seconds() / 3600).sum()) if len(g) else 0.0
        occ = float(((g[g["state"] == "occupied"]["end"] - g[g["state"] == "occupied"]["start"]).dt.total_seconds() / 3600).sum()) if len(g) else 0.0
        days = {r.start.date() for r in g.itertuples() if _is_class_day(r.start.date(), cfg.acfg)}
        visits = 0
        for d in {r.start.date() for r in g.itertuples()} | {r.end.date() for r in g.itertuples()}:
            for b in bounds:
                c = datetime.combine(d, b)
                # 경계 ±30분을 관찰 구간들이 함께 덮으면 1회(경계에서 상태가 바뀌어 구간이 나뉘는 것이 정상)
                cov = sum(max(0.0, (min(r.end, c + win) - max(r.start, c - win)).total_seconds()) for r in g.itertuples())
                if cov >= 2 * win.total_seconds():
                    visits += 1
        row = {"공간": space, "관찰 시간": round(hours, 1), "사용중 비율": round(occ / hours, 2) if hours else None,
               "수업일 수": len(days), "경계 포함": visits,
               "진행률(수업일)": f"{min(len(days), tgt['class_days_per_space'])}/{tgt['class_days_per_space']}",
               "진행률(경계)": f"{min(visits, tgt['boundary_visits_per_space'])}/{tgt['boundary_visits_per_space']}"}
        if space == "meeting":
            meetings = int(((g["state"] == "occupied")).sum())
            empty_h = hours - occ
            row["회의실(회의·빈 시간)"] = (f"{min(meetings, tgt['meeting']['meetings'])}/{tgt['meeting']['meetings']}회, "
                                       f"{min(empty_h, tgt['meeting']['empty_hours']):.1f}/{tgt['meeting']['empty_hours']}시간")
        open_ = df[(df["space_id"] == space) & (df["end"].str.strip() == "")]
        row["열린 구간"] = f"{open_.iloc[-1]['state']} {open_.iloc[-1]['start']}~" if len(open_) else ""
        rows.append(row)
    t = pd.DataFrame(rows)
    total = t["관찰 시간"].sum()
    t.attrs["total"] = f"전체 {total:.1f}/{tgt['total_hours']}시간"
    t.attrs["done"] = bool(total >= tgt["total_hours"] and
                           all(int(x.split('/')[0]) >= int(x.split('/')[1]) for x in t["진행률(수업일)"]) and
                           all(int(x.split('/')[0]) >= int(x.split('/')[1]) for x in t["진행률(경계)"]))
    return t


def door_mismatch(df: pd.DataFrame, door_events: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """research R3: '비어있음' 구간 안에 문 열림이 있으면 기록 확인 대상. door_events: space_id, t(KST naive)."""
    iv = intervals(df, now or now_kst())
    iv = iv[iv["state"] == "empty"]
    out = []
    for r in iv.itertuples(index=False):
        e = door_events[(door_events["space_id"] == r.space_id) & (door_events["t"] >= r.start) & (door_events["t"] < r.end)]
        if len(e):
            out.append({"space_id": r.space_id, "start": r.start, "end": r.end, "문 열림 수": len(e),
                        "처음": e["t"].min()})
    return pd.DataFrame(out)

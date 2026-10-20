"""저장된 메시지(JSONL)를 설정으로 해석만 해 본다 — 형식 변경 대응을 저장소 없이 확인한다."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime

from .dedup import FrameDedup
from .parser import parse
from .timeutil import UTC, kst, parse_time


def replay(cfg, path: str, limit: int | None = None) -> str:
    rows, stats, failures = [], Counter(), Counter()
    dedup = FrameDedup()
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if not line.strip():
                continue
            m = json.loads(line)
            recv = datetime.fromtimestamp(m["t"], UTC) if "t" in m else datetime.now(UTC)
            payload = m["payload"] if isinstance(m["payload"], str) else json.dumps(m["payload"])
            r = parse(m["topic"], payload.encode("utf-8"), recv, cfg)
            stats["messages"] += 1
            if r.failed:
                stats["failed"] += 1
                failures[r.failed] += 1
                continue
            if r.ignored_topic:
                stats["ignored_messages"] += 1
                continue
            if dedup.is_duplicate(r.frame_key):
                stats["duplicate"] += 1
                continue
            stats["ignored_fields"] += r.ignored
            for x in r.measurements:
                stats["measurements"] += 1
                rows.append((x["space_id"], x["device_id"], x["item"], x["value"], x["unit"] or "",
                             kst(parse_time(x["ts"]), "%H:%M:%S"), "보정" if x["ts_corrected"] else ""))
    out = ["| 공간 | 장치 | 항목 | 값 | 단위 | 기준 시각(KST) | 비고 |", "|---|---|---|---|---|---|---|"]
    out += [f"| {' | '.join(str(c) for c in r)} |" for r in rows]
    unmapped = sorted({r[1] for r in rows if r[0] == "unmapped"})
    out.append("")
    out.append(f"메시지 {stats['messages']} · 측정값 {stats['measurements']} · 무시 메시지 "
               f"{stats['ignored_messages']} · 중복(재전송) {stats['duplicate']} · 무시 항목 "
               f"{stats['ignored_fields']} · 실패 {stats['failed']}"
               + (f" {dict(failures)}" if failures else ""))
    if unmapped:
        out.append(f"공간 미지정 장치: {', '.join(unmapped)}")
    return "\n".join(out)

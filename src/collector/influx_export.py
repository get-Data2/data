"""맥(내부망 와이파이)용 InfluxDB 내보내기 스크립트 만들기 (2026-10-21: 보드 유선망 단절 대응).

보드가 쓰던 것과 같은 Flux 문장(flux_infer·flux_chunk)으로 남은 구간을 하루치 CSV로 받는다.
스크립트에는 토큰을 넣지 않는다(맥에서 실행할 때 입력). 내부 주소가 들어가므로 var/(git 제외)에 쓴다.
"""
from __future__ import annotations

import json
from datetime import datetime

from .backfill import chunk_file, flux_chunk, flux_infer
from .timeutil import kst

HEAD = """#!/bin/bash
# InfluxDB 미적재 구간 내보내기 — 맥에서 내부망 와이파이에 연결한 뒤 실행: bash {name} [저장 폴더]
# 만든 시각 {made}, 구간 {n}개({first} ~ {last} KST). 이미 받은 파일은 건너뛴다(다시 실행해도 됨).
set -u
URL="{url}"
BUCKET="{bucket}"
OUT="${{1:-influx_export}}"
mkdir -p "$OUT"
if [ -z "${{INFLUX_TOKEN:-}}" ]; then read -r -s -p "InfluxDB 토큰: " INFLUX_TOKEN; echo; fi
code=$(curl -s -m 10 -o /dev/null -w "%{{http_code}}" "$URL/health")
if [ "$code" != "200" ]; then echo "InfluxDB에 닿지 않음(HTTP $code) — 내부망 와이파이 연결 확인"; exit 1; fi
ORG=$(curl -s -m 20 -H "Authorization: Token $INFLUX_TOKEN" "$URL/api/v2/buckets?name=$BUCKET" | tr ',' '\\n' | sed -n 's/.*"orgID": *"\\([^"]*\\)".*/\\1/p' | head -1)
if [ -z "$ORG" ]; then echo "버킷 '$BUCKET'을 찾지 못함 — 토큰 확인"; exit 1; fi
fail=0
q() {{   # $1 파일 이름, 표준 입력 = 조회 JSON
  if [ -s "$OUT/$1" ]; then echo "  있음 $1"; cat > /dev/null; return; fi
  c=$(curl -sS -m 900 -X POST "$URL/api/v2/query?orgID=$ORG" -H "Authorization: Token $INFLUX_TOKEN" \\
      -H "Content-Type: application/json" -H "Accept: application/csv" --data-binary @- -o "$OUT/$1.part" -w "%{{http_code}}")
  if [ "$c" = "200" ]; then mv "$OUT/$1.part" "$OUT/$1"; echo "  받음 $1 ($(wc -l < "$OUT/$1")줄)";
  else echo "  실패 $1 (HTTP $c)"; head -c 300 "$OUT/$1.part"; echo; rm -f "$OUT/$1.part"; fail=$((fail+1)); fi
}}
"""

TAIL = """
echo "끝: 실패 $fail개. 실패가 있으면 같은 명령을 다시 실행한다."
echo "다음: tar czf influx_export.tgz $OUT  →  보드로 옮겨 collector backfill --from-dir 로 적재"
"""


def make_script(cfg, url: str, bucket: str, chunks: list[tuple[datetime, datetime]], infer_stop: datetime,
                name: str = "influx_export.sh") -> str:
    parts = [HEAD.format(name=name, made=kst(datetime.now().astimezone(), "%Y-%m-%d %H:%M"), n=len(chunks),
                         first=kst(chunks[0][0], "%Y-%m-%d") if chunks else "-",
                         last=kst(chunks[-1][1], "%Y-%m-%d %H:%M") if chunks else "-", url=url, bucket=bucket)]
    items = [("infer.csv", flux_infer(bucket, cfg, infer_stop))] + [(chunk_file(a), flux_chunk(bucket, cfg, a, b))
                                                                   for a, b in chunks]
    for i, (fname, flux) in enumerate(items):
        body = json.dumps({"query": flux, "type": "flux"}, ensure_ascii=False)
        parts.append(f"q {fname} <<'Q{i}'\n{body}\nQ{i}\n")
    parts.append(TAIL)
    return "".join(parts)

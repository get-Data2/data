"""저장소 크기와 1년 추정, 보관 정책 적용 시 크기 (research R9, 심화 B)."""
from __future__ import annotations

from .report import _table

RAW_KEEP_DAYS = 90      # 원본 1분 값 보관 기간
DOWNSAMPLE_MIN = 5      # 그 뒤 남기는 집계 간격(분)


def capacity_report(conn, prefix: str = "") -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT table_name, table_rows, data_length, index_length FROM information_schema.TABLES "
                    "WHERE table_schema = DATABASE() AND table_name IN (%s, %s, %s, %s, %s)",
                    [prefix + t for t in ("measurement", "device", "space", "collector_event", "ingest_stats")])
        tables = cur.fetchall()
        cur.execute(f"SELECT COUNT(*), MIN(ts), MAX(ts), COUNT(DISTINCT device_id, item) FROM `{prefix}measurement`")
        n, first, last, series = cur.fetchone()

    rows, total = [], 0
    meas_bytes = 0
    for name, est_rows, data, idx in tables:
        size = (data or 0) + (idx or 0)
        total += size
        if name == prefix + "measurement":
            meas_bytes = size
        rows.append([name, f"{est_rows or 0:,}", f"{size / 2**20:.1f} MB"])
    out = ["# 저장소 크기·1년 추정", "", _table(["테이블", "행(추정)", "데이터+인덱스"], rows),
           "", f"- 합계 {total / 2**20:.1f} MB"]
    if not n or not first or last <= first:
        out.append("- 측정값이 부족해 추정 불가(하루 이상 쌓인 뒤 다시 실행)")
        return "\n".join(out)

    days = max((last - first).total_seconds() / 86400, 1 / 24)
    per_day = n / days
    bytes_per_row = meas_bytes / n if n else 0
    year_rows = per_day * 365
    year_raw = year_rows * bytes_per_row
    # 정책: 최근 90일 원본 + 이후 5분 집계(평균·최소·최대·건수) — 계열 수 × 하루 288칸
    agg_rows_per_day = series * (1440 / DOWNSAMPLE_MIN)
    policy = per_day * RAW_KEEP_DAYS * bytes_per_row + agg_rows_per_day * (365 - RAW_KEEP_DAYS) * bytes_per_row * 1.3
    out += [
        f"- 측정 기간 {days:.1f}일, 측정값 {n:,}행 → 하루 {per_day:,.0f}행, 행당 {bytes_per_row:.0f}바이트(인덱스 포함)",
        "",
        _table(["시나리오", "1년 행 수", "1년 크기"], [
            ["원본 1분 값 전부 보관", f"{year_rows:,.0f}", f"{year_raw / 2**30:.2f} GB"],
            [f"원본 {RAW_KEEP_DAYS}일 + 이후 {DOWNSAMPLE_MIN}분 집계",
             f"{per_day * RAW_KEEP_DAYS + agg_rows_per_day * (365 - RAW_KEEP_DAYS):,.0f}",
             f"{policy / 2**30:.2f} GB"],
        ]),
        "",
        f"- 계열(장치×항목) {series}개 기준. 장치·항목이 늘면 비례해서 는다.",
    ]
    return "\n".join(out)

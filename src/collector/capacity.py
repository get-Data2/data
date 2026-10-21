"""저장소 크기와 1년 추정, 보관 정책 적용 시 크기 (research R9, 심화 B).

속도는 **최근 24시간 실시간 수집**(source가 influx가 아닌 행)으로 잡는다. 과거 적재분 평균은 장기 정지·미적재 구간 때문에
현재 속도를 낮게 보이게 한다(2026-10-21 첫 추정에서 실제 발생).
"""
from __future__ import annotations

from .report import _table

RAW_KEEP_DAYS = 90      # 원본 값 보관 기간
DOWNSAMPLE_MIN = 5      # 그 뒤 남기는 집계 간격(분)
AGG_ROW_FACTOR = 1.3    # 집계 행은 평균·최소·최대·건수를 담아 원본 행보다 조금 크다


def capacity_report(conn, prefix: str = "") -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT table_name, table_rows, data_length, index_length FROM information_schema.TABLES "
                    "WHERE table_schema = DATABASE() AND table_name IN (%s, %s, %s, %s, %s)",
                    [prefix + t for t in ("measurement", "device", "space", "collector_event", "ingest_stats")])
        tables = cur.fetchall()
        cur.execute(f"SELECT COUNT(*) FROM `{prefix}measurement`")
        total_rows = cur.fetchone()[0]
        # 최근 24시간 실시간 수집: 계열(장치×항목)별 행 수
        cur.execute(f"SELECT device_id, item, COUNT(*) FROM `{prefix}measurement` "
                    f"WHERE source<>'influx' AND received_at > UTC_TIMESTAMP() - INTERVAL 1 DAY GROUP BY 1, 2")
        series = [n for _, _, n in cur.fetchall()]
        cur.execute(f"SELECT TIMESTAMPDIFF(SECOND, MIN(received_at), MAX(received_at)) FROM `{prefix}measurement` "
                    f"WHERE source<>'influx' AND received_at > UTC_TIMESTAMP() - INTERVAL 1 DAY")
        span_s = cur.fetchone()[0] or 0

    rows, total, meas_bytes = [], 0, 0
    for name, est_rows, data, idx in tables:
        size = (data or 0) + (idx or 0)
        total += size
        if name == prefix + "measurement":
            meas_bytes = size
        rows.append([name, f"{est_rows or 0:,}", f"{size / 2**20:.1f} MB"])
    out = ["# 저장소 크기·1년 추정", "", _table(["테이블", "행(추정)", "데이터+인덱스"], rows),
           "", f"- 합계 {total / 2**20:.1f} MB (측정값 {total_rows:,}행)"]
    if not series or span_s < 3600:
        out.append("- 실시간 수집이 1시간 미만이라 속도 추정 불가")
        return "\n".join(out)

    scale = 86400 / span_s                     # 24시간이 덜 찼으면 비례 환산
    per_day = sum(series) * scale
    bytes_per_row = meas_bytes / total_rows if total_rows else 0
    agg_per_day = sum(min(n * scale, 1440 / DOWNSAMPLE_MIN) for n in series)
    raw_year = per_day * 365
    policy_rows = per_day * RAW_KEEP_DAYS + agg_per_day * (365 - RAW_KEEP_DAYS)
    policy_bytes = (per_day * RAW_KEEP_DAYS + agg_per_day * (365 - RAW_KEEP_DAYS) * AGG_ROW_FACTOR) * bytes_per_row
    out += [
        f"- 속도 기준: 최근 실시간 수집 {span_s / 3600:.1f}시간, 활성 계열(장치×항목) {len(series)}개 → "
        f"하루 {per_day:,.0f}행. 행당 {bytes_per_row:.0f}바이트(인덱스 포함, 현재 테이블 실측)",
        "",
        _table(["시나리오", "1년 행 수", "1년 크기", "비고"], [
            ["원본 전부 보관", f"{raw_year:,.0f}", f"{raw_year * bytes_per_row / 2**30:.2f} GB", ""],
            [f"원본 {RAW_KEEP_DAYS}일 + 이후 {DOWNSAMPLE_MIN}분 집계", f"{policy_rows:,.0f}",
             f"{policy_bytes / 2**30:.2f} GB",
             f"집계는 하루 {agg_per_day:,.0f}행(1분 계열은 1/5, 10분 계열은 그대로)"],
        ]),
        "",
        f"- 지금 쌓인 과거 적재분 {meas_bytes / 2**30:.2f} GB는 별도다. 같은 정책을 적용하면 90일 이전분은 집계로 바뀐다.",
    ]
    return "\n".join(out)

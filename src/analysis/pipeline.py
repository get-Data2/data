"""단계 실행: clean → split → events, 그리고 info. 결과 파일은 var/ 아래(git 제외)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .extract import cache_dir, load_devices, load_manifest, load_raw
from .quality import apply_axis_a, quality_counts
from .series import sensor_bins, space_series

SERIES_DIR = Path("var/series")
EVAL_DIR = Path("var/eval")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{x:,}" if isinstance(x, (int,)) and not isinstance(x, bool) else str(x)
                                       for x in r) + " |")
    return "\n".join(lines)


def run_clean(cfg, split_name=None) -> str:
    """달마다: 원본 → 축 A → 품질 파일 + 칸 평균. 그다음 칸 평균만 합쳐 결측 처리(메모리 절약)."""
    from .series import aggregate_bins, finish_bins
    models = dict(load_devices(cfg)[["device_id", "model"]].dropna().values)
    qdir = cache_dir(cfg) / "quality"
    qdir.mkdir(parents=True, exist_ok=True)
    SERIES_DIR.mkdir(parents=True, exist_ok=True)
    interval = cfg["interval"]["minutes"]
    counts, aggs, n_raw = [], [], 0
    for f in sorted((cache_dir(cfg) / "raw").glob("*.parquet")):
        raw = pd.read_parquet(f)
        if raw.empty:
            continue
        q = apply_axis_a(raw, cfg, models)
        pd.concat([raw[["ts", "space_id", "device_id", "item", "value"]], q], axis=1).to_parquet(qdir / f.name, index=False)
        counts.append(quality_counts(raw, q))
        aggs.append(aggregate_bins(raw, q, interval))
        n_raw += len(raw)
        del raw, q
    counts = pd.concat(counts).groupby(["space_id", "item", "quality", "rule"], observed=True)["rows"].sum().reset_index()
    counts.to_parquet(cache_dir(cfg) / "quality_counts.parquet", index=False)
    tot = counts.groupby("quality", observed=True)["rows"].sum().to_dict()
    check = f"검산: ok {tot.get('ok', 0):,} + saturated {tot.get('saturated', 0):,} + error {tot.get('error', 0):,} " \
            f"= {sum(tot.values()):,} / 원본 {n_raw:,} → {'일치' if sum(tot.values()) == n_raw else '불일치!'}"
    rules = counts[counts["rule"] != ""].groupby(["item", "rule"], observed=True)["rows"].sum().reset_index()

    sb = finish_bins(pd.concat(aggs, ignore_index=True), cfg, interval)
    del aggs
    sb.to_parquet(SERIES_DIR / "sensor_5min.parquet", index=False)
    ss = space_series(sb, cfg)
    ss.to_parquet(SERIES_DIR / "space_5min.parquet", index=False)
    meta = {"config": cfg.digest, "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "raw_rows": n_raw, "sensor_bins": len(sb), "space_bins": len(ss),
            "manifest": {k: v["sha256"][:12] for k, v in load_manifest(cfg)["months"].items()}}
    (SERIES_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    filled = sb.groupby("item", observed=True)["filled"].mean().round(4).to_dict()
    return "\n".join([
        f"원본 {n_raw:,}행 → 센서 5분 칸 {len(sb):,} → 공간 5분 칸 {len(ss):,}",
        check,
        "규칙별 오류·포화 건수:", _table(rules) if len(rules) else "없음",
        f"센서 칸 중 채운 비율(보간·주기 내 직전값): {filled}",
    ])


def load_quality(cfg) -> pd.DataFrame:
    files = sorted((cache_dir(cfg) / "quality").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("품질 파일 없음 — 먼저 `analysis clean`")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_series(kind: str = "space") -> pd.DataFrame:
    p = SERIES_DIR / f"{kind}_5min.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} — 먼저 `analysis clean`")
    return pd.read_parquet(p)


def run_split(cfg, split_name=None) -> str:
    from .split import fit_train_stats, split_table
    sp = cfg.split(split_name)
    ss = load_series("space")
    sb = load_series("sensor")
    stats = fit_train_stats(ss, sb, sp, cfg)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / f"norm_{sp.name}.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1, default=str))
    return split_table(ss, sp, cfg) + f"\n\n학습 기간 통계 → {EVAL_DIR}/norm_{sp.name}.json (학습 기간 칸만 사용)"


def load_stats(cfg, split_name=None) -> dict:
    sp = cfg.split(split_name)
    p = EVAL_DIR / f"norm_{sp.name}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} — 먼저 `analysis split`")
    return json.loads(p.read_text())


def run_events(cfg, split_name=None) -> str:
    from .events import detect, events_markdown, stuck_candidates
    sb = load_series("sensor")
    ss = load_series("space")
    stats = load_stats(cfg, split_name)
    ev = detect(sb, ss, cfg, stats)
    st = stuck_candidates(cfg)
    ev.to_parquet(cache_dir(cfg) / "events.parquet", index=False)
    st.to_parquet(cache_dir(cfg) / "stuck.parquet", index=False)
    out = Path(cfg["report"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    md = events_markdown(ev, st, cfg, cfg.split(split_name))
    (out / "events.md").write_text(md, encoding="utf-8")
    summary = ev.groupby(["kind", "space_id"]).size().rename("건수").reset_index() if len(ev) else pd.DataFrame()
    return (_table(summary) if len(summary) else "사건 후보 없음") + f"\n값 고정 후보 {len(st)}건\n→ {out}/events.md"


def info(cfg, split_name=None) -> str:
    sp = cfg.split(split_name)
    man = load_manifest(cfg)
    lines = [f"설정 해시 {cfg.digest}, 분할 {sp.name}",
             f"사본: {len(man['months'])}개 달, {sum(v['rows'] for v in man['months'].values()):,}행"]
    for p in [SERIES_DIR / "space_5min.parquet", EVAL_DIR / f"norm_{sp.name}.json",
              EVAL_DIR / f"eval_points_{sp.name}.parquet", EVAL_DIR / f"baseline_{sp.name}.parquet"]:
        lines.append(f"- {p}: " + (f"있음, sha256 {sha256_file(p)[:16]}" if p.exists() else "없음"))
    ep_meta = EVAL_DIR / f"eval_points_{sp.name}.json"
    bl_meta = EVAL_DIR / f"baseline_{sp.name}.json"
    if ep_meta.exists() and bl_meta.exists():
        a = json.loads(ep_meta.read_text())["sha256"]
        b = json.loads(bl_meta.read_text())["eval_points_sha256"]
        lines.append(f"기준선이 쓴 평가 시점 = 현재 평가 시점 파일: {'일치' if a == b else '불일치!'} ({a[:16]})")
    return "\n".join(lines)

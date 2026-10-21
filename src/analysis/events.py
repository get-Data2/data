"""축 B — 실제 사건(공간·시간 패턴) 판정과 값 고정 후보(research R3, R11).

입력은 센서 5분 칸(축 A error가 이미 빠진 값 + saturated 표시)뿐이다. 축 A 설정을 보지 않는다(SC-008).
데이터로 정하는 임계값(p99.9, MAD)은 학습 기간 통계(norm_<split>.json)에서만 가져온다.
결과는 값을 바꾸지 않는 목록이다 — 과제 4(이상 탐지)·8(경보)로 넘긴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KIND_NAME = {"B0": "단독 극단값(의심)", "B1": "화재·과열", "B2": "누수·결로", "B3": "밀집·환기 부족"}


def _runs(mask: pd.Series, min_len: int = 1):
    """연속 True 구간 → [(시작 index, 끝 index)] (끝 포함)."""
    out, start = [], None
    idx = mask.index
    vals = mask.to_numpy()
    for i, v in enumerate(vals):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((idx[start], idx[i - 1]))
            start = None
    if start is not None and len(vals) - start >= min_len:
        out.append((idx[start], idx[-1]))
    return out


def _wide(sb: pd.DataFrame, space: str, item: str, interval: int) -> pd.DataFrame:
    d = sb[(sb["space_id"].astype(str) == space) & (sb["item"].astype(str) == item)]
    if d.empty:
        return pd.DataFrame()
    w = d.pivot_table(index="bin", columns="device_id", values="value", aggfunc="mean", observed=True)
    return w.reindex(pd.date_range(w.index.min(), w.index.max(), freq=f"{interval}min"))


def detect(sb: pd.DataFrame, ss: pd.DataFrame, cfg, stats: dict) -> pd.DataFrame:
    b = cfg["axis_b"]
    interval = cfg["interval"]["minutes"]
    rows = []

    def add(space, kind, s, e, devs, ev):
        rows.append({"space_id": space, "kind": kind, "start": s, "end": e + pd.Timedelta(minutes=interval),
                     "devices": ",".join(sorted(map(str, devs))), "evidence": ev, "rule_version": cfg.digest})

    spaces = sorted(sb["space_id"].astype(str).unique())
    for space in spaces:
        st = stats.get("spaces", {}).get(space, {})
        # ---- B3 밀집: CO₂ 여러 센서가 동시에 높음(센서 1대면 지속 요구)
        co2 = _wide(sb, space, "co2", interval)
        if not co2.empty:
            c = b["crowding"]
            thr = max(c["co2_floor"], st.get("co2", {}).get(f"p{c['train_quantile'] * 100:g}", 0))
            high = co2 > thr
            n_sensors = co2.notna().sum(axis=1)
            cnt = high.sum(axis=1)
            need = np.where(n_sensors >= c["min_sensors"], c["min_sensors"], 1)
            mask = cnt >= need
            min_len = np.where(n_sensors.max() >= c["min_sensors"], 1, c["single_sensor_min_bins"])
            for s, e in _runs(mask, int(min_len)):
                seg = co2.loc[s:e]
                add(space, "B3", s, e, seg.columns[(seg > thr).any()],
                    f"CO₂ 최고 {seg.max().max():.0f}ppm > 기준 {thr:.0f}(학습 p99.9·바닥값 중 큰 값)")
        # ---- B1 화재·과열: 온도 급상승 센서 2대 이상 + 습도 하락
        temp = _wide(sb, space, "temperature", interval)
        if not temp.empty and temp.shape[1] >= b["fire"]["min_sensors"]:
            f = b["fire"]
            k = max(1, f["window_min"] // interval)
            rise = temp - temp.shift(k)
            hot = (rise >= f["temp_rise_c"]).sum(axis=1) >= f["min_sensors"]
            if f.get("humidity_drop"):
                hum = _wide(sb, space, "humidity", interval).reindex(temp.index)
                drop = (hum - hum.shift(k)).median(axis=1) < 0
                hot &= drop.fillna(False)
            for s, e in _runs(hot):
                seg = temp.loc[s:e]
                after = temp.loc[e:].iloc[1:1 + k]
                lost = int(after.isna().all(axis=1).sum()) if len(after) else 0
                add(space, "B1", s, e, seg.columns[(rise.loc[s:e] >= f["temp_rise_c"]).any()],
                    f"{f['window_min']}분 상승 최대 {rise.loc[s:e].max().max():.1f}℃, 최고 {seg.max().max():.1f}℃"
                    + (f", 직후 공간 전체 무신호 {lost}칸" if lost else ""))
        # ---- B2 누수·결로: 습도 고정 고값 지속
        hum = _wide(sb, space, "humidity", interval)
        if not hum.empty:
            lk = b["leak"]
            min_bins = max(1, lk["min_duration_min"] // interval)
            for dev in hum.columns:
                for s, e in _runs(hum[dev] >= lk["humidity_min"], min_bins):
                    add(space, "B2", s, e, [dev], f"습도 {lk['humidity_min']}% 이상 {((e - s).total_seconds() / 60 + interval):.0f}분")
        # ---- B0 단독 극단값(의심): (센서 − 공간 중앙값)이 그 센서의 학습 기간 편향에서 자기 MAD × k 넘게 벗어남,
        #      그 칸에 그런 센서가 하나뿐. 센서마다 고정 편향(예: 실습실 AM107 −0.8℃)이 있어 공간 공용 기준은 과다 판정한다
        excl = set(cfg["space"]["exclude_from_space"])
        sens = stats.get("sensors", {})
        for item in ("co2", "temperature", "humidity"):
            w = _wide(sb, space, item, interval)
            if w.empty:
                continue
            w = w[[c for c in w.columns if str(c) not in excl]]
            if w.shape[1] < 3:
                continue
            med = w.median(axis=1)
            dev = w.sub(med, axis=0)
            z = pd.DataFrame(index=w.index)
            floor = (b["outlier"].get("min_scale") or {}).get(item, 0)
            for c in w.columns:
                ss_ = sens.get(str(c), {}).get(item)
                if ss_:
                    # 센서 3~4대에서는 중앙값이 그 센서 자신이 되는 일이 잦아 MAD가 분해능보다 작아진다(실측 0.099℃) → 바닥값
                    scale = max(ss_["mad"], floor)
                    if scale > 0:
                        z[c] = (dev[c] - ss_["bias"]).abs() / scale
            if z.empty:
                continue
            flag = z > b["outlier"]["mad_k"]
            single = flag.sum(axis=1) == 1
            for t in flag.index[single]:
                d = flag.columns[flag.loc[t]][0]
                add(space, "B0", t, t, [d], f"{item} 평소 편향에서 {abs(dev.loc[t, d] - sens[str(d)][item]['bias']):.1f} 벗어남"
                                            f"(자기 MAD×{z.loc[t, d]:.0f})")
    ev = pd.DataFrame(rows, columns=["space_id", "kind", "start", "end", "devices", "evidence", "rule_version"])
    return ev.sort_values(["start", "space_id", "kind"]).reset_index(drop=True)


def stuck_candidates(cfg) -> pd.DataFrame:
    """값 고정 후보: 같은 값이 min_hours 이상 이어지고, 같은 공간 다른 센서는 그동안 변한 경우(달 파일 단위로 판정)."""
    from .extract import cache_dir
    st = cfg["axis_b"]["stuck"]
    min_s = st["min_hours"] * 3600
    out = []
    for f in sorted((cache_dir(cfg) / "quality").glob("*.parquet")):
        q = pd.read_parquet(f, columns=["ts", "space_id", "device_id", "item", "value", "quality"])
        q = q[(q["quality"].astype(str) != "error") & q["item"].astype(str).isin(["co2", "temperature", "humidity"])]
        for (dev, item), g in q.groupby(["device_id", "item"], observed=True):
            g = g.sort_values("ts")
            change = g["value"].ne(g["value"].shift())
            grp = change.cumsum()
            spans = g.groupby(grp).agg(start=("ts", "first"), end=("ts", "last"), value=("value", "first"),
                                       n=("ts", "size"), space=("space_id", "first"))
            spans = spans[(spans["end"] - spans["start"]).dt.total_seconds() >= min_s]
            for r in spans.itertuples():
                if st.get("require_others_change"):
                    o = q[(q["space_id"] == r.space) & (q["item"] == item) & (q["device_id"] != dev)
                          & (q["ts"] >= r.start) & (q["ts"] <= r.end)]
                    need = (st.get("others_min_change") or {}).get(str(item), 0)
                    if o.empty or (o.groupby("device_id", observed=True)["value"].agg(lambda x: x.max() - x.min())).max() < need \
                            or o.groupby("device_id", observed=True)["value"].nunique().max() <= 1:
                        continue
                out.append({"space_id": str(r.space), "device_id": str(dev), "item": str(item), "start": r.start,
                            "end": r.end, "hours": round((r.end - r.start).total_seconds() / 3600, 1),
                            "value": r.value, "reports": int(r.n)})
    return pd.DataFrame(out, columns=["space_id", "device_id", "item", "start", "end", "hours", "value", "reports"])


def events_markdown(ev: pd.DataFrame, st: pd.DataFrame, cfg, sp) -> str:
    from .config import KST
    lines = [f"# 사건 후보·의심·값 고정 목록 (과제 2 축 B)",
             f"- 분할 {sp.name}(임계값은 학습 기간 통계), 설정 {cfg.digest}, 실데이터. 값은 바꾸지 않은 목록이다 — 과제 4·8로 넘김",
             "", "## 종류별 건수", "", "| 종류 | 공간 | 건수 |", "|---|---|---|"]
    if len(ev):
        for (k, s), n in ev.groupby(["kind", "space_id"]).size().items():
            lines.append(f"| {k} {KIND_NAME.get(k, '')} | {s} | {n} |")
    lines += ["", "## 실제 사건 후보(B1~B3, 최근 50건)", "", "| 종류 | 공간 | 시작(KST) | 끝 | 센서 | 근거 |", "|---|---|---|---|---|---|"]
    real = ev[ev["kind"] != "B0"].tail(50)
    for r in real.itertuples():
        lines.append(f"| {r.kind} | {r.space_id} | {r.start.tz_convert(KST):%Y-%m-%d %H:%M} | {r.end.tz_convert(KST):%H:%M} "
                     f"| {r.devices} | {r.evidence} |")
    lines += ["", f"## 값 고정 후보({len(st)}건, 긴 순 상위 30)", "", "| 공간 | 장치 | 항목 | 시작(KST) | 시간 | 값 |", "|---|---|---|---|---|---|"]
    for r in st.sort_values("hours", ascending=False).head(30).itertuples():
        lines.append(f"| {r.space_id} | {r.device_id} | {r.item} | {r.start.tz_convert(KST):%Y-%m-%d %H:%M} | {r.hours}h | {r.value} |")
    return "\n".join(lines) + "\n"

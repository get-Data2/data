"""시험 기간 사본에 고장 5종 주입 → 정답 표 (spec "고장 5종과 주입 규칙", research R2).

원본은 바꾸지 않는다. 주입은 관측 칸에만 하며(결측 모양 유지), 고장 구간 밖의 값은 원본과 같다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.config import KST

from .config import KINDS
from .data import split_col

H = pd.Timedelta(hours=1)


def _hm(s: str) -> int:
    return int(s[:2]) * 60 + int(s[3:])


def _class_day(ts: pd.Timestamp, acfg) -> bool:
    import holidays
    from analysis.schedule import is_class_day
    d = ts.tz_convert(KST).date()
    return is_class_day(d, acfg, holidays.KR(years=d.year))


def _free(busy: list, space: str, a, b, gap) -> bool:
    return all(not (a < e + gap and s < b + gap) for sp, s, e in busy if sp == space)


def inject(boards: dict[str, pd.DataFrame], cfg, seed: int | None = None) -> tuple[dict, pd.DataFrame, dict]:
    """boards: 공간 → 시험 기간 넓은 표(관측만). 돌려줌: 주입된 표, 정답 표, 공간 → fault_id 표."""
    ic = cfg["inject"]
    seed = ic["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    out = {s: w.copy() for s, w in boards.items()}
    fid = {s: pd.DataFrame(0, index=w.index, columns=w.columns, dtype=np.int32) for s, w in boards.items()}
    gap = pd.Timedelta(hours=ic["min_gap_h"])
    slack = pd.Timedelta(minutes=cfg["alarms"]["match_slack_min"])
    u0, u1 = _hm(ic["usage"]["start"]), _hm(ic["usage"]["end"])
    busy, faults = [], []
    spaces = list(boards)
    n = 0
    for kind in KINDS:
        for k in range(ic["per_kind"]):
            for _attempt in range(200):
                eligible = [s for s in spaces if _eligible(boards[s], kind)]
                space = eligible[(k + _attempt) % len(eligible)]
                w = out[space]
                t = w.index
                f = _draw(kind, w, t, rng, ic, u0, u1, cfg)
                if f is None or not _free(busy, space, f["start"], f["end"], gap):
                    continue
                n += 1
                f.update(fault_id=n, kind=kind, space_id=space, seed=seed)
                f["match_end"] = (f["start"] + pd.Timedelta(hours=cfg["alarms"]["F3_match_h"]) if kind == "F3"
                                  else f["end"] + slack)
                _apply(f, w, fid[space], ic)
                busy.append((space, f["start"], f["end"]))
                faults.append(f)
                break
            else:
                raise RuntimeError(f"{kind} {k + 1}번째 고장을 넣을 시각을 찾지 못함(시험 기간이 짧거나 간격 설정이 큼)")
    cols = ["fault_id", "kind", "space_id", "device_id", "items", "start", "end", "match_end", "magnitude", "seed"]
    return out, pd.DataFrame(faults)[cols], fid


def _eligible(w: pd.DataFrame, kind: str) -> bool:
    items = {split_col(c)[1] for c in w.columns}
    return {"F1": True, "F2": True, "F3": "co2" in items, "F4": "temperature" in items, "F5": "co2" in items}[kind]


def _draw(kind, w, t, rng, ic, u0, u1, cfg):
    cols = list(w.columns)
    p = ic[kind]
    if kind in ("F1", "F2", "F3"):
        cand = [c for c in cols if kind != "F3" or split_col(c)[1] == "co2"]
        col = cand[rng.integers(len(cand))]
        dev, item = split_col(col)
    if kind == "F1":
        obs = np.where(w[col].notna().to_numpy())[0]
        obs = obs[(obs > 2) & (obs < len(t) - 3)]
        i = obs[rng.integers(len(obs))]
        lo, hi = p[item]
        mag = float(rng.uniform(lo, hi)) * (1 if rng.random() < 0.5 else -1)
        v = w[col].iloc[i]
        if item == "co2" and v + mag < 300:
            mag = abs(mag)
        if item == "humidity" and not (0 <= v + mag <= 100):
            mag = -mag
        return {"device_id": dev, "items": item, "start": t[i], "end": t[i] + pd.Timedelta(minutes=5), "magnitude": mag,
                "_col": col}
    if kind in ("F2", "F3"):
        hours = float(rng.uniform(*p["hours"]))
        total = hours + (p["persist_h"] if kind == "F3" else 0)
        i = int(rng.integers(len(t)))
        a, b = t[i], t[i] + pd.Timedelta(hours=total)
        if b > t[-1]:
            return None
        seg = w.loc[a:b, col]
        if seg.notna().mean() < 0.6 or pd.isna(w[col].iloc[i]):
            return None
        if kind == "F2":
            dcols = [c for c in cols if split_col(c)[0] == dev]
            return {"device_id": dev, "items": ",".join(split_col(c)[1] for c in dcols), "start": a, "end": b,
                    "magnitude": hours, "_cols": dcols}
        return {"device_id": dev, "items": "co2", "start": a, "end": b, "magnitude": float(rng.uniform(*p["ppm"])),
                "_col": col, "_ramp_h": hours}
    # F4·F5: 사용 시간대 수업일 시작, 공간 전체 해당 항목
    item = "temperature" if kind == "F4" else "co2"
    icols = [c for c in cols if split_col(c)[1] == item]
    k = t.tz_convert(KST)
    mins = k.hour * 60 + k.minute
    ok = np.where((mins >= u0) & (mins < u1 - 60))[0]
    i = ok[rng.integers(len(ok))]
    a = t[i]
    if not _class_day(a, cfg.acfg):
        return None
    hours = float(rng.uniform(*p["hours"]))
    b = a + pd.Timedelta(hours=hours)
    if b > t[-1] or w.loc[a:b, icols].notna().mean().mean() < 0.4:
        return None
    mag = float(rng.uniform(*p["celsius"])) if kind == "F4" else float(p["factor"])
    f = {"device_id": None, "items": item, "start": a, "end": b, "magnitude": mag, "_cols": icols}
    if kind == "F5":
        day = a.tz_convert(KST).normalize()
        bs, be = day + pd.Timedelta(minutes=_hm(p["baseline"]["start"])), day + pd.Timedelta(minutes=_hm(p["baseline"]["end"]))
        base = w.loc[bs.tz_convert("UTC"):be.tz_convert("UTC"), icols].mean()
        if base.isna().all():
            return None
        f["_base"] = base.fillna(base.mean())
    return f


def _apply(f, w, fid, ic):
    kind, a, b = f["kind"], f["start"], f["end"]
    if kind == "F1":
        w.loc[a, f["_col"]] += f["magnitude"]
        fid.loc[a, f["_col"]] = f["fault_id"]
        return
    sl = (w.index >= a) & (w.index < b)
    if kind == "F2":
        for c in f["_cols"]:
            first = w.loc[:a, c].dropna()
            if first.empty:
                continue
            m = sl & w[c].notna().to_numpy()
            w.loc[m, c] = first.iloc[-1]
            fid.loc[m, c] = f["fault_id"]
        return
    if kind == "F3":
        c = f["_col"]
        el = (w.index[sl] - a) / pd.Timedelta(hours=f["_ramp_h"])
        off = f["magnitude"] * np.minimum(1.0, np.asarray(el, float))
        m = w.loc[sl, c].notna().to_numpy()
        w.loc[w.index[sl][m], c] += off[m]
        fid.loc[w.index[sl][m], c] = f["fault_id"]
        return
    el = np.asarray((w.index[sl] - a) / (b - a), float)
    for c in f["_cols"]:
        m = w.loc[sl, c].notna().to_numpy()
        idx = w.index[sl][m]
        if kind == "F4":
            w.loc[idx, c] += f["magnitude"] * el[m]
        else:
            base = f["_base"][c]
            v = w.loc[idx, c]
            w.loc[idx, c] = np.where(v > base, base + (v - base) * f["magnitude"], v)
        fid.loc[idx, c] = f["fault_id"]

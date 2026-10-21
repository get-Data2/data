"""데이터 진단(FR-008~012): 상세 표 Markdown(report.out_dir/diagnosis_<split>.md)과 그래프(report.deliverable_dir/figures/).

모든 통계는 지정한 분할의 **학습 기간**에서 계산한다(시험 기간 정보 누수 방지). 그래프는 PNG로 같은 폴더에 둔다.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .baseline import SPACE_NAME, observed_grid  # noqa: E402
from .config import KST  # noqa: E402
from .extract import cache_dir  # noqa: E402
from .gaps import interpolation_test  # noqa: E402
from .pipeline import _table, load_series  # noqa: E402
from .schedule import events_for, is_class_day, mismatch_days, reaction_lags  # noqa: E402
from .series import aggregate_bins  # noqa: E402
from .split import autocorr  # noqa: E402


def _setup_font(cfg):
    plt.rcParams["font.family"] = cfg["report"]["font"]
    plt.rcParams["axes.unicode_minus"] = False


def _train(df, sp, col="bin"):
    a, b = sp.utc("train")
    return df[(df[col] >= a) & (df[col] < b)]


def quality_table(cfg) -> pd.DataFrame:
    c = pd.read_parquet(cache_dir(cfg) / "quality_counts.parquet")
    c = c[c["item"].isin(["co2", "temperature", "humidity"]) & (c["space_id"] != "unmapped")]
    p = c.pivot_table(index=["space_id", "item"], columns="quality", values="rows", aggfunc="sum", fill_value=0,
                      observed=True).reset_index()
    for q in ("ok", "saturated", "error"):
        if q not in p.columns:
            p[q] = 0
    p["원본"] = p[["ok", "saturated", "error"]].sum(axis=1)
    p["오류율"] = (p["error"] / p["원본"] * 100).round(3).astype(str) + "%"
    return p[["space_id", "item", "원본", "ok", "saturated", "error", "오류율"]].rename(
        columns={"space_id": "공간", "item": "항목", "saturated": "포화", "error": "오류(제외)"})


def missing_table(ss, sp, cfg) -> pd.DataFrame:
    a, b = sp.utc("train")
    expected = int((b - a).total_seconds() // 60 // cfg["interval"]["minutes"])
    t = _train(ss, sp)
    rows = []
    for (space, item), g in t.groupby(["space_id", "item"], observed=True):
        obs = int((~g["filled"]).sum())
        fil = int(g["filled"].sum())
        rows.append({"공간": space, "항목": item, "관측 칸": f"{obs / expected * 100:.1f}%",
                     "채운 칸(보간·주기 내 직전값)": f"{fil / expected * 100:.1f}%",
                     "긴 결측(빈 칸)": f"{(expected - obs - fil) / expected * 100:.1f}%"})
    return pd.DataFrame(rows)


def device_periods(sb, cfg, long_h: float = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """장치별 관측 칸 간격과 긴 결측(long_h시간 이상 관측 없음) 목록."""
    interval = cfg["interval"]["minutes"]
    rows, gaps = [], []
    obs = sb[~sb["filled"]]
    for dev, g in obs.groupby("device_id", observed=True):
        t = g["bin"].drop_duplicates().sort_values()
        d = t.diff().dt.total_seconds().div(60).dropna()
        rows.append({"장치": dev, "공간": g["space_id"].iloc[0], "설정 주기(초)": cfg.period_s(str(dev)),
                     "관측 칸 간격 중앙값(분)": float(d.median()) if len(d) else np.nan,
                     "p99(분)": float(d.quantile(0.99)) if len(d) else np.nan})
        big = d[d >= long_h * 60]
        for i, m in big.items():
            end = t.loc[i]
            gaps.append({"장치": dev, "공간": g["space_id"].iloc[0], "시작(KST)": (end - pd.Timedelta(minutes=m)).tz_convert(KST).strftime("%Y-%m-%d %H:%M"),
                         "길이(일)": round(m / 1440, 1)})
    return pd.DataFrame(rows), pd.DataFrame(gaps).sort_values("길이(일)", ascending=False) if gaps else pd.DataFrame()


def interval_compare(cfg, month: str = "2025-10") -> pd.DataFrame:
    """실습실 CO₂로 분석 간격 후보 비교: 칸당 값 수, 빈 칸 비율, 칸 안 잡음, 이웃 칸 차이."""
    f = cache_dir(cfg) / "raw" / f"{month}.parquet"
    raw = pd.read_parquet(f)
    raw = raw[(raw["item"] == "co2") & (raw["space_id"] == "lab")]
    rows = []
    for iv in cfg["interval"]["candidates"]:
        raw = raw.assign(bin=raw["ts"].dt.floor(f"{iv}min"))
        g = raw.groupby(["device_id", "bin"], observed=True)["value"]
        per = g.size()
        noise = g.std().median()
        full_bins = int((raw["bin"].max() - raw["bin"].min()).total_seconds() // 60 // iv) + 1
        med = raw.groupby("bin")["value"].median()
        rows.append({"간격(분)": iv, "센서 칸당 값 수(중앙)": float(per.median()),
                     "센서 빈 칸 비율": f"{(1 - per.groupby(level=0).size().mean() / full_bins) * 100:.0f}%",
                     "칸 안 잡음(표준편차 중앙, ppm)": round(float(noise), 1) if pd.notna(noise) else "—",
                     "이웃 칸 차이 중앙(ppm)": round(float(med.diff().abs().median()), 1)})
    return pd.DataFrame(rows)


def hourly_pattern(ss, sp, cfg, space, item):
    t = _train(ss[~ss["filled"]], sp)
    t = t[(t["space_id"].astype(str) == space) & (t["item"].astype(str) == item)]
    k = t["bin"].dt.tz_convert(KST)
    import holidays
    kr = holidays.KR(years=sorted(set(k.dt.year)))
    cls = np.array([is_class_day(d, cfg, kr) for d in k.dt.date])
    hod = k.dt.hour + k.dt.minute / 60
    wk = t[cls].groupby(hod[cls])["value"].mean()
    we = t[~cls].groupby(hod[~cls])["value"].mean()
    return wk, we


def plot_patterns(ss, sp, cfg, out: Path) -> list[str]:
    files = []
    ev_times = [9, 11.5, 13, 18]
    for item, unit in (("co2", "ppm"), ("temperature", "℃")):
        fig, axes = plt.subplots(1, len(cfg["space"]["targets"]), figsize=(13, 3.6), sharey=True)
        for ax, space in zip(axes, cfg["space"]["targets"]):
            wk, we = hourly_pattern(ss, sp, cfg, space, item)
            ax.plot(wk.index, wk.values, label="수업일(평일·공휴일 제외)", lw=1.6)
            ax.plot(we.index, we.values, label="주말·공휴일", lw=1.2, ls="--")
            for x in ev_times:
                ax.axvline(x, color="gray", lw=0.6, alpha=0.6)
            ax.set_title(f"{SPACE_NAME.get(space, space)}")
            ax.set_xticks(range(0, 25, 3))
            ax.set_xlabel("시각(KST)")
        axes[0].set_ylabel(f"{item} ({unit})")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"학습 기간 시간대별 평균 {item} — 세로선: 09:00 수업·11:30 점심·13:00 오후·18:00 퇴실", fontsize=10)
        fig.tight_layout()
        name = f"pattern_{item}_{sp.name}.png"
        fig.savefig(out / name, dpi=110)
        plt.close(fig)
        files.append(name)
    return files


def reaction_analysis(ss, sp, cfg):
    a, b = sp.train
    res, mm = [], []
    for space in sorted({r["space"] for r in cfg["schedule"]["rows"]}):
        s = observed_grid(ss, space, "co2", cfg["interval"]["minutes"])
        if s.empty:
            continue
        ev = events_for(a, b, cfg)
        ev = ev[ev["space_id"] == space]
        lags = reaction_lags(s, ev, cfg)
        lags["space_id"] = space
        res.append(lags)
        m = mismatch_days(s, lags, a, b, cfg)
        m["space_id"] = space
        mm.append(m)
    return pd.concat(res, ignore_index=True), pd.concat(mm, ignore_index=True)


def reaction_table(lags: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (space, name), g in lags[lags["measured"]].groupby(["space_id", "name"], sort=False):
        lag = g["lag_min"]
        rows.append({"공간": space, "사건": name, "측정한 날": len(g), "반응한 비율": f"{lag.notna().mean() * 100:.0f}%",
                     "지연 중앙(분)": lag.median(), "p25~p75(분)": f"{lag.quantile(0.25):.0f}~{lag.quantile(0.75):.0f}"})
    return pd.DataFrame(rows)


def plot_reaction(lags, sp, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for (space, name), g in lags[lags["measured"]].groupby(["space_id", "name"], sort=False):
        v = g["lag_min"].dropna()
        if len(v) > 5:
            ax.hist(v, bins=range(0, 65, 5), alpha=0.45, label=f"{SPACE_NAME.get(space, space)} {name}")
    ax.set_xlabel("일정 사건 뒤 CO2가 ±30ppm 변할 때까지(분)")
    ax.set_ylabel("날 수")
    ax.legend(fontsize=7)
    ax.set_title("일정–CO2 반응 지연 분포(학습 기간)")
    fig.tight_layout()
    name = f"reaction_{sp.name}.png"
    fig.savefig(out / name, dpi=110)
    plt.close(fig)
    return name


def autocorr_table(ss, sp, cfg) -> pd.DataFrame:
    rows = []
    for space in cfg["space"]["targets"]:
        for item in ("co2", "temperature"):
            s = observed_grid(_train(ss, sp), space, item, cfg["interval"]["minutes"])
            if s.empty:
                continue
            r = autocorr(s, (1, 12, 288))
            rows.append({"공간": space, "항목": item, "5분": round(r[1], 3), "1시간": round(r[12], 3), "1일": round(r[288], 3)})
    return pd.DataFrame(rows)


def interp_table(ss, sp, cfg) -> pd.DataFrame:
    out = []
    for space, item in (("lab", "co2"), ("office", "co2"), ("lab", "temperature")):
        s = observed_grid(_train(ss, sp), space, item, cfg["interval"]["minutes"])
        t = interpolation_test(s.to_numpy(), cfg["interval"]["minutes"], cfg["gaps"]["interpolation_test_min"])
        t.insert(0, "계열", f"{space} {item}")
        out.append(t)
    d = pd.concat(out, ignore_index=True)
    d["mae"] = d["mae"].round(2)
    d["p90"] = d["p90"].round(2)
    d["noise"] = d["noise"].round(2)
    return d.rename(columns={"gap_min": "인위 결측(분)", "trials": "시도", "mae": "보간 MAE", "p90": "MAE p90",
                             "noise": "이웃 칸 차이 중앙(기준)"})


def run_diagnose(cfg, split_name=None) -> str:
    sp = cfg.split(split_name)
    _setup_font(cfg)
    out = Path(cfg["report"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(cfg["report"]["deliverable_dir"]) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_rel = Path(os.path.relpath(fig_dir, out))
    ss = load_series("space")
    sb = load_series("sensor")
    qt = quality_table(cfg)
    mt = missing_table(ss, sp, cfg)
    per, long_gaps = device_periods(sb, cfg)
    ic = interval_compare(cfg)
    it = interp_table(ss, sp, cfg)
    ac = autocorr_table(ss, sp, cfg)
    lags, mm = reaction_analysis(ss, sp, cfg)
    rt = reaction_table(lags)
    figs = plot_patterns(ss, sp, cfg, fig_dir) + [plot_reaction(lags, sp, fig_dir)]
    ev_md = (out / "events.md").exists()
    ev = pd.read_parquet(cache_dir(cfg) / "events.parquet") if (cache_dir(cfg) / "events.parquet").exists() else pd.DataFrame()

    # 숫자에서 바로 나오는 사실 요약(해석은 산출물 문서에서 사람이 쓴다)
    facts = []
    r9 = rt[(rt["공간"] == "lab") & (rt["사건"] == "수업 시작")]
    if len(r9):
        facts.append(f"실습실 09:00 수업 시작 뒤 CO₂가 +30ppm 오르기까지 중앙 {r9['지연 중앙(분)'].iloc[0]:.0f}분, 반응한 날 {r9['반응한 비율'].iloc[0]}")
    if len(mm):
        facts.append("어긋나는 날 " + ", ".join(f"{k} {v}일" for k, v in mm.groupby(["space_id", "type"]).size().items()))
    if len(ac):
        facts.append(f"자기상관(실습실 CO₂): 5분 {ac.iloc[0]['5분']}, 1시간 {ac.iloc[0]['1시간']}, 1일 {ac.iloc[0]['1일']} → 인접 칸이 거의 같은 값이라 무작위로 섞으면 시험 칸의 이웃이 학습에 들어가 성능이 부풀려진다")
    if len(ev):
        b3 = ev[ev["kind"] == "B3"]
        facts.append(f"밀집 사건 후보 {len(b3)}건 중 회의실 {int((b3['space_id'] == 'meeting').sum())}건 — 회의 중 CO₂ 최고 "
                     f"{b3[b3['space_id'] == 'meeting']['evidence'].str.extract(r'최고 (\d+)')[0].astype(float).max():.0f}ppm")

    a, b = sp.train
    md = [f"# 데이터 진단 상세 (과제 2, 분할 {sp.name})", "",
          f"- 측정 조건: 실데이터, 학습 기간 {a} ~ {b}(KST) 기준 통계, 분석 간격 {cfg['interval']['minutes']}분, 설정 {cfg.digest}",
          "- 다시 만들기: `analysis diagnose` (해석은 산출물 `docs/deliverables/task2/01-data-diagnosis.md`)", "",
          "## 사실 요약(자동)", ""] + [f"- {f}" for f in facts] + [
          "", "## 1. 정제 결과(축 A, 전체 기간)", "", _table(qt), "",
          "## 2. 결측(학습 기간, 공간 대표값)", "", _table(mt), "",
          "## 3. 장치별 기록 주기(관측 칸 기준)", "", _table(per.round(1)), "",
          "### 긴 결측(6시간 이상, 전체 기간)", "", _table(long_gaps.head(25)) if len(long_gaps) else "없음", "",
          "## 4. 분석 간격 후보 비교(실습실 CO₂, 2025-10)", "", _table(ic), "",
          "## 5. 짧은 결측 보간 시험(학습 기간, 인위 결측)", "", _table(it), "",
          "## 6. 자기상관(학습 기간, 관측 칸)", "", _table(ac), "",
          "## 7. 평일 시간대 패턴", "", *(f"![{f}]({fig_rel / f})" for f in figs[:2]), "",
          "## 8. 일정–CO₂ 반응 지연(고정 일과표)", "", _table(rt), "", f"![{figs[2]}]({fig_rel / figs[2]})", "",
          "### 어긋나는 날(학습 기간)", "", _table(mm.head(40)) if len(mm) else "없음", "",
          "## 9. 사건 후보(축 B)", "", "[events.md](events.md)" if ev_md else "없음", ""]
    p = out / f"diagnosis_{sp.name}.md"
    p.write_text("\n".join(md), encoding="utf-8")
    return "\n".join(["사실 요약:"] + [f"- {f}" for f in facts] + [f"→ {p}, 그래프 {len(figs)}개 → {fig_dir}"])

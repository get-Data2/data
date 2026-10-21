"""시험 예측 분석: 일과 단계별·상위 오차·창문 개방 추정·사건 구간, 일정 정보의 가치, 그래프 (research R12, FR-032·033)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.baseline import SPACE_NAME
from analysis.config import KST

from .features import offday_mask

PHASES = ["야간(19~08시)", "출근·수업 시작(08~09:30)", "오전 일과(09:30~11:30)", "점심(11:30~13:00)",
          "오후 일과(13:00~18:00)", "퇴실(18~19시)", "주말·공휴일"]


def phase_of(t: pd.Series, acfg) -> pd.Series:
    """정답 시각(t0+h)의 일과 단계."""
    ti = pd.DatetimeIndex(t)
    k = ti.tz_convert(KST)
    m = np.asarray(k.hour * 60 + k.minute)
    lab = np.select([m < 480, m < 570, m < 690, m < 780, m < 1080, m < 1140], PHASES[:6], PHASES[0])
    lab = np.where(offday_mask(ti, acfg), PHASES[6], lab)
    return pd.Series(pd.Categorical(lab, PHASES), index=t.index)


def seed_mean(df: pd.DataFrame) -> pd.DataFrame:
    """씨앗 평균 예측 한 줄씩(상위 오차·궤적용)."""
    k = ["space_id", "item", "t0", "horizon_min"]
    return df.groupby(k, as_index=False).agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
                                             usage_hours=("usage_hours", "first"))


def with_baseline(df: pd.DataFrame, bl: pd.DataFrame) -> pd.DataFrame:
    k = ["space_id", "item", "t0", "horizon_min"]
    return df.merge(bl[k + ["persistence"]], on=k, how="left")


def phase_errors(df, bl, acfg, item="co2", h=60) -> pd.DataFrame:
    d = with_baseline(seed_mean(df[(df["item"] == item) & (df["horizon_min"] == h)]), bl)
    d["phase"] = phase_of(d["t0"] + pd.Timedelta(minutes=h), acfg)
    g = d.assign(em=(d["y_pred"] - d["y_true"]).abs(), ep=(d["persistence"] - d["y_true"]).abs())
    t = g.groupby(["space_id", "phase"], observed=True).agg(n=("em", "size"), model=("em", "mean"),
                                                             persistence=("ep", "mean")).reset_index()
    t["개선율(%)"] = (1 - t["model"] / t["persistence"]) * 100
    return t


def top_errors(df, bl, acfg, n=20, item="co2", h=60) -> pd.DataFrame:
    d = with_baseline(seed_mean(df[(df["item"] == item) & (df["horizon_min"] == h)]), bl)
    d["err"] = d["y_pred"] - d["y_true"]
    d = d.reindex(d["err"].abs().sort_values(ascending=False).index).head(n)
    d["정답 시각(KST)"] = (d["t0"] + pd.Timedelta(minutes=h)).dt.tz_convert(KST).dt.strftime("%m-%d(%a) %H:%M")
    d["단계"] = phase_of(d["t0"] + pd.Timedelta(minutes=h), acfg).astype(str)
    return d


def window_open_segments(arrays, rule: dict, acfg) -> pd.DataFrame:
    """창문 개방 추정: 사용 시간대(평일 08~19시)에 CO₂가 within_min 안에 drop_ppm 이상 떨어지고, 일과표 경계(±15분)가 아님."""
    out = []
    w = rule["within_min"] // 5
    k = arrays.t.tz_convert(KST)
    m = np.asarray(k.hour * 60 + k.minute)
    off = offday_mask(arrays.t, acfg)
    near_bound = np.zeros(len(m), bool)
    for b in (540, 690, 780, 1080):
        near_bound |= np.abs(m - b) <= 15
    use = (~off) & (m >= 480) & (m < 1140) & ~near_bound
    for si, s in enumerate(arrays.spaces):
        y = arrays.y[si, :, 0]
        drop = y[:-w] - y[w:]
        idx = np.where((drop >= rule["drop_ppm"]) & use[:-w])[0]
        last = -10**9
        for i in idx:
            if i - last > 6:          # 30분 안의 연속 검출은 한 번으로
                out.append({"space_id": s, "t": arrays.t[i], "drop_ppm": float(drop[i])})
            last = i
    return pd.DataFrame(out)


def rows_near(df, segs: pd.DataFrame, before=30, after=30, col="t") -> pd.Series:
    """정답 시각이 구간 [t−before, t+after] 안인 행."""
    if segs.empty:
        return pd.Series(False, index=df.index)
    tgt = df["t0"] + pd.to_timedelta(df["horizon_min"], unit="min")
    mask = pd.Series(False, index=df.index)
    for s, g in segs.groupby("space_id"):
        sel = df["space_id"] == s
        if not sel.any():
            continue
        tt = tgt[sel].to_numpy()
        a = (pd.DatetimeIndex(g[col]) - pd.Timedelta(minutes=before)).as_unit("ns").asi8
        b = (pd.DatetimeIndex(g[col]) + pd.Timedelta(minutes=after)).as_unit("ns").asi8
        tv = pd.DatetimeIndex(tt).as_unit("ns").asi8
        i = np.searchsorted(a, tv, side="right") - 1
        hit = (i >= 0) & (tv <= b[np.clip(i, 0, None)])
        mask[sel] = hit
    return mask


def subset_mae(df, bl, mask, item="co2", h=60) -> pd.DataFrame:
    d = with_baseline(seed_mean(df[mask & (df["item"] == item) & (df["horizon_min"] == h)]), bl)
    g = d.assign(em=(d["y_pred"] - d["y_true"]).abs(), ep=(d["persistence"] - d["y_true"]).abs())
    return g.groupby("space_id").agg(n=("em", "size"), model=("em", "mean"), persistence=("ep", "mean")).reset_index()


def event_spans(events_path, split) -> pd.DataFrame:
    e = pd.read_parquet(events_path)
    a, b = split.utc("test")
    e = e[(e["kind"] == "B3") & (e["start"] >= pd.Timestamp(a)) & (e["start"] < pd.Timestamp(b))]
    return e


def schedule_value(with_df, without_df, bl, acfg, after_event_min=30) -> pd.DataFrame:
    """일정 포함 − 제외: 같은 씨앗끼리 MAE 차이 → 평균 ± 표준편차. 구간: 전체·사용 시간대·일정 경계 직후."""
    k = ["space_id", "item", "t0", "horizon_min", "seed"]
    m = with_df[k + ["y_true", "y_pred", "usage_hours"]].merge(
        without_df[k + ["y_pred"]].rename(columns={"y_pred": "y_wo"}), on=k)
    tgt = m["t0"] + pd.to_timedelta(m["horizon_min"], unit="min")
    kk = pd.DatetimeIndex(tgt).tz_convert(KST)
    mins = np.asarray(kk.hour * 60 + kk.minute)
    near = np.zeros(len(m), bool)
    for b in (540, 690, 780, 1080):
        near |= (mins >= b) & (mins < b + after_event_min)
    near &= ~offday_mask(pd.DatetimeIndex(tgt), acfg)
    rows = []
    for name, mask in (("전체", np.ones(len(m), bool)), ("사용 시간대", m["usage_hours"].to_numpy()),
                       (f"일정 경계 직후 {after_event_min}분", near)):
        d = m[mask]
        g = d.assign(a=(d["y_pred"] - d["y_true"]).abs(), b=(d["y_wo"] - d["y_true"]).abs()) \
             .groupby(["space_id", "item", "horizon_min", "seed"]).agg(a=("a", "mean"), b=("b", "mean"), n=("a", "size")).reset_index()
        g["diff"] = g["b"] - g["a"]                         # 양수 = 일정 정보로 줄어든 오차
        s = g.groupby(["space_id", "item", "horizon_min"]).agg(with_=("a", "mean"), without=("b", "mean"),
                                                               reduce=("diff", "mean"), reduce_sd=("diff", "std"),
                                                               n=("n", "first")).reset_index()
        s["구간"] = name
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def conclusion(sv: pd.DataFrame, space="lab", item="co2", h=60, part="전체") -> str:
    r = sv[(sv["space_id"] == space) & (sv["item"] == item) & (sv["horizon_min"] == h) & (sv["구간"] == part)]
    if r.empty:
        return "판정 불가(자료 없음)"
    r = r.iloc[0]
    unit = "ppm" if item == "co2" else "℃"
    sd = 0.0 if np.isnan(r["reduce_sd"]) else r["reduce_sd"]
    pct = r["reduce"] / r["without"] * 100
    name = SPACE_NAME.get(space, space)
    if abs(r["reduce"]) <= sd:
        return (f"{name} {item} {h}분({part}): 일정 정보의 효과는 씨앗 변동 안이라 구별할 수 없다"
                f"({r['reduce']:+.2f} ± {sd:.2f}{unit}).")
    verb = "줄어든다" if r["reduce"] > 0 else "늘어난다"
    return (f"일정 정보를 넣으면 {name} {'CO₂' if item == 'co2' else '온도'} {h}분 예측 오차가({part}) "
            f"**{abs(r['reduce']):.1f}{unit}({abs(pct):.0f}%) {verb}**(씨앗 변동 ± {sd:.1f}{unit}).")


# ---------- 그래프 ----------

def _plt(font="Noto Sans CJK KR"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_horizon_curves(table: pd.DataFrame, out: Path, name="horizon_mae.png") -> str:
    """table: comparison_table(all) — 공간 × 항목별 지평 곡선."""
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for j, s in enumerate(["lab", "office", "meeting"]):
        for i, item in enumerate(["co2", "temperature"]):
            ax = axes[i, j]
            d = table[(table["space_id"] == s) & (table["item"] == item)].sort_values("horizon_min")
            for col, lab, st in (("persistence", "현재값 유지", "--"), ("mean_1h", "최근 1시간 평균", ":"),
                                 ("yesterday", "어제 같은 시각", "-."), ("mae", "학습 모델", "-")):
                ax.plot(d["horizon_min"], d[col], st, marker="o", label=lab, lw=2 if col == "mae" else 1.2)
            ax.set_title(f"{SPACE_NAME[s]} {'CO2(ppm)' if item == 'co2' else '온도(℃)'}")
            ax.set_xticks([5, 15, 30, 60])
            ax.set_xlabel("지평(분)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("시험 기간 지평별 MAE — 학습 모델 vs 기준선 3종", fontsize=11)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=110)
    plt.close(fig)
    return name


def plot_hourly_error(df, bl, out: Path, name="hourly_error_co2_60.png") -> str:
    plt = _plt()
    d = with_baseline(seed_mean(df[(df["item"] == "co2") & (df["horizon_min"] == 60)]), bl)
    d["hod"] = (d["t0"] + pd.Timedelta(minutes=60)).dt.tz_convert(KST).dt.hour
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=False)
    for ax, s in zip(axes, ["lab", "office", "meeting"]):
        g = d[d["space_id"] == s].groupby("hod")
        ax.plot((g.apply(lambda x: (x["persistence"] - x["y_true"]).abs().mean())), "--", label="현재값 유지")
        ax.plot((g.apply(lambda x: (x["y_pred"] - x["y_true"]).abs().mean())), "-", lw=2, label="학습 모델")
        for x in (9, 11.5, 13, 18):
            ax.axvline(x, color="gray", lw=0.6, alpha=0.6)
        ax.set_title(SPACE_NAME[s])
        ax.set_xlabel("정답 시각(KST)")
        ax.set_xticks(range(0, 25, 3))
    axes[0].set_ylabel("CO2 60분 MAE(ppm)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / name, dpi=110)
    plt.close(fig)
    return name


def plot_day(df, bl, space, day, out: Path, name: str, h=60) -> str:
    """하루 동안 60분 앞 예측(정답 시각 기준) vs 실제 vs 현재값 유지."""
    plt = _plt()
    d = with_baseline(seed_mean(df[(df["item"] == "co2") & (df["horizon_min"] == h) & (df["space_id"] == space)]), bl)
    d["tt"] = (d["t0"] + pd.Timedelta(minutes=h)).dt.tz_convert(KST)
    d = d[d["tt"].dt.strftime("%Y-%m-%d") == day].sort_values("tt")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(d["tt"], d["y_true"], color="black", lw=1.5, label="실제")
    ax.plot(d["tt"], d["y_pred"], lw=1.6, label=f"학습 모델({h}분 전 예측)")
    ax.plot(d["tt"], d["persistence"], "--", lw=1.0, label=f"현재값 유지({h}분 전 값)")
    ax.set_title(f"{SPACE_NAME[space]} {day} CO2 — 정답 시각 기준")
    ax.set_ylabel("ppm")
    ax.legend(fontsize=8)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=KST))
    fig.tight_layout()
    fig.savefig(out / name, dpi=110)
    plt.close(fig)
    return name

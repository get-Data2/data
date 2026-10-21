"""단계 실행: labels → features → train → select → evaluate → analyze. 작업 파일은 var/state/ (data-model, research R13)."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_features, columns_for, space_grid
from .labels import build_labels, load_observations
from .models import SIMPLICITY, Candidate, Model, candidates
from .selection import (SelectionMissingError, freeze_selection, metrics, occ_threshold, pick, require_selection,
                        vent_threshold)

ROOT = Path("var/state")
INPUT_SPEC_VERSION = 1
PROBLEMS = ("occupancy", "vent", "leak")


class Data:
    """입력을 한 번 읽어 둔다. 시험에서는 series·norm·root를 직접 넣는다."""

    def __init__(self, cfg, root: Path = ROOT, series=None, norm=None, bookings=None, obs=None):
        self.cfg, self.root = cfg, Path(root)
        inp = cfg["inputs"]
        self.series = series if series is not None else pd.read_parquet(inp["series"])
        self.norm = norm if norm is not None else json.loads(Path(inp["norm"]).read_text())
        if bookings is None:
            bf = inp.get("booking_file")
            src = cfg["labels"]["occupancy"].get("meeting_source", "none")
            if bf and src == "booking_file":
                bookings = pd.read_csv(bf)
            elif src == "booking_estimated" and Path(inp["booking_estimated"]).exists():
                bookings = pd.read_parquet(inp["booking_estimated"])
        self.bookings = bookings
        self.obs = obs if obs is not None else load_observations(cfg)


# ---------- 정답·특징 ----------

def run_labels(cfg, data: Data | None = None, log=print):
    data = data or Data(cfg)
    occ, vent = build_labels(data.series, cfg, data.obs, data.bookings)
    data.root.mkdir(parents=True, exist_ok=True)
    occ.to_parquet(data.root / "labels_occupancy.parquet", index=False)
    vent.to_parquet(data.root / "labels_vent.parquet", index=False)
    from .features import assign_split
    v = vent.assign(split=assign_split(vent["bin"], cfg))
    t = v.groupby(["space_id", "split"])["label"].value_counts(normalize=True).unstack(fill_value=0).round(4)
    log("환기 라벨 비율(공간 × 분할):\n" + t.to_string())
    o = occ.dropna(subset=["obs_label"])
    if len(o):
        both = o.dropna(subset=["label"])
        log(f"관찰 칸 {len(o):,}, 일정 정답과 겹치는 칸 {len(both):,}, 불일치 {(both['label'] != both['obs_label']).mean() * 100:.1f}%")
    return occ, vent


def run_features(cfg, data: Data | None = None, log=print):
    data = data or Data(cfg)
    f = build_features(data.series, data.norm, cfg)
    data.root.mkdir(parents=True, exist_ok=True)
    f.to_parquet(data.root / "features.parquet", index=False)
    cols = {g: columns_for(gs) for g, gs in cfg["features"]["groups"].items()}
    (data.root / "features.json").write_text(json.dumps({"config": cfg.digest, "version": INPUT_SPEC_VERSION,
                                                         "columns": cols}, ensure_ascii=False, indent=1))
    log(f"특징 {len(f):,}칸 × {len(f.columns)}열, 분할별 {f['split'].value_counts().to_dict()}")
    return f


def dataset(root: Path, cfg, problem: str) -> pd.DataFrame:
    f = pd.read_parquet(root / "features.parquet")
    f = f[f["input_ok"]]
    if problem in ("occupancy", "leak"):
        lab = pd.read_parquet(root / "labels_occupancy.parquet")
        d = f.merge(lab, on=["space_id", "bin"], how="left")
        d["y"] = d["label"]
    else:
        lab = pd.read_parquet(root / "labels_vent.parquet")
        d = f.merge(lab, on=["space_id", "bin"], how="left")
        d = d[d["label"].isin(["positive", "negative"])]
        d["y"] = (d["label"] == "positive").astype(float)
    return d


def feat_cols(cfg, problem: str) -> list[str]:
    g = "leak" if problem == "leak" else problem
    return columns_for(cfg["features"]["groups"][g])


# ---------- 학습 ----------

def _run_dir(root, problem, cand, seed) -> Path:
    return root / "runs" / problem / f"{cand.hash}-s{seed}"


def _val_score(problem, cfg, y, p) -> dict:
    if problem == "vent":
        thr, ok, m = vent_threshold(y, p, cfg["selection"]["vent"]["threshold"]["min_recall"])
        return {"threshold": thr, "recall_ok": ok, **m}
    thr, m = occ_threshold(y, p)
    return {"threshold": thr, **m}


def run_train(cfg, problem: str = "all", device="auto", root: Path = ROOT, log=print):
    probs = PROBLEMS if problem == "all" else (problem,)
    for prob in probs:
        d = dataset(root, cfg, prob)
        tr = d[(d["split"] == "train") & d["y"].notna()]
        va = d[(d["split"] == "val") & d["y"].notna()]
        cols = feat_cols(cfg, prob)
        log(f"[{prob}] 학습 {len(tr):,}칸(양성 {tr['y'].mean() * 100:.1f}%), 검증 {len(va):,}칸, 특징 {len(cols)}")
        if tr["y"].nunique() < 2 or va.empty:
            log(f"[{prob}] 학습할 정답이 부족함 — 건너뜀")
            continue
        for cand in candidates(cfg):
            for seed in cfg["models"]["seeds"]:
                out = _run_dir(root, prob, cand, seed)
                if (out / "run.json").exists() and json.loads((out / "run.json").read_text())["status"] == "done":
                    continue
                out.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                try:
                    m = Model(cand).fit(tr[cols].to_numpy(), tr["y"].to_numpy(), seed)
                    p = m.predict_proba(va[cols].to_numpy())
                    b = m.to_bytes()
                    (out / "model.joblib").write_bytes(b)
                    pd.DataFrame({"space_id": va["space_id"].to_numpy(), "bin": va["bin"].to_numpy(), "y": va["y"].to_numpy(),
                                  "p": p}).to_parquet(out / "val_pred.parquet", index=False)
                    rec = {"status": "done", "problem": prob, "candidate": cand.label(), "kind": cand.kind,
                           "params": cand.params, "hash": cand.hash, "seed": seed, "features": cols,
                           "val": _val_score(prob, cfg, va["y"].to_numpy(), p), "model_sha256": hashlib.sha256(b).hexdigest()[:16],
                           "train_rows": int(len(tr)), "train_seconds": round(time.time() - t0, 2), "config": cfg.digest,
                           "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                except Exception as e:      # 한 후보 실패가 전체를 멈추지 않는다(헌법 VI)
                    rec = {"status": "failed", "problem": prob, "candidate": cand.label(), "hash": cand.hash, "seed": seed,
                           "error": f"{type(e).__name__}: {e}"}
                (out / "run.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str))
                v = rec.get("val", {})
                log(f"  {rec['candidate']} s{seed}: {rec['status']} macro-F1 {v.get('macro_f1', float('nan')):.4f}"
                    + (f" 재현율 {v.get('recall'):.3f}" if prob == "vent" and v else ""))


# ---------- 선택 고정(검증만) ----------

def _runs(root, prob) -> list[dict]:
    out = []
    for p in sorted((root / "runs" / prob).glob("*/run.json")):
        r = json.loads(p.read_text())
        if r["status"] == "done":
            r["_dir"] = p.parent.name
            out.append(r)
    return out


def _avg_val(root, prob, rids) -> pd.DataFrame:
    ps = [pd.read_parquet(root / "runs" / prob / rid / "val_pred.parquet") for rid in rids]
    d = ps[0].copy()
    d["p"] = np.mean([x["p"].to_numpy() for x in ps], axis=0)
    return d


def run_select(cfg, root: Path = ROOT, log=print) -> dict:
    body = {"config": cfg.digest}
    for prob in ("occupancy", "vent"):
        runs = _runs(root, prob)
        if not runs:
            raise SelectionMissingError(f"{prob} 학습 결과 없음 — `state train {prob}` 먼저")
        by = {}
        for r in runs:
            by.setdefault(r["hash"], []).append(r)
        keys = list(by)
        stats = [{"mean": float(np.mean([x["val"]["macro_f1"] for x in by[k]])),
                  "sd": float(np.std([x["val"]["macro_f1"] for x in by[k]])),
                  "simplicity": SIMPLICITY[by[k][0]["kind"]]} for k in keys]
        i, why = pick(stats)
        chosen = by[keys[i]]
        rids = [x["_dir"] for x in chosen]
        va = _avg_val(root, prob, rids)
        if prob == "vent":
            thr, ok, m = vent_threshold(va["y"], va["p"], cfg["selection"]["vent"]["threshold"]["min_recall"])
            extra = {"recall_ok": ok}
        else:
            thr, m = occ_threshold(va["y"], va["p"])
            extra = {}
        body[prob] = {"candidate": chosen[0]["candidate"], "kind": chosen[0]["kind"], "params": chosen[0]["params"],
                      "hash": keys[i], "runs": rids, "features": chosen[0]["features"], "threshold": thr, "reason": why,
                      "val": {k: v for k, v in m.items()}, **extra,
                      "model_sha256": {x["_dir"]: x["model_sha256"] for x in chosen},
                      "table": [{"candidate": by[k][0]["candidate"], "mean": s["mean"], "sd": s["sd"], "n": len(by[k])}
                                for k, s in zip(keys, stats)]}
        log(f"[{prob}] 선택: {chosen[0]['candidate']} ({why}), 임계값 {thr:.4f}, 검증 macro-F1 {m['macro_f1']:.4f}"
            + (f", 재현율 {m['recall']:.3f}{'' if extra.get('recall_ok', True) else ' (85% 미달)'}" if prob == "vent" else ""))
    # 누수 실험: 사용 여부와 같은 후보를 일정·시각 입력으로 학습한 모델(비교 전용, 선택 아님)
    leak = [r for r in _runs(root, "leak") if r["hash"] == body["occupancy"]["hash"]]
    if leak:
        rids = [x["_dir"] for x in leak]
        thr, m = occ_threshold(*(lambda d: (d["y"], d["p"]))(_avg_val(root, "leak", rids)))
        body["leak"] = {"candidate": leak[0]["candidate"], "runs": rids, "features": leak[0]["features"], "threshold": thr,
                        "model_sha256": {x["_dir"]: x["model_sha256"] for x in leak}, "note": "비교 전용(선택 아님)"}
    return freeze_selection(root / "selection.json", body)


# ---------- 평가 ----------

def _predict(root, prob, sel_p, X) -> np.ndarray:
    ps = [Model.from_bytes((root / "runs" / prob / rid / "model.joblib").read_bytes()).predict_proba(X) for rid in sel_p["runs"]]
    return np.mean(ps, axis=0)


def rule_occupancy(series, cfg, space, bins) -> np.ndarray:
    r = cfg["rules"]["occupancy"]
    w = space_grid(series, space)["co2"].ffill()
    k = r["window_min"] // 5
    rise = (w - w.shift(k)).reindex(pd.DatetimeIndex(bins).as_unit("ns"))
    return (rise.to_numpy() >= r["rise_ppm"]).astype(int)


def run_evaluate(cfg, device="auto", data: Data | None = None, log=print) -> dict:
    data = data or Data(cfg)
    root = data.root
    sel = require_selection(root / "selection.json", root / "runs")
    from .observe import read, status
    st = status(read(cfg["inputs"]["observations"]), cfg)
    obs_done = bool(st.attrs["done"])
    res = {"selection_sha256": sel["sha256"], "observation": {"done": obs_done, "status": st.attrs["total"]}, "vent": {}, "occupancy": {}}
    days = (cfg.split.test[1] - cfg.split.test[0]).days + 1
    (root / "eval").mkdir(parents=True, exist_ok=True)

    # 환기 경보(시험 기간)
    d = dataset(root, cfg, "vent")
    te = d[d["split"] == "test"].copy()
    te["p"] = _predict(root, "vent", sel["vent"], te[sel["vent"]["features"]].to_numpy())
    te["pred"] = (te["p"] >= sel["vent"]["threshold"]).astype(int)
    te["rule"] = (te["co2_now"] >= cfg["rules"]["vent"]["co2_ppm"]).astype(int)
    te.drop(columns=[c for c in te.columns if c.startswith(("lvl_", "d5_", "d15_", "d30_", "slope", "sd30", "rise", "age_", "t_", "sch_"))]) \
      .to_parquet(root / "eval" / "pred_vent.parquet", index=False)
    for name, col in (("classifier", "pred"), ("rule_850", "rule")):
        m = metrics(te["y"], te[col])
        m["false_alarms_per_day"] = m["fp"] / days
        res["vent"][name] = {"all": m, "by_space": {s: metrics(g["y"], g[col]) for s, g in te.groupby("space_id")}}
    tp = te[(te["pred"] == 1) & (te["y"] == 1)]
    lead = ((pd.to_datetime(tp["first_exceed"]) - pd.to_datetime(tp["bin"])).dt.total_seconds() / 60)
    res["vent"]["lead_min"] = {"median": float(lead.median()) if len(lead) else None, "p10": float(lead.quantile(0.1)) if len(lead) else None}
    from .forecast_vent import forecast_vent_eval
    res["vent"]["forecast_model"] = forecast_vent_eval(cfg, data, te, sel, log)

    # 사용 여부: 일정 기준(시험 기간, 일정 정답이 있는 공간) + 관찰 기준(관찰 칸 전체)
    d = dataset(root, cfg, "occupancy")
    d["p"] = _predict(root, "occupancy", sel["occupancy"], d[sel["occupancy"]["features"]].to_numpy())
    d["pred"] = (d["p"] >= sel["occupancy"]["threshold"]).astype(int)
    if "leak" in sel:
        d["p_leak"] = _predict(root, "leak", sel["leak"], d[sel["leak"]["features"]].to_numpy())
        d["pred_leak"] = (d["p_leak"] >= sel["leak"]["threshold"]).astype(int)
    d["rule"] = 0
    for s, g in d.groupby("space_id"):
        d.loc[g.index, "rule"] = rule_occupancy(data.series, cfg, s, g["bin"])
    keep = ["space_id", "bin", "split", "y", "obs_label", "obs_people", "p", "pred", "rule"] + [c for c in ("p_leak", "pred_leak") if c in d]
    d[keep].to_parquet(root / "eval" / "pred_occupancy.parquet", index=False)
    methods = [("classifier", "pred"), ("rule_15min_30ppm", "rule")] + ([("leak_with_schedule", "pred_leak")] if "pred_leak" in d else [])
    sch = d[(d["split"] == "test") & d["y"].notna()]
    ob = d[d["obs_label"].notna()]
    for basis, sub, ycol in (("schedule_test", sch, "y"), ("observation", ob, "obs_label")):
        res["occupancy"][basis] = {name: {"all": metrics(sub[ycol], sub[c]) if len(sub) else None,
                                          "by_space": {s: metrics(g[ycol], g[c]) for s, g in sub.groupby("space_id")}}
                                   for name, c in methods}
    res["occupancy"]["observation_rows"] = int(len(ob))
    res["criteria"] = criteria(res, obs_done)
    (root / "eval" / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    save_bundle(cfg, data, sel)
    for r in res["criteria"]:
        log(f"  {r['기준']}: {r['결과']} {r['판정']}")
    return res


def criteria(res, obs_done: bool) -> list[dict]:
    v, vr = res["vent"]["classifier"]["all"], res["vent"]["rule_850"]["all"]
    out = [{"기준": "환기 경보 재현율 ≥ 85%", "결과": f"{v['recall'] * 100:.1f}%", "판정": "✅" if v["recall"] >= 0.85 else "❌"},
           {"기준": "환기 경보 macro-F1 > 단순 규칙(850ppm)", "결과": f"{v['macro_f1']:.3f} vs {vr['macro_f1']:.3f}",
            "판정": "✅" if v["macro_f1"] > vr["macro_f1"] else "❌"}]
    o = res["occupancy"]["observation"]
    oc, orl = o["classifier"]["all"], o["rule_15min_30ppm"]["all"]
    if not obs_done or oc is None:
        out.append({"기준": "사용 여부(직접 관찰 기준) 정확도·F1·비어있음→사용중 오류율, 규칙보다 좋음",
                    "결과": f"관찰 목표 미달({res['observation']['status']})", "판정": "⏸ 판단 보류"})
    else:
        better = oc["macro_f1"] > orl["macro_f1"]
        out.append({"기준": "사용 여부(직접 관찰 기준) 규칙보다 좋음",
                    "결과": f"정확도 {oc['accuracy']:.3f}, F1 {oc['f1']:.3f}, 비어있음→사용중 {oc['false_occupied_rate']:.3f} "
                           f"(규칙 macro-F1 {orl['macro_f1']:.3f})", "판정": "✅" if better else "❌"})
    return out


def save_bundle(cfg, data: Data, sel: dict) -> Path:
    """과제 8이 읽는 묶음(spec FR-062): 문제별 모델(씨앗 평균), 입력 규약 버전, 특징 순서, 정규화, 임계값, 해시."""
    import joblib
    out = data.root / "bundle"
    out.mkdir(parents=True, exist_ok=True)
    body = {"input_spec_version": INPUT_SPEC_VERSION, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "selection_sha256": sel["sha256"], "config": cfg.digest, "spaces": cfg.spaces,
            "feature_window_min": cfg["features"]["window_min"], "min_input_obs": cfg["features"]["min_input_obs"],
            "norm": {s: {it: {"mean": v["mean"], "std": v["std"]} for it, v in data.norm["spaces"][s].items()}
                     for s in cfg.spaces if s in data.norm["spaces"]},
            "preprocess": "analysis 공간 5분 시계열(관측 칸만, 앞으로 채우기) — 과제 8 src/preprocess/로 이전 예정",
            "problems": {}}
    for prob in ("occupancy", "vent"):
        p = sel[prob]
        models = [(data.root / "runs" / prob / rid / "model.joblib").read_bytes() for rid in p["runs"]]
        f = out / f"{prob}.joblib"
        joblib.dump({"models": models, "average": "mean_proba"}, f)
        body["problems"][prob] = {"kind": p["kind"], "file": f.name, "features": p["features"], "threshold": p["threshold"],
                                  "seeds": len(models), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()[:16],
                                  "tensorrt_target": p["kind"] == "mlp"}
    (out / "bundle.json").write_text(json.dumps(body, ensure_ascii=False, indent=1))
    return out


def load_bundle(path: Path) -> dict:
    """과제 8용 읽기: bundle.json + 문제별 Model 목록."""
    import io
    import joblib
    body = json.loads((Path(path) / "bundle.json").read_text())
    if body["input_spec_version"] != INPUT_SPEC_VERSION:
        raise ValueError(f"입력 규약 버전 불일치: {body['input_spec_version']} ≠ {INPUT_SPEC_VERSION}")
    for prob, p in body["problems"].items():
        d = joblib.load(Path(path) / p["file"])
        p["models"] = [Model.from_bytes(b) for b in d["models"]]
    return body


def run_analyze(cfg, data: Data | None = None, log=print):
    from .analyze import analyze_all
    data = data or Data(cfg)
    return analyze_all(cfg, data, log)

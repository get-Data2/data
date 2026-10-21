"""공간별 결과 만들기: 예측·1,000ppm 첫 초과·이상·상태·규칙. 판단마다 예외를 격리한다(한 판단 실패가 다른 판단을 멈추지 않음)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import rules
from .config import SPACE_NAME
from .prep import Prepared, forecast_input

SCHEMA = "team2.inference.v1"
HORIZONS = list(range(5, 65, 5))


def iso(t) -> str | None:
    return None if t is None else pd.Timestamp(t).isoformat().replace("+00:00", "Z")


def kst(t) -> str:
    return pd.Timestamp(t).tz_convert("Asia/Seoul").isoformat()


def _co2_over(co2: list[float], current: float, thr: float) -> dict:
    first = next((h for h, v in zip(HORIZONS, co2) if v >= thr), None)
    return {"threshold_ppm": thr, "first_over_min": first if current < thr else 0, "current_ppm": round(float(current), 1)}


def compute(prep: Prepared, bundles: dict, cfg, rule_state: dict) -> tuple[dict[str, dict], list[dict]]:
    """→ (공간별 결과, 경보 판정 목록)."""
    acfg = cfg.acfg
    fr = cfg["freshness"]
    thr = float(cfg["thresholds"]["co2_ppm"])
    findings: list[dict] = []
    results: dict[str, dict] = {}

    # 규칙(장치 단위, 공간에 나눠 담는다)
    rule_out = {"data_gap": [], "stuck": [], "out_of_range": []}
    try:
        rule_out["data_gap"] = rules.data_gap(prep.raw, prep.now, acfg, rule_state.setdefault("known", {}),
                                              cfg["rules"]["data_gap_min"])
        rule_out["stuck"] = rules.stuck(prep.sensor, cfg["rules"]["stuck"])
        rule_out["out_of_range"] = rules.out_of_range(prep.raw, prep.quality, prep.now, cfg["rules"]["out_of_range_recent_min"])
        rules_avail = "ok"
    except Exception as e:
        rules_avail = f"error:{type(e).__name__}"
    for kind, items in rule_out.items():
        for r in items:
            if r["space_id"] in cfg.spaces:
                findings.append({"kind": kind, "space_id": r["space_id"], "device_id": r["device_id"], "evidence": r})

    # 이상·상태(과제 4·5 묶음, 공간 전체를 한 번에)
    plug = {}
    for name in ("anomaly", "state"):
        b = bundles[name]
        if not b.ok:
            plug[name] = ({}, b.availability())
            continue
        try:
            plug[name] = (b.obj.predict(prep) or {}, "ok")
        except Exception as e:
            plug[name] = ({}, f"error:{type(e).__name__}")

    for space in cfg.spaces:
        fresh = prep.freshness_min.get(space)
        stale = fresh is None or fresh > fr["stale_min"]
        stop = fresh is None or fresh > fr["stop_min"]
        res = {"schema": SCHEMA, "space_id": space, "space_name": SPACE_NAME.get(space, space),
               "t0": iso(prep.end), "t0_kst": kst(prep.end), "computed_at": None,
               "freshness_min": fresh, "stale": bool(stale), "availability": {}, "models": {},
               "forecast": None, "co2_over": None, "anomaly": None, "state": None,
               "rules": {k: [r for r in v if r["space_id"] == space] for k, v in rule_out.items()}}
        res["availability"]["rules"] = rules_avail

        # 예측
        fb = bundles["forecast"]
        if not fb.ok:
            res["availability"]["forecast"] = fb.availability()
        elif stop:
            res["availability"]["forecast"] = "unavailable:data_gap"
        else:
            try:
                runner = fb.obj
                inp = forecast_input(prep, runner, space, acfg) if space in runner.spaces else None
                if inp is None:
                    res["availability"]["forecast"] = "unavailable:input_missing"
                else:
                    y = runner.predict(*inp[:2], space, inp[2])
                    co2 = [round(float(v), 1) for v in y[:, 0]]
                    tmp = [round(float(v), 2) for v in y[:, 1]]
                    res["forecast"] = {"unit": {"co2": "ppm", "temperature": "°C"}, "step_min": 5,
                                       "horizons_min": HORIZONS, "co2": co2, "temperature": tmp, "interval": None}
                    res["co2_over"] = _co2_over(co2, float(inp[2][0]), thr)
                    res["availability"]["forecast"] = "ok"
                    fo = res["co2_over"]["first_over_min"]
                    if fo:
                        findings.append({"kind": "co2_over_forecast", "space_id": space, "device_id": None,
                                         "evidence": res["co2_over"]})
            except Exception as e:
                res["availability"]["forecast"] = f"error:{type(e).__name__}"
            res["models"]["forecast"] = fb.model_meta()

        # 이상·상태
        for name in ("anomaly", "state"):
            out, avail = plug[name]
            if avail != "ok":
                res["availability"][name] = avail
                continue
            if name == "anomaly" and stale:
                res["availability"][name] = "unavailable:stale"      # 신선도 초과 시 이상 판단 보류(research R6)
                continue
            r = out.get(space)
            if r is None:
                res["availability"][name] = "unavailable:no_result"
                continue
            res[name] = r
            res["availability"][name] = "ok"
            res["models"][name] = bundles[name].model_meta()
            if name == "anomaly" and r.get("is_anomaly"):
                findings.append({"kind": "anomaly", "space_id": space, "device_id": (r.get("causes") or [{}])[0].get("device_id"),
                                 "evidence": {"score_q": r.get("score_q"), "causes": r.get("causes", [])[:3]}})
            if name == "state" and r.get("ventilation_soon"):
                findings.append({"kind": "ventilation_soon", "space_id": space, "device_id": None,
                                 "evidence": {"p_vent": r.get("p_vent")}})
        res["summary_ko"] = summary(res)
        results[space] = res
    return results, findings


def summary(r: dict) -> str:
    name = r["space_name"]
    parts = []
    co = r.get("co2_over")
    if co:
        if co["first_over_min"] == 0:
            parts.append(f"CO₂ 이미 {co['threshold_ppm']:.0f}ppm 이상(현재 {co['current_ppm']:.0f}ppm)")
        elif co["first_over_min"]:
            parts.append(f"CO₂ {co['first_over_min']}분 뒤 {co['threshold_ppm']:,.0f}ppm 초과 예상(현재 {co['current_ppm']:.0f}ppm)")
        else:
            parts.append(f"CO₂ 60분 안 {co['threshold_ppm']:,.0f}ppm 초과 없음(현재 {co['current_ppm']:.0f}ppm)")
    elif r["availability"].get("forecast", "").startswith("unavailable"):
        parts.append(f"예측 불가({r['availability']['forecast'].split(':', 1)[1]})")
    gaps = r["rules"].get("data_gap") or []
    if gaps:
        parts.append(f"데이터 끊김 장치 {len(gaps)}대")
    if r.get("anomaly") and r["anomaly"].get("is_anomaly"):
        parts.append("이상 감지")
    if r.get("stale"):
        parts.append("데이터 오래됨")
    return f"{name} " + (", ".join(parts) if parts else "판단 없음")

"""메모리 예산표: 관찰값(tegrastats·프로세스) + 측정값(모델) + 가정 LLM(1B·3B·7B 4비트) (research R9, 결정 기록)."""
from __future__ import annotations

import pandas as pd


def budget_table(cfg, measured_mb: dict[str, float], observed: dict[str, float] | None = None) -> pd.DataFrame:
    """LLM 가정마다 합계와 판정(보드 총량 × limit_ratio 이하)."""
    b = cfg["budget"]
    obs = {k: v for k, v in (b.get("observed") or {}).items() if v is not None}
    obs |= {k: v for k, v in (observed or {}).items() if v is not None}
    limit = b["board_total"] * b["limit_ratio"]
    rows = []
    for llm, llm_mb in b["llm_assumptions"].items():
        for model, mb in measured_mb.items():
            total = sum(obs.values()) + mb + llm_mb
            rows.append({"LLM 가정": llm, "시계열 모델": model, "관찰(OS·수집기·다른 계정)": round(sum(obs.values())),
                         "모델 측정": round(mb), "LLM": llm_mb, "합계": round(total), "한도": round(limit),
                         "판정": "들어감" if total <= limit else "넘침"})
    return pd.DataFrame(rows)

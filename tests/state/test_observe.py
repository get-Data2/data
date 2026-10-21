from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from state import observe as ob
from tests.state.conftest import synth_obs

pytestmark = pytest.mark.unit
SP = ["lab", "office", "meeting"]


def T(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M")


def test_transition_closes_previous(tmp_path):
    p = tmp_path / "o.csv"
    ob.record(p, "lab", "empty", SP, at=T("2026-10-22 08:40"), observer="KW")
    msg = ob.record(p, "lab", "occupied", SP, at=T("2026-10-22 09:03"), people=25, observer="KW")
    assert "닫음" in msg
    ob.record(p, "lab", "stop", SP, at=T("2026-10-22 09:40"))
    df = ob.read(p)
    assert list(df["state"]) == ["empty", "occupied"]
    assert list(df["end"]) == ["2026-10-22 09:03", "2026-10-22 09:40"]
    assert df.iloc[1]["people"] == "25"
    assert ob.validate(df, SP) == []


def test_rejects_overlap_unknown_and_stop_without_open(tmp_path):
    p = tmp_path / "o.csv"
    with pytest.raises(ob.ObservationError, match="모르는 공간"):
        ob.record(p, "hall", "empty", SP)
    with pytest.raises(ob.ObservationError, match="열린 구간이 없음"):
        ob.record(p, "lab", "stop", SP)
    ob.record(p, "lab", "empty", SP, at=T("2026-10-22 08:00"))
    ob.record(p, "lab", "stop", SP, at=T("2026-10-22 09:00"))
    with pytest.raises(ob.ObservationError, match="겹침"):
        ob.record(p, "lab", "occupied", SP, at=T("2026-10-22 08:30"))
    with pytest.raises(ob.ObservationError, match="머리글자"):
        ob.record(p, "office", "empty", SP, at=T("2026-10-22 10:00"), observer="홍길동홍길동")


def test_validate_reports_line_numbers(tmp_path):
    p = synth_obs(tmp_path / "o.csv", [
        ["lab", "2026-10-22 09:00", "2026-10-22 10:00", "occupied", "", "KW", ""],
        ["lab", "2026-10-22 09:30", "2026-10-22 11:00", "empty", "", "KW", ""],      # 겹침(줄 3)
        ["office", "2026-10-22 12:00", "2026-10-22 11:00", "empty", "", "KW", ""],   # 끝 ≤ 시작(줄 4)
        ["office", "2026/10/22 13:00", "", "occupied", "", "KW", ""],               # 형식(줄 5)
        ["hall", "2026-10-22 13:00", "2026-10-22 14:00", "occupied", "", "KW", ""],  # 공간(줄 6)
        ["meeting", "2026-10-22 13:00", "2026-10-22 14:00", "busy", "x", "KW", ""],  # 상태·인원(줄 7)
    ])
    probs = ob.validate(ob.read(p), SP, "2026-10-20")
    lines = {p.line for p in probs}
    assert {3, 4, 5, 6, 7} <= lines


def test_status_progress_and_boundaries(tmp_path, scfg):
    p = tmp_path / "o.csv"
    rows = []
    for d in ("2026-10-22", "2026-10-23"):                   # 목·금(수업일)
        rows += [["lab", f"{d} 08:20", f"{d} 09:00", "empty", "", "KW", ""],
                 ["lab", f"{d} 09:00", f"{d} 11:45", "occupied", "20", "KW", ""],
                 ["lab", f"{d} 11:45", f"{d} 13:40", "empty", "", "KW", ""]]
    synth_obs(p, rows)
    t = ob.status(ob.read(p), scfg, now=T("2026-10-24 00:00")).set_index("공간")
    assert t.loc["lab", "수업일 수"] == 2
    # 09:00·11:30·13:00 ±30분을 이틀 → 6회
    assert t.loc["lab", "경계 포함"] == 6
    assert t.loc["lab", "관찰 시간"] == pytest.approx(2 * (40 + 165 + 115) / 60, abs=0.1)
    assert t.loc["lab", "진행률(수업일)"] == "2/2" and t.loc["office", "관찰 시간"] == 0
    assert not t.attrs["done"]


def test_door_mismatch(tmp_path):
    p = synth_obs(tmp_path / "o.csv", [["office", "2026-10-22 12:00", "2026-10-22 13:00", "empty", "", "KW", ""],
                                       ["office", "2026-10-22 13:00", "2026-10-22 14:00", "occupied", "", "KW", ""]])
    ev = pd.DataFrame({"space_id": ["office", "office"], "t": [T("2026-10-22 12:30"), T("2026-10-22 13:30")]})
    mm = ob.door_mismatch(ob.read(p), ev)
    assert len(mm) == 1 and mm.iloc[0]["문 열림 수"] == 1


def test_cli_observe_roundtrip(tmp_path, monkeypatch, capsys):
    import state.__main__ as cli
    from tests.conftest import ROOT
    obs = tmp_path / "obs.csv"
    cfgp = tmp_path / "state.yaml"
    txt = (ROOT / "config" / "state.yaml").read_text().replace("data/observations.csv", str(obs))
    cfgp.write_text(txt)
    monkeypatch.chdir(ROOT)
    assert cli.main(["--config", str(cfgp), "observe", "office", "occupied", "--at", "2026-10-22 09:00", "--people", "3", "--by", "KW"]) == 0
    assert cli.main(["--config", str(cfgp), "observe", "office", "stop", "--at", "2026-10-22 10:00"]) == 0
    assert cli.main(["--config", str(cfgp), "observe", "--check"]) == 0
    assert cli.main(["--config", str(cfgp), "observe", "--status"]) == 0
    out = capsys.readouterr().out
    assert "office" in out and "1.0" in out
    assert cli.main(["--config", str(cfgp), "observe", "office", "stop"]) == 3

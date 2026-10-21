import numpy as np
import pytest

from edgeopt.bench import contention, latency_stats, measure, other_users_gpu, our_training

pytestmark = pytest.mark.unit


def test_latency_stats_known():
    ms = np.arange(1, 101, dtype=float)                    # 1..100ms
    s = latency_stats(ms, chunks=3)
    assert s["mean"] == 50.5 and s["median"] == 50.5 and s["max"] == 100
    assert s["p95"] == pytest.approx(np.percentile(ms, 95)) and len(s["chunk_p95"]) == 3
    assert s["chunk_p95"][0] < s["chunk_p95"][2]


def test_refuses_when_our_training_runs():
    procs = [(10, "1002", "/x/.venv/bin/python -m forecast run D3 --retry-failed"),
             (11, "1003", "/home/t2/venv/bin/python forecast.py"),           # 다른 계정 — 막지 않음, 기록만
             (12, "1002", "/x/.venv/bin/python -m edgeopt bench")]
    assert our_training(procs, uid="1002") == [procs[0][2]]
    assert our_training(procs[1:], uid="1002") == []
    assert other_users_gpu(procs, uid="1002") == ["/home/t2/venv/bin/python forecast.py"]


def test_contention():
    assert contention(20.0, 5) and not contention(1.0, 5) and not contention(float("nan"), 5)


def test_measure_loop_counts_and_syncs():
    class R:
        calls, syncs = 0, 0

        def run(self, x):
            R.calls += 1

        def sync(self):
            R.syncs += 1

    t = measure(R(), {}, warmup=5, repeat=20)
    assert len(t) == 20 and R.calls == 25 and R.syncs == 21 and (t >= 0).all()

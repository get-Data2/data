"""조회 API (spec US2, tasks T027): 엔드포인트별 응답, 오류 코드, 실제 HTTP 응답 시간."""
import json
import time
import urllib.error
import urllib.request

import numpy as np
import pytest
import torch

from service.api import Api, PortInUseError
from service.loop import Service
from tests.service.conftest import FakeSource, at

pytestmark = pytest.mark.unit


@pytest.fixture
def api(scfg, raw, fc_bundle):
    svc = Service(scfg, FakeSource(raw), torch.device("cpu"))
    svc.step(now=at("10:03:20"))
    a = Api(svc.store, scfg)
    port = a.start("127.0.0.1", 0)
    yield a, port, svc
    a.stop()


def get(port, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_routes(api):
    _, port, svc = api
    assert get(port, "/v1/spaces")[0] == 200
    c, b = get(port, "/v1/spaces/meeting/forecast?minutes=30")
    assert c == 200 and b["minutes"] == 30 and b["co2"]["unit"] == "ppm" and b["target_time"] == "2026-03-04T01:30:00Z"
    assert get(port, "/v1/spaces/lab/forecast?minutes=33")[1]["minutes"] == 35     # 5분 단위 반올림
    assert get(port, "/v1/spaces/lab")[0] == 200
    assert get(port, "/v1/alerts?status=active")[0] == 200
    assert get(port, "/v1/health")[1]["status"] in ("ok", "degraded")
    assert get(port, "/v1/openapi.json")[1]["openapi"].startswith("3")


def test_errors(api):
    _, port, _ = api
    assert get(port, "/v1/spaces/lab/forecast?minutes=90") == (400, {"error": "minutes는 5~60"})
    assert get(port, "/v1/spaces/lab/forecast?minutes=abc")[0] == 400
    assert get(port, "/v1/spaces/lab2")[0] == 404
    c, b = get(port, "/v1/spaces/lab/anomaly")
    assert c == 503 and b["reason"] == "no_bundle"


def test_first_cycle_pending(scfg):
    from service.store import Store
    a = Api(Store(), scfg)
    assert a.route("/v1/spaces/lab", {})[0] == 503


def test_response_time_p95(api):
    _, port, _ = api
    ts = []
    for _ in range(50):
        t = time.perf_counter()
        get(port, "/v1/spaces/lab/forecast?minutes=30")
        ts.append(time.perf_counter() - t)
    assert np.quantile(ts, 0.95) < 0.2


def test_port_in_use(api, scfg):
    a, port, svc = api
    with pytest.raises(PortInUseError):
        Api(svc.store, scfg).start("127.0.0.1", port)

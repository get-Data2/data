"""실제 TensorRT가 필요한 시험(마커 trt — 기본 실행에서 빠짐). GPU가 한가할 때 `pytest -m trt tests/edgeopt`."""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.trt


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU 없음(그룹 미반영이면 tools/edgeopt 래퍼 환경에서 실행)")
def test_fp16_engine_close_to_torch(tmp_path):
    from edgeopt.export import export_onnx
    from edgeopt.models import forecast_module
    from edgeopt.runners import TorchRunner
    from edgeopt.trt import TrtRunner, build_engine
    from tests.edgeopt.conftest import forecast_state, small_exp
    exp = small_exp("tcn")
    ld = forecast_module(exp, forecast_state(exp))
    export_onnx(ld, tmp_path / "m.onnx")
    rec = build_engine(tmp_path / "m.onnx", tmp_path / "m.plan", ld.input_shapes, {"min": 1, "opt": 3, "max": 4}, 64)
    assert rec["ok"], rec
    ex = ld.example(3)
    out = TrtRunner(tmp_path / "m.plan", ld.input_names, ld.output_names).run(ex)
    assert np.max(np.abs(out - TorchRunner(ld, "cpu").run(ex))) <= 1e-2

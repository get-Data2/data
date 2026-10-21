import numpy as np
import pytest
import torch

from edgeopt.export import export_onnx
from edgeopt.models import ae_module, forecast_module, infer_forecast_dims
from edgeopt.runners import OrtRunner, TorchRunner
from tests.edgeopt.conftest import forecast_state, small_exp

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("model,output,target", [("gru", "direct", "delta"), ("lstm", "direct", "value"),
                                                  ("tcn", "direct", "delta"), ("gru", "recursive", "delta")])
def test_forecast_export_matches_torch(tmp_path, model, output, target):
    exp = small_exp(model, output, target)
    st = forecast_state(exp)
    assert infer_forecast_dims(st, exp) == (19, 10, 3)
    ld = forecast_module(exp, st)
    rec = export_onnx(ld, tmp_path / "m.onnx", 17)
    assert rec["verify"]["ok"], rec["verify"]                  # 배치 1·3·4에서 ORT CPU = PyTorch CPU(1e-5)
    if output == "recursive":
        assert ld.input_names[-3:] == ["current", "mu", "sd"]
    ex = ld.example(4)
    a = TorchRunner(ld, "cpu").run(ex)
    b = OrtRunner("ort_cpu", tmp_path / "m.onnx", ld.input_names).run(ex)
    assert a.shape == (4, 12, 2) and np.max(np.abs(a - b)) <= 1e-5


def test_autoencoder_export(tmp_path):
    from anomaly.autoencoder import ConvAE
    torch.manual_seed(0)
    st = ConvAE(2 * 5 + 5, 5, 12, 8, 4).state_dict()
    ld = ae_module(5, 12, 8, 4, st)
    rec = export_onnx(ld, tmp_path / "ae.onnx")
    assert rec["verify"]["ok"]
    out = OrtRunner("ort_cpu", tmp_path / "ae.onnx", ["x"]).run(ld.example(3))
    assert out.shape == (3, 12, 5)


def test_missing_provider_is_reported(tmp_path):
    """이 판에 없는 실행 방식(ORT TensorRT EP)은 조용히 CPU로 바뀌지 않고 오류로 알린다."""
    exp = small_exp()
    ld = forecast_module(exp, forecast_state(exp))
    export_onnx(ld, tmp_path / "m.onnx")
    import onnxruntime as ort
    if "TensorrtExecutionProvider" in ort.get_available_providers():
        pytest.skip("TensorRT EP가 있는 판")
    with pytest.raises(RuntimeError, match="TensorrtExecutionProvider"):
        OrtRunner("ort_trt", tmp_path / "m.onnx", ld.input_names)

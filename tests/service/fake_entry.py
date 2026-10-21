"""시험용 과제 4·5 묶음 진입점(service_entry). 실제 진입점은 각 과제가 제공한다."""


class _Fake:
    version = "fake-1"
    sha256 = "fake"

    def __init__(self, kind, fail=False):
        self.kind, self.fail = kind, fail

    def predict(self, prep):
        if self.fail:
            raise RuntimeError("시험용 모델 오류")
        out = {}
        for s in ("lab", "office", "meeting"):
            if self.kind == "anomaly":
                out[s] = {"is_anomaly": s == "lab", "score_q": 0.9999 if s == "lab" else 0.5, "threshold_q": 0.999,
                          "since": None, "causes": [{"device_id": "24e124126d152590", "item": "temperature",
                                                     "contribution": 0.8}]}
            else:
                out[s] = {"occupied": True, "p_occupied": 0.9, "ventilation_soon": s == "meeting", "p_vent": 0.7}
        return out


def anomaly(path, device):
    return _Fake("anomaly")


def state(path, device):
    return _Fake("state")


def broken(path, device):
    return _Fake("anomaly", fail=True)

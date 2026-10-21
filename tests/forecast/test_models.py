from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from forecast.data import build_arrays, split_t0
from forecast.dataset import Windows
from forecast.models import ForecastModel, tcn_levels
from forecast.train import masked_l1

pytestmark = pytest.mark.unit
NO_BOOK = pd.DataFrame(columns=["space_id", "start", "end", "source"])


@pytest.fixture
def setup(fcfg, series, norm):
    a = build_arrays(fcfg, series, bookings=NO_BOOK)
    return fcfg, a, norm


@pytest.mark.parametrize("model", ["gru", "lstm", "tcn"])
@pytest.mark.parametrize("output", ["direct", "recursive"])
def test_output_shape(setup, model, output):
    cfg, a, norm = setup
    exp = replace(cfg.defaults(), model=model, output=output)
    W = Windows(a, exp, norm, 12, torch.device("cpu"))
    idx = split_t0(a, cfg, "train", W.L)[:7]
    m = ForecastModel(exp, W.fp, W.ff, W.n_space, 12)
    assert m(W.batch(idx)).shape == (7, 12, 2)


def test_tcn_receptive_field():
    for L in (6, 12, 24, 48, 288):
        n = tcn_levels(L, 3)
        assert 1 + 2 * 2 * (2 ** n - 1) >= L


def test_zero_delta_is_persistence(setup):
    cfg, a, norm = setup
    exp = cfg.defaults()
    W = Windows(a, exp, norm, 12, torch.device("cpu"))
    idx = split_t0(a, cfg, "val", W.L)[:20]
    b = W.batch(idx)
    out = W.to_original(torch.zeros(20, 12, 2), b)
    np.testing.assert_allclose(out[:, 5].numpy(), b["current"].numpy())


def test_masked_l1_ignores_masked():
    p = torch.zeros(2, 12, 2)
    t = torch.ones(2, 12, 2)
    m = torch.zeros(2, 12, 2, dtype=torch.bool)
    m[0, 0, 0] = True
    t2 = t.clone()
    t2[1] = 999
    assert masked_l1(p, t, m) == masked_l1(p, t2, m) == 1.0

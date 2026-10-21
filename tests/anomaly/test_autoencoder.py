import numpy as np
import pandas as pd
import pytest
import torch

from anomaly.autoencoder import ConvAE, column_errors, make_inputs, masked_mse, space_score, train_ae, window_starts
from anomaly.data import sensor_stats, space_cols, wide

pytestmark = pytest.mark.unit
CPU = torch.device("cpu")


def test_shapes_and_window_exclusion():
    m = ConvAE(10, 4, 12)
    assert m(torch.zeros(3, 12, 10)).shape == (3, 12, 4)
    ex = np.zeros(100, bool)
    ex[50] = True
    e = window_starts(100, 12, ex)
    assert not any((x - 11 <= 50 <= x) for x in e) and e[0] == 11


def test_masked_mse_ignores_masked():
    p, t = torch.zeros(1, 2, 2), torch.ones(1, 2, 2)
    m = torch.tensor([[[True, False], [False, False]]])
    t2 = t.clone()
    t2[0, 1] = 50
    assert masked_mse(p, t, m) == masked_mse(p, t2, m) == 1.0


def test_injected_spike_is_top_cause(ncfg, sensors):
    ncfg.raw["autoencoder"]["train"]["max_epochs"] = 3
    space = "lab"
    cols = space_cols(sensors, space, ncfg)
    tr = wide(sensors, space, cols, *ncfg.split.utc("train"))
    va = wide(sensors, space, cols, *ncfg.split.utc("val"))
    st = sensor_stats(tr)
    Xtr, Ztr = make_inputs(tr, st, ncfg.acfg)
    Xva, Zva = make_inputs(va, st, ncfg.acfg)
    W = ncfg["autoencoder"]["window"]
    model, rec = train_ae(Xtr, Ztr, window_starts(len(tr), W), Xva, Zva, window_starts(len(va), W), ncfg, CPU, 0,
                          log=lambda *_: None)
    va2 = va.copy()
    i = 700
    va2.iloc[i, va2.columns.get_loc("L2/co2")] += 1500
    X2, Z2 = make_inputs(va2, st, ncfg.acfg)
    err = column_errors(model, X2, Z2, W, 3, CPU)
    norm = np.nanmedian(column_errors(model, Xva, Zva, W, 3, CPU), 0)
    sc = space_score(err, norm, cols, va2.index, space)
    row = sc[sc["t"] == va2.index[i]].iloc[0]
    assert row["cause"] == "L2/co2"
    assert row["score"] > sc["score"].quantile(0.99)

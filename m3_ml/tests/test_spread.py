"""Study 4 forecasts use no data after the origin; the spread series is clean."""
import numpy as np
import pytest

from m3_ml.spread import run as sp


@pytest.fixture(scope='module')
def data():
    full = sp.Data('brent_wti')
    origins = sp.fold_origins(full, 20, 2024)[:30]
    cut = sp.Data('brent_wti', end=full.dates.iloc[int(origins[-1])])
    return full, cut, origins


@pytest.mark.parametrize('method', list(sp.METHODS))
def test_no_future_data(data, method):
    full, cut, origins = data
    params = sp.METHODS[method][1][-1]
    a = sp.walk_forward(full, method, params, 20, origins, 63)
    b = sp.walk_forward(cut, method, params, 20, origins, 63)
    assert np.allclose(a, b), method


def test_series(data):
    full, _, _ = data
    assert full.dates.is_monotonic_increasing and full.dates.is_unique
    assert full.raw_s.notna().all() and full.raw_s.abs().max() < 40
    h = 5
    assert np.allclose(full.y[h].iloc[:-h], (full.raw_s.shift(-h) - full.s).iloc[:-h], equal_nan=True)

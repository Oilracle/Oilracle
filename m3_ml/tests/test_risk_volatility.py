"""Study 1 forecasts and features use no data after the forecast origin."""
import numpy as np
import pandas as pd
import pytest

from m3_ml.common.data import load_table
from m3_ml.distribution import run as dist
from m3_ml.kronos import corpus
from m3_ml.risk import volatility as vol
from m3_ml.risk.features import realized_variance, volatility_features

pytestmark = pytest.mark.skipif(not corpus.MANIFEST.exists(), reason='corpus not downloaded')


@pytest.fixture(scope='module')
def contexts():
    table = load_table()
    full = vol.Context(table)
    origins = dist.fold_origins(full.inp, 5, 2024)[:30]
    last_date = full.inp.dates.iloc[int(origins[-1])]
    futures = pd.read_parquet(corpus.path('CL=F'))
    cut = vol.Context(table.iloc[:int(origins[-1]) + 1], futures[futures.index <= last_date])
    return full, cut, origins


@pytest.mark.parametrize('method', list(vol.METHODS))
def test_no_future_data(contexts, method):
    full, cut, origins = contexts
    params = vol.METHODS[method][1][-1]
    a = vol.walk_forward(full, method, params, 5, origins, 63)
    b = vol.walk_forward(cut, method, params, 5, origins, 63)
    assert np.allclose(a, b), method


def test_range_features_use_bars_up_to_the_origin(contexts):
    full, _, _ = contexts
    futures = pd.read_parquet(corpus.path('CL=F'))
    for p in (900, 2000, 3300):
        d = full.inp.dates.iloc[p]
        early = volatility_features(full.inp, futures[futures.index <= d])
        assert np.allclose(early.iloc[p], full.vol.iloc[p], equal_nan=True)


def test_targets_are_strictly_positive(contexts):
    full, _, _ = contexts
    for h in (1, 5, 20, 60):
        rv = full.rv[h].dropna()
        assert (rv > 0).all() and np.isfinite(vol.qlike(rv, rv.mean())).all()


def test_negative_price_windows_are_excluded(contexts):
    full, _, _ = contexts
    p = int(np.flatnonzero(full.inp.dates == pd.Timestamp('2020-04-20'))[0])
    rv = realized_variance(full.inp, 5)
    assert rv.iloc[p - 5:p + 1].isna().sum() >= 5 and np.isfinite(rv.iloc[p + 2])

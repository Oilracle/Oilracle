"""Study 2: event labels, derived probabilities and classifiers are correct and causal."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from m3_ml.common.data import load_table
from m3_ml.distribution import run as dist
from m3_ml.distribution.tails import ALL_TAUS
from m3_ml.kronos import corpus
from m3_ml.risk import events as ev
from m3_ml.risk import volatility as vol

pytestmark = pytest.mark.skipif(not corpus.MANIFEST.exists(), reason='corpus not downloaded')


def test_derived_probability_matches_a_normal_distribution():
    q = stats.norm.ppf(ALL_TAUS, scale=0.05)[None, :]
    p_down = ev.derived_probability(q, ALL_TAUS, 5, 'down')
    p_up = ev.derived_probability(q, ALL_TAUS, 5, 'up')
    exact = stats.norm.cdf(-0.07, scale=0.05)
    assert abs(p_down[0] - exact) < 0.005 and abs(p_up[0] - exact) < 0.005


@pytest.fixture(scope='module')
def contexts():
    table = load_table()
    full = vol.Context(table)
    origins = dist.fold_origins(full.inp, 20, 2024)[:30]
    last_date = full.inp.dates.iloc[int(origins[-1])]
    futures = pd.read_parquet(corpus.path('CL=F'))
    cut = vol.Context(table.iloc[:int(origins[-1]) + 1], futures[futures.index <= last_date])
    return full, cut, origins


@pytest.mark.parametrize('method', list(ev.MODELS))
def test_classifiers_use_no_future_data(contexts, method):
    full, cut, origins = contexts
    params = ev.MODELS[method][1][-1]
    for side in ev.SIDES:
        a = ev.walk_forward(full, method, params, 20, side, origins, 63)
        b = ev.walk_forward(cut, method, params, 20, side, origins, 63)
        assert np.allclose(a, b) and np.all((a > 0) & (a < 1)), (method, side)


def test_labels(contexts):
    full, _, _ = contexts
    y = full.inp.y[5]
    down, up = ev.labels(full.inp, 5, 'down'), ev.labels(full.inp, 5, 'up')
    assert ((down == 1) == (y < -0.07)).loc[y.notna()].all() and ((up == 1) == (y > 0.07)).loc[y.notna()].all()
    assert down.isna().equals(y.isna())

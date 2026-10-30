"""Route C mixture quantiles are exact and the model uses no future data
(the richest grid point is also covered by test_distribution_causality)."""
import numpy as np
import pytest
from scipy import stats

from m3_ml.common.data import load_table
from m3_ml.distribution import mixture, run
from m3_ml.distribution.data import load_inputs
from m3_ml.distribution.tails import ALL_TAUS


@pytest.mark.parametrize('core', ['har_gk', 'har_gk_implied'])
def test_empirical_mixture_uses_no_future_data(core):
    table = load_table()
    full = load_inputs(table)
    origins = run.fold_origins(full, 20, 2024)[:30]
    cut = load_inputs(table.iloc[:int(origins[-1]) + 1])
    params = {'core': core, 'alpha': 10, 'mixing': 'empirical'}
    a = run.walk_forward(full, 'c_mixture', params, 20, origins, 63, taus=ALL_TAUS)
    b = run.walk_forward(cut, 'c_mixture', params, 20, origins, 63, taus=ALL_TAUS)
    assert np.allclose(a, b)


def test_no_uncertainty_gives_the_normal_distribution():
    q = mixture.empirical_quantiles(np.zeros(50), ALL_TAUS)
    assert np.allclose(q, stats.norm.ppf(ALL_TAUS), atol=1e-6)
    q = mixture.lognormal_quantiles(np.array([1e-6]), ALL_TAUS)[0]
    assert np.allclose(q, stats.norm.ppf(ALL_TAUS), atol=1e-4)


def test_lognormal_mixture_matches_simulation():
    rng = np.random.default_rng(0)
    s = 0.8
    y = np.exp(s * rng.standard_normal(2_000_000) / 2) * rng.standard_normal(2_000_000)
    taus = np.array([0.005, 0.025, 0.1, 0.5, 0.9, 0.975, 0.995])
    exact = mixture.lognormal_quantiles(np.array([s]), taus)[0]
    assert np.allclose(exact, np.quantile(y, taus), rtol=0.01, atol=0.01)


def test_empirical_mixture_is_symmetric_and_fatter_tailed():
    e = np.random.default_rng(1).normal(0, 0.8, 3000)
    q = mixture.empirical_quantiles(e - e.mean(), ALL_TAUS)
    assert np.allclose(q, -q[::-1], atol=0.05)
    assert q[-1] > stats.norm.ppf(ALL_TAUS[-1])          # mixing fattens the tails

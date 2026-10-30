"""Every distribution method must give identical forecasts when all data after
the last forecast origin is removed, and route A's cross-fitting must purge
overlapping targets and embargo look-back features."""
import numpy as np
import pytest

from m3_ml.common.data import load_table
from m3_ml.distribution import run
from m3_ml.distribution.data import load_inputs
from m3_ml.distribution.quantiles import TAUS
from m3_ml.distribution.scale import EMBARGO, purged_folds
from m3_ml.distribution.tails import ALL_TAUS, MAIN


@pytest.fixture(scope='module')
def inputs():
    table = load_table()
    full = load_inputs(table)
    origins = run.fold_origins(full, 5, 2024)[:30]
    cut = load_inputs(table.iloc[:int(origins[-1]) + 1])
    return full, cut, origins


@pytest.mark.parametrize('taus', [TAUS, ALL_TAUS], ids=['main', 'all'])
@pytest.mark.parametrize('method', list(run.METHODS))
def test_no_future_data(inputs, method, taus):
    full, cut, origins = inputs
    params = run.METHODS[method][1][-1]                   # richest grid point (news features where offered)
    selected = {f'{m}/h5': {'params': run.METHODS[m][1][0], 'score': i}
                for i, m in enumerate(('a_ridge', 'a_lightgbm'))}
    a = run.walk_forward(full, method, params, 5, origins, 63, taus=taus, selected=selected)
    b = run.walk_forward(cut, method, params, 5, origins, 63, taus=taus, selected=selected)
    assert np.allclose(a, b, equal_nan=False), method


def test_main_levels_unchanged_by_tail_grid(inputs):
    full, _, origins = inputs
    for method in ('hs', 'ewma', 'a_ridge', 'b_forest'):
        params = run.METHODS[method][1][0]
        main = run.walk_forward(full, method, params, 5, origins, 63)
        both = run.walk_forward(full, method, params, 5, origins, 63, taus=ALL_TAUS)
        assert np.allclose(main, both[:, MAIN]), method


def test_purged_folds():
    train = np.arange(1000)
    for h in (1, 20, 60):
        for fit, block in purged_folds(train, h):
            a, b = block[0], block[-1]
            assert not np.any((fit > a - h) & (fit < b + h + EMBARGO))
            assert np.intersect1d(fit, block).size == 0

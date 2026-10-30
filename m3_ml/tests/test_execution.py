"""Ordered full-CPU jobs, exact checkpoint reuse, and real forecast equivalence."""
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pytest

from m3_ml.common.compute import cpu_threads
from m3_ml.common.execution import ValidationCache, run_jobs


def _toy(context, task):
    time.sleep((3 - task) * .01)
    return context + task


def _fail(context, task):
    raise ValueError('worker failure')


def _forecast(inp, task):
    from m3_ml.distribution.run import fitted_block, fold_origins
    from m3_ml.distribution.quantiles import TAUS
    method, params = task
    origins = fold_origins(inp, 5, 2024)[:4]
    return fitted_block(inp, method, params, 5, int(origins[0]), origins, TAUS)


def test_parallel_order_cache_and_resource_restoration(monkeypatch, tmp_path):
    monkeypatch.setenv('M3_CPU_THREADS', '2')
    monkeypatch.setenv('M3_OUTER_JOBS', '2')
    cache = ValidationCache('test', {'data': pd.Series([1., 2.])}, root=tmp_path)
    assert run_jobs(_toy, [0, 1, 2], 10, 'test', cache) == [10, 11, 12]
    assert os.environ['M3_CPU_THREADS'] == '2'
    assert run_jobs(_fail, [0, 1, 2], None, 'cached', cache) == [10, 11, 12]
    changed = ValidationCache('test', {'data': pd.Series([1., 3.])}, root=tmp_path)
    assert changed.path != cache.path
    assert not list(cache.path.glob('*.tmp'))


def test_worker_error_propagates(monkeypatch):
    monkeypatch.setenv('M3_CPU_THREADS', '2')
    monkeypatch.setenv('M3_OUTER_JOBS', '2')
    with pytest.raises(ValueError, match='worker failure'):
        run_jobs(_fail, [0, 1], None, 'failure')
    assert os.environ['M3_CPU_THREADS'] == '2'


def test_all_uses_available_cpu_count(monkeypatch):
    monkeypatch.setenv('M3_CPU_THREADS', 'all')
    assert cpu_threads() == (getattr(os, 'process_cpu_count', os.cpu_count)() or 1)


def test_real_forecasts_match_between_serial_and_parallel(monkeypatch):
    from m3_ml.distribution.data import load_inputs
    inp = load_inputs()
    jobs = [('hs', {'window': 'expanding'}),
            ('ewma', {'lam': .97, 'shape': 'empirical'}),
            ('a_ridge', {'alpha': 1., 'features': 'base_vol_news'}),
            ('garch', {'vol': 'garch', 'method': 'simulation'})]
    monkeypatch.setenv('M3_CPU_THREADS', '1')
    monkeypatch.setenv('M3_OUTER_JOBS', '1')
    # BLAS can select different numerical kernels at import time. Use a fresh
    # single worker for the sequential reference, matching parallel startup.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(name, '1')
    with ProcessPoolExecutor(max_workers=1) as pool:
        expected = [pool.submit(_forecast, inp, job).result() for job in jobs]
    monkeypatch.setenv('M3_CPU_THREADS', '2')
    monkeypatch.setenv('M3_OUTER_JOBS', '2')
    actual = run_jobs(_forecast, jobs, inp, 'parallel')
    for job, a, b in zip(jobs, actual, expected):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12, err_msg=str(job))

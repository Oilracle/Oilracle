"""Regression tests for two bugs found before the test run (see PLAN.md, bug log)."""
import numpy as np

from m3_ml.distribution import benchmarks, run
from m3_ml.distribution.data import load_inputs


def test_garch_simulation_is_reproducible():
    inp = load_inputs()
    origins = run.fold_origins(inp, 5, 2024)[:10]
    train = np.arange(int(origins[0]) - 5)
    for method in ('simulation', 'bootstrap'):
        a = benchmarks.predict_garch(inp, 5, int(origins[0]), train, origins, 'gjr', method)
        b = benchmarks.predict_garch(inp, 5, int(origins[0]), train, origins, 'gjr', method)
        assert np.allclose(a, b), method


def test_mature_validation_fold_is_scored():
    inp = load_inputs()
    for h in (1, 60):
        score = run.fold_score(inp, 'hs', {'window': 'expanding'}, h, 2024, 'crps', {})
        assert np.isfinite(score)

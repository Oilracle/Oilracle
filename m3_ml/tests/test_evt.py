"""The hand-written GJR recursion must reproduce arch's own simulation when both
use the fitted Student-t innovations."""
import numpy as np
from scipy import stats

from m3_ml.distribution import evt, run
from m3_ml.distribution.benchmarks import known_prefix
from m3_ml.distribution.data import load_inputs


def test_recursion_matches_arch_simulation():
    inp = load_inputs()
    origins = run.fold_origins(inp, 5, 2024)[:5]
    refit = int(origins[0])
    res = evt.fit_garch(inp, refit, int(origins[-1]), 'gjr')
    nu = res.params['nu']
    t_unit = lambda u: stats.t.ppf(u, nu) * np.sqrt((nu - 2) / nu)   # unit-variance Student-t
    mine = evt.simulate_sums(dict(res.params), evt.first_step_variance(res, inp, origins), 5, t_unit, sims=20_000)
    known = np.array([known_prefix(inp, o) for o in origins])
    first = int(known.min())
    fc = res.forecast(start=inp.dates.iloc[first], horizon=5, method='simulation', simulations=20_000,
                      random_state=np.random.RandomState(1), reindex=False)
    theirs = fc.simulations.values[known - first].sum(axis=2)
    levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    a, b = np.quantile(mine, levels, axis=1), np.quantile(theirs, levels, axis=1)
    sd = theirs.std(axis=1)
    # Monte Carlo error of a 5% quantile with 20,000 paths is about 0.015 sd
    assert np.all(np.abs(a - b) / sd < 0.05)
    assert np.allclose(mine.std(axis=1), sd, rtol=0.02)

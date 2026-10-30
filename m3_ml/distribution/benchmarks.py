"""Frozen distribution benchmarks with publication-aware quoted-row state.

Every predict function fits/calibrates once at refit, then filters only newly
published history at each origin. Missing suffix rows are forecast across, never
removed from the trading-row horizon.
"""
import warnings

import numpy as np
import pandas as pd
from arch.univariate import GARCH, StudentsT, ZeroMean
from scipy import stats

from m3_ml.common.data import label_available_dates
from .quantiles import TAUS

EWMA_INIT = 60
SIMULATIONS = 2000
SEED = 0


def known_prefix(inp, origin):
    """Last contiguous quotation row whose label is public at this origin.

    An unavailable interior observation blocks the prefix rather than silently
    compressing the calendar; the forecast bridges every row after that prefix.
    """
    if getattr(inp, '_public_prefix_dates', None) is None:
        available = label_available_dates(inp.table).to_numpy(dtype='datetime64[ns]')
        available = np.where(np.isnat(available), np.datetime64('2262-01-01'), available)
        inp._public_prefix_dates = np.maximum.accumulate(available)
    last = int(np.searchsorted(inp._public_prefix_dates,
                                np.datetime64(inp.dates.iloc[int(origin)]), side='right') - 1)
    return last


def anchor_offset(inp, origins, known):
    """Convert a distribution relative to last known raw quote to origin anchor."""
    return np.log(inp.price.iloc[known].to_numpy() / inp.anchor.iloc[origins].to_numpy())


def predict_hs(inp, h, refit, train, origins, window, taus=TAUS):
    y = inp.y[h].iloc[train].dropna().to_numpy()
    if window != 'expanding':
        y = y[-int(window):]
    return np.tile(np.quantile(y, taus), (len(origins), 1))


def ewma_sigma(returns, lam):
    """Daily EWMA state on the original quoted-row grid."""
    r = returns.to_numpy()
    out = np.full(len(r), np.nan)
    var = np.nanvar(r[1:EWMA_INIT + 1])
    for t in range(1, len(r)):
        var = lam * var + (1 - lam) * r[t] ** 2
        if t >= EWMA_INIT:
            out[t] = np.sqrt(var)
    return pd.Series(out, index=returns.index)


def predict_ewma(inp, h, refit, train, origins, lam, shape, taus=TAUS):
    sigma = ewma_sigma(inp.returns, lam)
    known = np.array([known_prefix(inp, o) for o in origins])
    if np.any(known < EWMA_INIT):
        raise ValueError('forecast origin lacks EWMA initialization history')
    scale = sigma.iloc[known].to_numpy() * np.sqrt(np.asarray(origins) + h - known)
    offset = anchor_offset(inp, origins, known)
    if shape == 'normal':
        z_q = stats.norm.ppf(taus)
    else:
        kt = np.array([known_prefix(inp, t) for t in train])
        keep = kt >= EWMA_INIT
        calibration, kt = np.asarray(train)[keep], kt[keep]
        st = sigma.iloc[kt].to_numpy() * np.sqrt(calibration + h - kt)
        z = (inp.y[h].iloc[calibration].to_numpy() - anchor_offset(inp, calibration, kt)) / st
        z_q = np.quantile(z[np.isfinite(z)], taus)
    return offset[:, None] + scale[:, None] * z_q[None, :]


def fit_garch_state(inp, refit, last, vol):
    """Likelihood fits only the public prefix; later public rows filter that fit."""
    fitted_end, state_end = known_prefix(inp, refit), known_prefix(inp, last)
    if fitted_end < EWMA_INIT:
        raise ValueError('not enough contiguous published history to fit GARCH')
    r = 100 * inp.returns.iloc[1:state_end + 1]
    r.index = inp.dates.iloc[1:state_end + 1].to_numpy()
    model = ZeroMean(r, volatility=GARCH(p=1, o=1 if vol == 'gjr' else 0, q=1),
                     distribution=StudentsT(seed=np.random.default_rng(SEED)))
    after_fit = inp.dates.iloc[fitted_end + 1] if fitted_end < state_end else None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return model.fit(last_obs=after_fit, disp='off', show_warning=False)


def first_variance(res, inp, origins):
    known = np.array([known_prefix(inp, o) for o in origins])
    first = int(known.min())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fc = res.forecast(start=inp.dates.iloc[first], horizon=1, method='analytic', reindex=False)
    return fc.variance.to_numpy()[known - first, 0], known


def public_simulation(inp, h, origins, res, draw, sims, taus):
    """Frozen GJR recursion, including all unpublished rows before origin+h."""
    from .evt import simulate_sums
    var1, known = first_variance(res, inp, origins)
    lengths = np.asarray(origins) + h - known
    out = np.empty((len(origins), len(taus)))
    # Seed each origin independently: adding later forecast rows must not alter
    # the fixed-origin Monte Carlo distribution through a changed array shape.
    for i, (origin, length) in enumerate(zip(origins, lengths)):
        paths = simulate_sums(dict(res.params), var1[i:i + 1], int(length), draw,
                              sims=sims, seed=SEED + int(origin)) / 100
        out[i] = np.quantile(paths[0], taus)
    return out + anchor_offset(inp, origins, known)[:, None]


def predict_garch(inp, h, refit, train, origins, vol, method, taus=TAUS, sims=SIMULATIONS):
    res = fit_garch_state(inp, refit, int(origins[-1]), vol)
    if method == 'simulation':
        nu = res.params['nu']
        draw = lambda u: stats.t.ppf(u, nu) * np.sqrt((nu - 2) / nu)
    else:
        innovations = res.std_resid.dropna().to_numpy()
        draw = lambda u: np.quantile(innovations, u)
    return public_simulation(inp, h, origins, res, draw, sims, taus)


BENCHMARKS = {
    'hs': (predict_hs, [{'window': w} for w in ('expanding', 750, 1500)]),
    'ewma': (predict_ewma, [{'lam': l, 'shape': s} for l in (0.94, 0.97, 0.99) for s in ('normal', 'empirical')]),
    'garch': (predict_garch, [{'vol': v, 'method': m} for v in ('garch', 'gjr') for m in ('simulation', 'bootstrap')]),
}

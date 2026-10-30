"""Quantile-forecast scoring: pinball loss, CRPS, interval coverage, PIT and tests.

A forecast is a matrix Q (n origins x 23 levels) of quantiles of the h-day log
return, columns ordered as TAUS; y is the realized log return per origin.
"""
import numpy as np
import statsmodels.api as sm
from scipy import stats

TAUS = np.round(np.concatenate([[0.01, 0.025], np.arange(0.05, 0.951, 0.05), [0.975, 0.99]]), 3)
LEVELS = (0.5, 0.8, 0.9, 0.95)  # central prediction intervals


def column(tau):
    return f'q_{tau:.3f}'


QUANTILE_COLUMNS = [column(t) for t in TAUS]


def rearrange(Q):
    """Sort each row so quantiles never cross; this never increases pinball loss."""
    return np.sort(np.asarray(Q, dtype=float), axis=1)


def pinball(Q, y):
    """Pinball loss per origin and level, shape (n, 23)."""
    Q, y = np.asarray(Q, dtype=float), np.asarray(y, dtype=float)[:, None]
    diff = y - Q
    return np.maximum(TAUS * diff, (TAUS - 1) * diff)


def crps(Q, y):
    """Quantile approximation of the CRPS per origin: (2 / K) * sum of pinball losses."""
    return 2 * pinball(Q, y).mean(axis=1)


def interval(Q, level):
    """Lower and upper bound of the central `level` prediction interval."""
    lo, hi = (1 - level) / 2, (1 + level) / 2
    i, j = int(np.argmin(np.abs(TAUS - lo))), int(np.argmin(np.abs(TAUS - hi)))
    assert np.isclose(TAUS[i], lo) and np.isclose(TAUS[j], hi), f'level {level} not on the grid'
    Q = np.asarray(Q, dtype=float)
    return Q[:, i], Q[:, j]


def covered(Q, y, level):
    lo, hi = interval(Q, level)
    y = np.asarray(y, dtype=float)
    return (y >= lo) & (y <= hi)


def pit(Q, y):
    """Probability integral transform by linear interpolation of the quantile
    function; values beyond the outer quantiles are placed at 0.005 / 0.995."""
    Q, y = np.asarray(Q, dtype=float), np.asarray(y, dtype=float)
    return np.array([np.interp(v, q, TAUS, left=0.005, right=0.995) for q, v in zip(Q, y)])


def hac_mean_pvalue(x, lags):
    """Two-sided p-value for mean(x) = 0 with Newey-West standard errors."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= max(int(lags) + 2, 3):
        return np.nan  # insufficient observations for this HAC lag specification
    fit = sm.OLS(x, np.ones(len(x))).fit(cov_type='HAC', cov_kwds={'maxlags': max(int(lags), 1)})
    return float(fit.pvalues[0])


def kupiec_pvalue(hits, level):
    """Unconditional coverage LR test (valid for non-overlapping forecasts, h = 1)."""
    hits = np.asarray(hits, dtype=bool)
    n, x = len(hits), int((~hits).sum())  # x = violations
    p = 1 - level
    if x in (0, n):
        return float('nan')
    phat = x / n
    lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - phat) - x * np.log(phat))
    return float(stats.chi2.sf(lr, 1))


def christoffersen_pvalue(hits):
    """Independence LR test of violations (first-order Markov), h = 1 only."""
    v = (~np.asarray(hits, dtype=bool)).astype(int)
    a, b = v[:-1], v[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    if min(n00 + n01, n10 + n11) == 0 or (n01 + n11) in (0, len(a)):
        return float('nan')
    p01, p11, p = n01 / (n00 + n01), n11 / (n10 + n11), (n01 + n11) / len(a)

    def ll(prob, k0, k1):
        return (k0 * np.log(1 - prob) if k0 else 0) + (k1 * np.log(prob) if k1 else 0)

    lr = -2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    return float(stats.chi2.sf(lr, 1))

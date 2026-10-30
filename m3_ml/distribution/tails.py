"""Tail risk from quantile forecasts: VaR, ES, FZ0 loss, Acerbi-Szekely Z2, GPD
tails, and direction metrics (PLAN.md addendum).

Forecasts at ALL_TAUS (63 levels) = the 23 main levels plus 40 tail midpoints.
The main levels keep their own rearrangement (so CRPS is unchanged); ES uses
the full grid sorted per origin. Returns are h-day log returns; "lower" is the
long position's loss tail, "upper" the short position's.
"""
import numpy as np
from scipy import stats

from .quantiles import TAUS

STEP = 0.0025
TAIL_LOWER = np.round((np.arange(20) + 0.5) * STEP, 5)      # 0.00125 ... 0.04875
TAIL_UPPER = np.round(1 - TAIL_LOWER[::-1], 5)               # 0.95125 ... 0.99875
TAIL_TAUS = np.concatenate([TAIL_LOWER, TAIL_UPPER])
ALL_TAUS = np.sort(np.concatenate([TAUS, TAIL_TAUS]))
MAIN = np.isin(ALL_TAUS, TAUS)
ALPHAS = (0.01, 0.025, 0.05)
SIDES = ('lower', 'upper')


def tail_column(tau):
    return f't_{tau:.5f}'


ALL_COLUMNS = [f'q_{t:.3f}' if m else tail_column(t) for t, m in zip(ALL_TAUS, MAIN)]


def split(Q_all):
    """Main quantiles (sorted among themselves) and the full grid sorted per origin."""
    Q_all = np.asarray(Q_all, dtype=float)
    return np.sort(Q_all[:, MAIN], axis=1), np.sort(Q_all, axis=1)


def var_es(Q_main, Q_full, alpha, side):
    """VaR quantile v and ES e in return space for one tail.

    lower: v = q_alpha, e = mean of q(u) for tail midpoints u < alpha (both < 0
    normally). upper: mirrored onto -y, so v = -q_{1-alpha}, e = -mean q(u), u > 1 - alpha.
    Losses for reporting are -v and -e."""
    if side == 'lower':
        v = Q_main[:, int(np.argmin(np.abs(TAUS - alpha)))]
        e = Q_full[:, np.isin(ALL_TAUS, TAIL_LOWER[TAIL_LOWER < alpha])].mean(axis=1)
    else:
        v = -Q_main[:, int(np.argmin(np.abs(TAUS - (1 - alpha))))]
        e = -Q_full[:, np.isin(ALL_TAUS, TAIL_UPPER[TAIL_UPPER > 1 - alpha])].mean(axis=1)
    return v, e


def side_returns(y, side):
    return np.asarray(y, dtype=float) if side == 'lower' else -np.asarray(y, dtype=float)


def fz0(y, v, e, alpha):
    """FZ0 loss of Patton, Ziegel & Chen (2019) for (VaR, ES) in return space; needs e < 0."""
    y, v, e = (np.asarray(a, dtype=float) for a in (y, v, e))
    out = -((y <= v) * (v - y)) / (alpha * e) + v / e + np.log(-np.where(e < 0, e, np.nan)) - 1
    return np.where(e < 0, out, np.nan)


def mean_fz0(Q_all, y):
    """Average FZ0 over the 3 levels x 2 tails, per origin."""
    Q_main, Q_full = split(Q_all)
    losses = [fz0(side_returns(y, s), *var_es(Q_main, Q_full, a, s), a) for a in ALPHAS for s in SIDES]
    return np.mean(losses, axis=0)


def inverse_cdf(Q_full, u):
    """Sample values for probabilities u (origins x draws) from each origin's quantile
    function by linear interpolation; beyond the outer levels the end value is used."""
    return np.vstack([np.interp(row_u, ALL_TAUS, q) for q, row_u in zip(Q_full, u)])


def z2(y, v, e, alpha):
    """Acerbi-Szekely Z2 in return space (v, e < 0): sum(X I / (T alpha |e|)) + 1."""
    y, v, e = (np.asarray(a, dtype=float) for a in (y, v, e))
    return float(np.sum(y * (y <= v) / (len(y) * alpha * -e)) + 1)


def z2_test(Q_all, y, alpha, side, draws=10_000, seed=0):
    """Z2 statistic and its p-value simulated from the forecast distributions
    (independence across origins assumed: formal test only for h = 1)."""
    Q_main, Q_full = split(Q_all)
    v, e = var_es(Q_main, Q_full, alpha, side)
    observed = z2(side_returns(y, side), v, e, alpha)
    rng = np.random.default_rng(seed)
    u = rng.random((len(v), draws))
    sims = inverse_cdf(Q_full, u)
    sims = sims if side == 'lower' else -sims
    stat = (sims * (sims <= v[:, None]) / (len(v) * alpha * -e[:, None])).sum(axis=0) + 1
    return observed, float(np.mean(stat <= observed))


def prob_up(Q_main):
    """P(return > 0) from the main quantiles by linear interpolation of the CDF."""
    return np.array([1 - np.interp(0.0, q, TAUS, left=0.0, right=1.0) for q in np.asarray(Q_main)])


def direction_metrics(p_up, y):
    y = np.asarray(y, dtype=float)
    up, pred = y > 0, p_up > 0.5
    out = {'brier': float(np.mean((p_up - up) ** 2)), 'accuracy': float(np.mean(pred == up))}
    for name, truth, guess in (('up', up, pred), ('down', ~up, ~pred)):
        tp = np.sum(truth & guess)
        precision = tp / guess.sum() if guess.sum() else np.nan
        recall = tp / truth.sum() if truth.sum() else np.nan
        out[f'precision_{name}'], out[f'recall_{name}'] = precision, recall
        out[f'f1_{name}'] = 2 * precision * recall / (precision + recall) if precision + recall else np.nan
    return out


# ---- Generalized Pareto tails (EVT) ----

def fit_gpd_tails(z, fraction):
    """Fit a GPD to each tail of the sample z beyond its `fraction` quantile.
    Returns parameters of both tails as losses (lower tail uses -z)."""
    z = np.asarray(z, dtype=float)
    tails = {}
    for side, x in (('lower', -z), ('upper', z)):
        threshold = np.quantile(x, 1 - fraction)
        excess = x[x > threshold] - threshold
        shape, _, scale = stats.genpareto.fit(excess, floc=0)
        tails[side] = (threshold, shape, scale)
    return {'fraction': fraction, 'body': np.sort(z), **tails}


def gpd_quantile(p_tail, fraction, threshold, shape, scale):
    """Loss quantile at tail probability p_tail (< fraction) of a GPD tail."""
    ratio = np.asarray(p_tail, dtype=float) / fraction
    if abs(shape) < 1e-8:
        return threshold - scale * np.log(ratio)
    return threshold + scale / shape * (ratio ** -shape - 1)


def semiparametric_ppf(model, u):
    """Quantiles of the 'empirical body + GPD tails' distribution at probabilities u."""
    u = np.asarray(u, dtype=float)
    f = model['fraction']
    body = model['body']
    out = np.quantile(body, np.clip(u, f, 1 - f))
    lo, hi = u < f, u > 1 - f
    out = np.where(lo, -gpd_quantile(np.where(lo, u, f), f, *model['lower']), out)
    out = np.where(hi, gpd_quantile(np.where(hi, 1 - u, f), f, *model['upper']), out)
    return out

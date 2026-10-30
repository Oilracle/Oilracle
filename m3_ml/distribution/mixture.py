"""Route C (proposed after the test, so not a blind test; see PLAN.md and
doc/m3_ml/README.md): variance-mixture distribution. No look-ahead.

Given the realized variance, h-day returns are close to normal, and the fat
tails of the forecast distribution come from the error of the variance
forecast. So the forecast distribution is a normal mixture over the
predictive distribution of the h-day realized variance v:
    y | F_t ~ integral N(0, v) p(v | F_t) dv,   ln v = core forecast + bias + error.
The error is either the empirical cross-fitted log-variance errors
(unconditional) or N(0, s_t^2) with a state-dependent s_t (conditional).
Quantiles are exact (mixture CDF inverted by bisection), no Monte Carlo.
"""
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline

from m3_ml.common.preprocessing import make_preprocessor
from m3_ml.risk.features import FEATURE_SETS, FLOOR, volatility_features

from .quantiles import TAUS
from .scale import purged_folds

HORIZONS = (5, 20, 60)
LOG_CHI2_MEAN = -1.2704                      # E[ln chi^2_1]
S_BOUNDS = (0.2, 2.0)
NODES, WEIGHTS = np.polynomial.hermite_e.hermegauss(40)
WEIGHTS = WEIGHTS / WEIGHTS.sum()            # expectations under N(0, 1)


def inputs(inp):
    """Volatility features and log realized-variance targets for these inputs,
    cached on the object itself (an id-keyed cache could hand a new object the
    features of a garbage-collected one)."""
    if getattr(inp, '_mixture_inputs', None) is None:
        raw_returns = np.log(inp.price.where(inp.price > 0)).diff()
        log_rv = pd.DataFrame({h: np.log(raw_returns.pow(2).rolling(h).sum().shift(-h) + FLOOR)
                               for h in range(1, 61)})
        inp._mixture_inputs = (volatility_features(inp), log_rv)
    return inp._mixture_inputs


def usable_mask(inp, h):
    vol, log_rv = inputs(inp)
    return (log_rv[h].notna() & vol[FEATURE_SETS['cc_gk_implied']].notna().all(axis=1)).to_numpy()


def core_predict(core, alpha, vol, y, fit, pred):
    """ln RV forecast at `pred` positions from a core fitted on `fit` positions."""
    def ridge(columns):
        model = make_pipeline(make_preprocessor('scaled'), Ridge(alpha=alpha))
        return model.fit(vol.iloc[fit][columns], y.iloc[fit]).predict(vol.iloc[pred][columns])

    if core == 'har_gk':
        return ridge(FEATURE_SETS['cc_gk'])
    if core == 'har_gk_implied':
        return ridge(FEATURE_SETS['cc_gk_implied'])
    ovx = LinearRegression().fit(vol.iloc[fit][['ovx2']], y.iloc[fit]).predict(vol.iloc[pred][['ovx2']])
    return 0.5 * ridge(FEATURE_SETS['cc_gk']) + 0.5 * ovx


def bisect(cdf, taus, shape, lo=-200.0, hi=200.0, steps=80):
    """Solve cdf(q) = tau elementwise for a nondecreasing cdf (q in standard-deviation units)."""
    a, b = np.full(shape, lo), np.full(shape, hi)
    target = np.broadcast_to(taus, shape)
    for _ in range(steps):
        mid = (a + b) / 2
        below = cdf(mid) < target
        a, b = np.where(below, mid, a), np.where(below, b, mid)
    return (a + b) / 2


def empirical_quantiles(errors, taus):
    """Standardized quantiles Q with mean_j Phi(Q / exp(e_j / 2)) = tau."""
    c = np.exp(np.asarray(errors) / 2)
    cdf = lambda q: ndtr(q[:, None] / c[None, :]).mean(axis=1)
    return bisect(cdf, np.asarray(taus), (len(taus),))


def lognormal_quantiles(s, taus):
    """Standardized quantiles (origins x taus) of a normal with ln-variance ~ N(0, s^2)."""
    s, taus = np.asarray(s, dtype=float), np.asarray(taus, dtype=float)
    scale = np.exp(s[:, None] * NODES[None, :] / 2)                       # origins x nodes
    cdf = lambda q: (ndtr(q[:, :, None] / scale[:, None, :]) * WEIGHTS).sum(axis=2)
    return bisect(cdf, taus[None, :], (len(s), len(taus)))


def predict_c_mixture(inp, h, refit, train, origins, core, alpha, mixing, taus=TAUS):
    vol, log_rv = inputs(inp)
    y = log_rv[h]
    train = np.sort(train)
    errors = np.empty(len(train))
    where = {p: i for i, p in enumerate(train)}
    for fit, block in purged_folds(train, h):
        errors[[where[p] for p in block]] = y.iloc[block].to_numpy() - core_predict(core, alpha, vol, y, fit, block)
    bias = errors.mean()
    centre = core_predict(core, alpha, vol, y, train, origins) + bias       # ln v location
    e = errors - bias
    if mixing == 'empirical':
        return np.exp(centre / 2)[:, None] * empirical_quantiles(e, taus)[None, :]
    X = vol[FEATURE_SETS['cc_gk_implied']]
    disp = make_pipeline(make_preprocessor('scaled'), Ridge(alpha=alpha)).fit(X.iloc[train], np.log(e ** 2 + 1e-12))
    s = np.clip(np.sqrt(np.exp(disp.predict(X.iloc[origins]) - LOG_CHI2_MEAN)), *S_BOUNDS)
    return np.exp(centre / 2)[:, None] * lognormal_quantiles(s, taus)


MIXTURE_MODELS = {
    'c_mixture': (predict_c_mixture, [{'core': c, 'alpha': a, 'mixing': m}
                                      for c in ('har_gk', 'har_gk_implied', 'combo')
                                      for a in (1, 10, 100) for m in ('empirical', 'conditional')]),
}

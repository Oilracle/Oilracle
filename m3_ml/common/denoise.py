"""Causal Kalman denoising of noisy weekly EIA series.

Local level model, one per series, on the weekly report values:
    observed_k = level_k + noise_k,    noise_k ~ N(0, r)
    level_k    = level_{k-1} + shock_k, shock_k ~ N(0, q)
The two variances are estimated by maximum likelihood on reports up to
FIT_END only, then frozen. The filter uses only reports up to k to estimate
level_k (filtered state, never the smoothed state), so it is causal.

The noisy-series rule and its variances are calibrated on the initial training
period only, then held fixed during validation and the final test.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .protocol import TRAIN_END as FIT_END


def kalman_level(values, q, r):
    """Filtered level of a local level model with fixed variances."""
    level, p = values[0], r
    out = np.empty(len(values))
    for k, y in enumerate(values):
        p += q
        gain = p / (p + r)
        level += gain * (y - level)
        p *= 1 - gain
        out[k] = level
    return out


def fit_local_level(values):
    """Maximum-likelihood (q, r) of a local level model."""
    fit = sm.tsa.UnobservedComponents(values, level='local level').fit(disp=False)
    params = dict(zip(fit.model.param_names, fit.params))
    return params['sigma2.level'], params['sigma2.irregular']


def noisy_series(reports, report_dates):
    """Series whose weekly changes are significantly negatively autocorrelated
    before FIT_END."""
    early = reports[report_dates <= FIT_END].diff().dropna()
    threshold = -2 / np.sqrt(len(early))
    autocorr = early.apply(lambda s: s.autocorr(1))
    return list(autocorr[autocorr < threshold].index), autocorr, threshold


def denoise_reports(reports, report_dates):
    """Replace the noisy series in a reports table (one row per weekly report)
    by their causal Kalman level. Returns the new table and the fitted settings."""
    names, autocorr, threshold = noisy_series(reports, report_dates)
    out, settings = reports.copy(), {}
    early = report_dates <= FIT_END
    for name in names:
        q, r = fit_local_level(reports.loc[early, name].to_numpy())
        out[name] = kalman_level(reports[name].to_numpy(dtype=float), q, r)
        settings[name] = {'autocorr': round(float(autocorr[name]), 3), 'signal_to_noise': q / r,
                          'steady_state_gain': steady_gain(q, r)}
    return out, settings, threshold


def steady_gain(q, r):
    """Long-run weight on each new observation (1 = no smoothing)."""
    p = (q + np.sqrt(q * q + 4 * q * r)) / 2
    return float(p / (p + r))

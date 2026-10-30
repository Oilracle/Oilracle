"""Publication-aware seasonal ARIMA baseline with a frozen final parameter fit.

Only the contiguous quoted history available at an origin enters its state.
Final coefficients are estimated by FIT_END; subsequent releases update only
the innovation. The gap between latest known quote and target remains in the
calendar sum rather than silently renumbering the forecast horizon.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from m3_ml.common.data import label_available_dates, log_price, observable_price, safe_log_price
from m3_ml.common.protocol import FIT_END, TRAIN_END

REFIT = 21
MIN_HISTORY = 252


def calendar_design(dates):
    """Intercept, February–December and Tuesday–Friday effects."""
    month, weekday = dates.dt.month.to_numpy(), dates.dt.dayofweek.to_numpy()
    x = np.zeros((len(dates), 16))
    x[:, 0] = 1.0
    for m in range(2, 13):
        x[:, m - 1] = month == m
    for w in range(1, 5):
        x[:, 11 + w] = weekday == w
    return x


def _fit_season(returns, design, end):
    rows = np.flatnonzero(np.isfinite(returns[:end + 1]))
    rows = rows[rows >= 1]
    if len(rows) < MIN_HISTORY:
        return None
    return np.linalg.lstsq(design[rows], returns[rows], rcond=None)[0]


def _fit_arima(adjusted):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = ARIMA(adjusted, order=(0, 1, 1), trend='n').fit()
    return float(np.asarray(result.maparams)[0]), float(np.asarray(result.resid)[-1])


def build_base(table, refit_every=REFIT, fit_end=FIT_END, horizons=tuple(range(1, 61))):
    """Causal historical fits, then frozen coefficients and released-state updates.

    Dates beyond the observed table are a tagged weekday scenario, never
    invented realized quote dates. Their forecasts remain unscored.
    """
    dates = pd.to_datetime(table['date']).reset_index(drop=True)
    raw = table['target_wti'].astype(float).reset_index(drop=True)
    filled, genuine = safe_log_price(raw).to_numpy(), log_price(raw).to_numpy()
    anchor = safe_log_price(observable_price(table)).to_numpy()
    returns = np.diff(filled, prepend=np.nan)
    returns[~np.isfinite(genuine) | ~np.isfinite(np.roll(genuine, 1))] = np.nan
    returns[0] = np.nan
    n = len(table)
    future = pd.Series(pd.bdate_range(dates.iloc[-1] + pd.offsets.BDay(), periods=max(horizons)))
    calendar = pd.concat([dates, future], ignore_index=True)
    design = calendar_design(calendar)
    available = pd.to_datetime(label_available_dates(table))
    last_known = np.full(n, -1, dtype=int)
    for t, date in enumerate(dates):
        known = np.flatnonzero(np.logical_and.accumulate(available.le(date).to_numpy()))
        if len(known):
            last_known[t] = int(known[-1])
    cutoff = np.flatnonzero(dates.le(pd.Timestamp(fit_end)).to_numpy())
    final = int(cutoff[-1]) if len(cutoff) else -1
    refits = sorted(set(range(MIN_HISTORY, final + 1, refit_every)) | ({final} if final >= MIN_HISTORY else set()))
    initial = np.flatnonzero(dates.le(pd.Timestamp(TRAIN_END)).to_numpy())
    if len(initial) and MIN_HISTORY <= initial[-1] <= final:
        refits = sorted(set(refits) | {int(initial[-1])})
    ts = np.full(n, np.nan)
    seasonal = {h: np.full(n, np.nan) for h in horizons}
    for i, refit in enumerate(refits):
        known = last_known[refit]
        beta = _fit_season(returns, design, known)
        if beta is None:
            continue
        daily = design @ beta
        adjusted = np.empty(known + 1)
        adjusted[0] = filled[0]
        for s in range(1, known + 1):
            adjusted[s] = adjusted[s - 1] + (returns[s] - daily[s] if np.isfinite(returns[s]) else 0.0)
        theta, error_at_fit = _fit_arima(adjusted)
        stop = refits[i + 1] if i + 1 < len(refits) else n
        for t in range(refit, stop):
            observed = last_known[t]
            error = error_at_fit
            for s in range(known + 1, observed + 1):
                error = (returns[s] - daily[s] if np.isfinite(returns[s]) else 0.0) - theta * error
            # ARIMA(0,1,1) cumulative change is theta * latest innovation for
            # every positive horizon, including the unpublished origin gap.
            offset = filled[observed] - anchor[t]
            ts[t] = offset + theta * error
            for h in horizons:
                seasonal[h][t] = daily[observed + 1:t + h + 1].sum()
    out = pd.DataFrame({'ts': ts}, index=table.index)
    for h in horizons:
        out[f'seas_{h}'] = seasonal[h]
        out[f'base_{h}'] = ts + seasonal[h]
    out['calendar_assumption'] = 'observed_quote_dates_then_weekday_scenario'
    return out

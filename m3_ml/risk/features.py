"""Volatility features and targets for the risk studies (ml/risk/PLAN.md, study 1).

All features use information up to the close of day t:
- close-to-close: log of the mean squared daily WTI spot log return over 1, 5, 20, 60 days;
- range-based: log of the mean Garman-Klass variance of WTI front-month futures
  (CL=F, Yahoo) over its last 1, 5, 20 bars dated on or before t; bars with
  inconsistent or non-positive OHLC are skipped;
- implied: log OVX^2 and log VIX^2 (same publication lag as the shared features).
"""
import numpy as np
import pandas as pd

from m3_ml.common.data import HORIZONS, label_available_dates
from m3_ml.kronos.corpus import path as corpus_path

FLOOR = 1e-8                 # variance floor before taking logs
NEGATIVE_PRICE_DAYS = ('2020-04-20', '2020-04-21')
CC = ['cc_1d', 'cc_5d', 'cc_20d', 'cc_60d']
GK = ['gk_1d', 'gk_5d', 'gk_20d']
IMPLIED = ['ovx2', 'vix2']
FEATURE_SETS = {'cc': CC, 'cc_gk': CC + GK, 'cc_gk_implied': CC + GK + IMPLIED}


def garman_klass(frame):
    o, h, l, c = (frame[k] for k in ('open', 'high', 'low', 'close'))
    ok = (frame[['open', 'high', 'low', 'close']] > 0).all(axis=1) & (h >= l) & (c <= h) & (c >= l) & (o <= h) & (o >= l)
    gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    return gk.where(ok).dropna().clip(lower=0)


def volatility_features(inp, futures=None):
    """One row per wti_daily position. `futures` (CL=F OHLC) defaults to the corpus file."""
    available = pd.to_datetime(label_available_dates(inp.table))
    raw = inp.returns ** 2
    out = {f'cc_{k}d': np.full(len(raw), np.nan) for k in (1, 5, 20, 60)}
    for p, date in enumerate(inp.dates):
        known = np.logical_and.accumulate(available.le(date).to_numpy())
        history = raw[known]
        for k in (1, 5, 20, 60):
            out[f'cc_{k}d'][p] = np.log(history.iloc[-k:].mean() + FLOOR) if history.notna().sum() >= k else np.nan
    gk = garman_klass(pd.read_parquet(corpus_path('CL=F')) if futures is None else futures)
    dates = pd.DatetimeIndex(inp.dates)
    for k in (1, 5, 20):
        series = np.log(gk.rolling(k).mean() + FLOOR)
        out[f'gk_{k}d'] = series.reindex(series.index.union(dates)).ffill().reindex(dates).to_numpy()
    out['ovx2'] = np.log((inp.table['oil_implied_volatility'] / 100) ** 2)
    out['vix2'] = np.log((inp.table['vix_close'] / 100) ** 2)
    return pd.DataFrame(out, index=inp.dates.index)


def realized_variance(inp, h):
    """RV_{t,h} = sum of the next h squared daily log returns; NaN for origins
    whose window contains the 2020-04-20 / 04-21 negative-price days, and where
    RV is exactly 0 (stale quotes: only at h = 1, 0.5% of days; QLIKE and the
    log target are undefined there)."""
    r2 = inp.returns ** 2
    rv = r2.rolling(h).sum().shift(-h)
    rv = rv.where(rv != 0)
    dates = inp.dates.reset_index(drop=True)
    for day in NEGATIVE_PRICE_DAYS:
        hit = np.flatnonzero(dates == pd.Timestamp(day))
        if len(hit):                               # absent when the table ends earlier
            p = int(hit[0])
            rv.iloc[max(p - h, 0):p] = np.nan      # origins t with t < p <= t + h
    return rv


def targets(inp):
    return pd.DataFrame({h: realized_variance(inp, h) for h in HORIZONS}, index=inp.dates.index)

"""Inputs for the distribution forecasts: returns, targets and feature sets.

All positions refer to quoted rows, not publication events. Realized raw quote
returns are retained for targets; observed_returns and anchor are separate
published-price inputs. A raw quote is not assumed known on its quotation day.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from m3_ml.common.data import HORIZONS, PRICE, load_table, observable_price, safe_log_price, targets
from m3_ml.common.features import build_features
from m3_ml.common.news import news_features

VOL_COLUMNS = ['rv_1d', 'rv_5d', 'rv_20d', 'rv_60d', 'ovx', 'vix']
FEATURE_SETS = ('vol', 'base_vol', 'base_vol_news')
VOL_FLOOR = 1e-6  # daily variance floor for the log volatility target (0.1% daily vol)


def volatility_block(table, returns):
    """Realized volatility (root mean square of daily log returns) over the last
    1, 5, 20 and 60 days, plus OVX and VIX levels. Causal: uses r up to r_t."""
    out = {'rv_1d': returns.abs()}
    for k in (5, 20, 60):
        out[f'rv_{k}d'] = np.sqrt((returns ** 2).rolling(k).mean())
    out['ovx'] = table['oil_implied_volatility']
    out['vix'] = table['vix_close']
    return pd.DataFrame(out)


def log_volatility(frame):
    """Log of the volatility columns (for linear models of log volatility)."""
    out = frame.copy()
    cols = [c for c in VOL_COLUMNS if c in out]
    out[cols] = np.log(out[cols].clip(lower=np.sqrt(VOL_FLOOR)))
    return out


def volatility_target(returns, h):
    """log(sum of the next h squared daily returns + floor), aligned at the origin."""
    future = (returns ** 2).rolling(h).sum().shift(-h)
    return np.log(future + VOL_FLOOR * h)


@dataclass
class Inputs:
    table: pd.DataFrame
    dates: pd.Series
    price: pd.Series
    returns: pd.Series          # r_t, NaN at the first row
    y: pd.DataFrame             # h-day log return targets, columns = HORIZONS
    vol_target: pd.DataFrame    # log realized-variance targets, columns = HORIZONS
    features: dict              # name -> DataFrame
    complete: pd.Series         # True where every feature set is complete (evaluation origins)
    anchor: pd.Series           # published origin price; never the hidden target
    observed_returns: pd.Series # published-price changes, for causal features only


def load_inputs(table=None):
    table = load_table() if table is None else table.reset_index(drop=True)
    returns = safe_log_price(table[PRICE]).diff()
    anchor = observable_price(table)
    observed_returns = safe_log_price(anchor).diff()
    vol = volatility_block(table, observed_returns)
    base = build_features(table)
    news = news_features(table['date'])
    features = {'vol': vol, 'base_vol': pd.concat([base, vol], axis=1),
                'base_vol_news': pd.concat([base, vol, news], axis=1)}
    complete = features['base_vol_news'].notna().all(axis=1)
    return Inputs(table=table, dates=table['date'], price=table[PRICE], returns=returns,
                  y=targets(table), vol_target=pd.DataFrame({h: volatility_target(returns, h) for h in HORIZONS}),
                  features=features, complete=complete, anchor=anchor, observed_returns=observed_returns)

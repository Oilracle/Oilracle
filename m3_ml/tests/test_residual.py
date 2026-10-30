"""The time-series-plus-seasonality base at day t must not use prices after t."""
import numpy as np

from m3_ml.common.data import load_table
from m3_ml.residual.base import build_base


def test_base_forecast_ignores_future_prices():
    table = load_table()
    full = build_base(table, refit_every=252)
    cut = 1500
    truncated = table.copy()
    truncated.loc[cut + 1:, 'target_wti'] = np.nan
    short = build_base(truncated, refit_every=252)
    for column in ('ts', 'base_1', 'base_5', 'base_20'):
        left = full.loc[:cut, column].to_numpy()
        right = short.loc[:cut, column].to_numpy()
        assert np.allclose(left, right, equal_nan=True)

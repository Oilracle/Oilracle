"""Shared feature engineering for the ML models.

Every feature on row t uses only that row's as-of aligned, published columns.
The raw target_wti quote is never used as a same-origin input; WTI momentum
uses the observable WTI anchor. Levels are converted to changes, spreads and
year-over-year differences.
"""
import numpy as np
import pandas as pd

from .data import observable_price, safe_log_price

# Weekly EIA quantities (stocks, supply, trade, products supplied).
EIA_QUANTITIES = [
    'commercial_crude_stocks', 'spr_crude_stocks', 'gasoline_stocks', 'distillate_stocks',
    'jet_fuel_stocks', 'total_petroleum_stocks_ex_spr', 'crude_production', 'crude_imports',
    'crude_exports', 'crude_net_imports', 'crude_refinery_inputs', 'total_products_supplied',
    'gasoline_products_supplied', 'distillate_products_supplied', 'jet_fuel_products_supplied',
]
GALLONS_PER_BARREL = 42
REPORTS_PER_YEAR = 52


def report_ids(table):
    """Number the weekly EIA reports: a new report starts when the quantities change."""
    return table[EIA_QUANTITIES].diff().ne(0).any(axis=1).cumsum()


def eia_features(table, denoise=False):
    """Change since the previous report, and year-over-year change of the
    4-report average (removes seasonality). Held constant until the next report.

    denoise=True computes the weekly change of the noisy series from their
    causal Kalman level instead of the raw report values (see denoise.py)."""
    ids = report_ids(table)
    reports = table.groupby(ids)[EIA_QUANTITIES].first()
    smooth = reports.rolling(4).mean()
    level = reports
    if denoise:
        from .denoise import denoise_reports
        level, _, _ = denoise_reports(reports, table.groupby(ids)['date'].first())
    weekly = level.diff().add_suffix('_wk_chg')
    yoy = (smooth - smooth.shift(REPORTS_PER_YEAR)).add_suffix('_yoy_chg')
    out = pd.concat([weekly, yoy], axis=1)
    return out.loc[ids].set_axis(table.index)


def wti_features(table):
    """Momentum and volatility of the already-published WTI price process."""
    lp = safe_log_price(observable_price(table))
    daily = lp.diff()
    out = {f'wti_ret_{k}d': lp - lp.shift(k) for k in (1, 5, 20, 60)}
    out.update({f'wti_vol_{k}d': daily.rolling(k).std() for k in (20, 60)})
    return pd.DataFrame(out)


def spread_features(table):
    """Benchmark spread and refining margins (USD/barrel) from EIA quotes."""
    wti = observable_price(table)
    gasoline = table['ny_gasoline_spot'] * GALLONS_PER_BARREL
    heating_oil = table['ny_heating_oil_spot'] * GALLONS_PER_BARREL
    diesel = table['gulf_diesel_spot'] * GALLONS_PER_BARREL
    return pd.DataFrame({
        'brent_wti_spread': table['brent_spot'] - wti,
        'gasoline_crack': gasoline - wti,
        'diesel_crack': diesel - wti,
        'crack_321': (2 * gasoline + heating_oil) / 3 - wti,
        'brent_ret_20d': np.log(table['brent_spot']).diff(20),
    })


def market_features(table):
    """Rates, volatility indices, equities and the dollar."""
    rates = ['treasury_2y_yield', 'treasury_10y_yield', 'real_treasury_10y_yield', 'breakeven_inflation_10y']
    out = {c: table[c] for c in rates + ['oil_implied_volatility', 'vix_close']}
    out.update({f'{c}_chg_20d': table[c].diff(20) for c in rates})
    out['term_spread'] = table['treasury_10y_yield'] - table['treasury_2y_yield']
    for c, name in [('sp500_close', 'sp500'), ('yahoo_dollar_index_close', 'dollar')]:
        lp = np.log(table[c])
        out.update({f'{name}_ret_{k}d': lp.diff(k) for k in (5, 20)})
    return pd.DataFrame(out)


def build_features(table, denoise=False, news=False):
    """All shared features, one row per trading day. Rows in the warm-up period
    (first ~13 months, before year-over-year changes exist) contain NaN.

    news=True adds the news-index features (ml/common/news.py)."""
    blocks = [wti_features(table), eia_features(table, denoise),
              spread_features(table), market_features(table)]
    if news:
        from .news import news_features
        blocks.append(news_features(table['date']))
    return pd.concat(blocks, axis=1)

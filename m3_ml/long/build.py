"""Long-history daily table from 1991-02-15 (doc/data/long/README.md); wti_daily is not touched.

  uv run python -I ml/long/build.py

Same clock as data/processed/wti_daily: every feature on a row was public before
00:00 New York time that day, and target_wti is that day's close (FRED). 26 of
wti_daily's 32 features; the 6 that start after 1991 are dropped. All inputs are
current vintages, and publication dates follow fixed rules instead of the dated
report archive (which starts in 2011):
  - weekly EIA data for the week ending Friday F: released on the 3rd US federal
    business day after F, usable 2 business days after the release (F + 5 business
    days; wti_daily uses the actual release + 1);
  - daily EIA prices: published with the report of the Saturday-Friday week that
    contains them, so usable on the same day as that week's quantities;
  - market closes: usable 2 US federal business days later (as in wti_daily).
Values older than 21 (EIA) or 7 (market) calendar days count as stale and, as in
wti_daily's export, the last published value is carried forward. Nothing published
after the 2026-10-15 data freeze is used.

Writes data/long/wti_daily_long.{parquet,csv}, features_long.csv and
overlap_comparison.csv (every column against wti_daily on the common dates).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
OUT = DATA / 'long'
# First week of exports and products supplied ends 1991-02-08 and is usable from 1991-02-15.
START, END, FREEZE = '1991-02-15', '2026-10-06', '2026-10-15'
BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
WEEKLY_LAG = 5      # business days from the Friday week end: release on the 3rd, usable 2 later
MARKET_LAG = 2
MAX_AGE = {'eia': 21, 'market': 7}

WEEKLY = {  # column: (file under data/, multiplier to wti_daily units)
    'commercial_crude_stocks': ('raw/eia_api/commercial_crude_stocks_weekly.csv', .001),
    'crude_exports': ('raw/eia_api/crude_exports_weekly.csv', 1),
    'crude_imports': ('raw/eia_api/crude_imports_weekly.csv', 1),
    'crude_production': ('raw/eia_api/crude_production_weekly.csv', 1),
    'crude_refinery_inputs': ('raw/eia_api/refinery_crude_inputs_weekly.csv', 1),
    'distillate_products_supplied': ('long/eia_api/distillate_products_supplied_weekly.csv', 1),
    'distillate_stocks': ('long/eia_api/distillate_stocks_weekly.csv', .001),
    'gasoline_products_supplied': ('long/eia_api/gasoline_products_supplied_weekly.csv', 1),
    'gasoline_stocks': ('long/eia_api/gasoline_stocks_weekly.csv', .001),
    'jet_fuel_products_supplied': ('long/eia_api/jet_fuel_products_supplied_weekly.csv', 1),
    'jet_fuel_stocks': ('long/eia_api/jet_fuel_stocks_weekly.csv', .001),
    'spr_crude_stocks': ('long/eia_api/spr_crude_stocks_weekly.csv', .001),
    'total_petroleum_stocks_ex_spr': ('long/eia_api/total_stocks_ex_spr_weekly.csv', .001),
    'total_products_supplied': ('long/eia_api/total_products_supplied_weekly.csv', 1),
}
DAILY_EIA = {  # column: (file under data/, value column); daily quotes released weekly
    'brent_spot': ('raw/fred/brent_spot_daily.csv', 'DCOILBRENTEU'),
    'gulf_gasoline_spot': ('long/eia_api/gulf_conventional_gasoline_spot_daily.csv', 'value'),
    'gulf_jet_fuel_spot': ('long/eia_api/gulf_jet_fuel_spot_daily.csv', 'value'),
    'ny_gasoline_spot': ('raw/eia_api/spot_prices/ny_conventional_gasoline_spot_daily.csv', 'value'),
    'ny_heating_oil_spot': ('raw/eia_api/spot_prices/ny_heating_oil_spot_daily.csv', 'value'),
    'wti_spot_last_published': ('raw/fred/wti_spot_daily.csv', 'DCOILWTICO'),
}
MARKET = {  # column: (file under data/, value column)
    'sp500_close': ('raw/futures/sp500_daily.csv', 'Close'),
    'treasury_10y_yield': ('raw/fred/treasury_yield_10y_daily.csv', 'DGS10'),
    'treasury_2y_yield': ('raw/fred/treasury_yield_2y_daily.csv', 'DGS2'),
    'vix_close': ('raw/futures/vix_daily.csv', 'Close'),
    'yahoo_dollar_index_close': ('raw/futures/dollar_index_daily.csv', 'Close'),
}
DROPPED = {  # wti_daily columns without history back to 1991
    'gulf_diesel_spot': 'starts 2006', 'la_rbob_spot': 'starts 2003', 'mont_belvieu_propane_spot': 'starts 1992-07',
    'oil_implied_volatility': 'starts 2007 (wti_daily uses it from 2009)', 'real_treasury_10y_yield': 'starts 2003',
    'breakeven_inflation_10y': 'starts 2003',
}
SOURCES = {
    'eia_weekly': 'EIA API weekly series, current vintage',
    'eia_daily': 'EIA daily spot price (via EIA API or FRED), current vintage',
    'market': 'FRED or Yahoo close',
}


def shifted(dates, n):
    lookup = {d: pd.Timestamp(d) + n * BUSINESS_DAY for d in dates.unique()}
    return dates.map(lookup).astype('datetime64[ns]')


def read(path, value, date='observation_date'):
    raw = pd.read_csv(DATA / path, dtype=str, keep_default_na=False)
    if date not in raw:
        date = 'date'
    frame = pd.DataFrame({'observation_date': pd.to_datetime(raw[date]),
                          'value': pd.to_numeric(raw[value], errors='coerce').astype(float)})
    if path.startswith('raw/futures/'):           # Yahoo: close only, rows with consistent OHLC (as in prepare_data)
        prices = raw[['Open', 'High', 'Low', 'Close']].apply(pd.to_numeric, errors='coerce')
        tol = prices.abs().max(axis=1).clip(lower=1) * 1e-7
        ok = prices.notna().all(axis=1) & (prices.High + tol >= prices.max(axis=1)) & (prices.Low - tol <= prices.min(axis=1))
        frame = frame[ok]
    frame = frame.dropna().sort_values('observation_date')
    assert not frame.observation_date.duplicated().any(), path
    return frame.reset_index(drop=True)


def week_end(dates):
    return dates + pd.to_timedelta((4 - dates.dt.weekday) % 7, unit='D')


def events(column, cutoff):
    """Observations of one feature with the date they become usable; observations on or
    after `cutoff` are removed (for the truncation test)."""
    if column in WEEKLY or column == 'crude_net_imports':
        if column == 'crude_net_imports':
            imports, exports = (read(WEEKLY[c][0], 'value').set_index('observation_date').value for c in ('crude_imports', 'crude_exports'))
            frame = (imports - exports).dropna().rename('value').reset_index()
        else:
            path, multiplier = WEEKLY[column]
            frame = read(path, 'value')
            frame['value'] *= multiplier
        assert (frame.observation_date.dt.weekday == 4).all(), f'{column}: weeks must end on Friday'
        frame['available_date'] = shifted(frame.observation_date, WEEKLY_LAG)
    elif column in DAILY_EIA:
        frame = read(*DAILY_EIA[column])
        frame['available_date'] = shifted(week_end(frame.observation_date), WEEKLY_LAG)
    else:
        frame = read(*MARKET[column])
        frame['available_date'] = shifted(frame.observation_date, MARKET_LAG)
    if cutoff is not None:
        frame = frame[frame.observation_date < cutoff]
    return frame[frame.available_date <= FREEZE]


def align(frame, calendar, max_age):
    """Newest observation usable at each date (later reprints of old weeks cannot rewind it)."""
    frame = frame.sort_values(['available_date', 'observation_date']).drop_duplicates('available_date', keep='last')
    frame = frame[frame.observation_date >= frame.observation_date.cummax()]
    aligned = pd.merge_asof(calendar, frame, left_on='date', right_on='available_date', direction='backward')
    age = (aligned.date - aligned.observation_date).dt.days
    return aligned.value.where(age <= max_age), age


def columns():
    reference = pd.read_parquet(DATA / 'processed/wti_daily.parquet').columns
    return [c for c in reference[2:] if c not in DROPPED]


def build(cutoff=None):
    """The long table and, per feature, how many rows carried a stale value forward."""
    target = read('raw/fred/wti_spot_daily.csv', 'DCOILWTICO')
    target = target[target.observation_date.between(START, END)]
    table = pd.DataFrame({'date': target.observation_date.astype('datetime64[ns]').to_numpy(),
                          'target_wti': target.value.to_numpy()})
    stale = {}
    for column in columns():
        group = 'market' if column in MARKET else 'eia'
        table[column], age = align(events(column, cutoff), table[['date']], MAX_AGE[group])
        stale[column] = int((age > MAX_AGE[group]).sum())
    features = table.columns[2:]
    table[features] = table[features].ffill()     # causal: the last published value, never interpolated
    return table, stale


def dictionary(table, stale):
    rows = [{'column': 'date', 'group': 'index', 'units': '', 'frequency': 'daily', 'source': '', 'available': '',
             'first_observation': '', 'stale_rows_carried_forward': '',
             'description': 'Trading day being forecast. Every feature on this row was public before 00:00 New York time that day.'},
            {'column': 'target_wti', 'group': 'target', 'units': 'USD/barrel', 'frequency': 'daily', 'source': 'FRED DCOILWTICO',
             'available': 'not available at forecast time', 'first_observation': '', 'stale_rows_carried_forward': '',
             'description': 'WTI Cushing spot price on this date. The value to forecast; never use as a feature.'}]
    reference = pd.read_csv(DATA / 'processed/features.csv').set_index('column')
    for column in table.columns[2:]:
        if column in MARKET:
            group, source, available = 'market', reference.loc[column, 'source'], '2 US federal business days after the close'
        elif column in DAILY_EIA:
            group, source = 'eia', SOURCES['eia_daily']
            available = "with the week's EIA report: 5 US federal business days after that week's Friday"
        else:
            group, source = 'eia', SOURCES['eia_weekly']
            available = '5 US federal business days after the Friday week end (release on the 3rd, usable 2 later)'
        first = events(column, None).observation_date.min()
        rows.append({'column': column, 'group': group, 'units': reference.loc[column, 'units'],
                     'frequency': 'weekly' if column in WEEKLY or column == 'crude_net_imports' else 'daily',
                     'source': source, 'available': available, 'first_observation': f'{first:%Y-%m-%d}',
                     'stale_rows_carried_forward': stale[column], 'description': reference.loc[column, 'description']})
    return pd.DataFrame(rows)


def compare(table):
    """Each column against wti_daily on the common dates: same value on the same row, within
    two rows (publication timing), or ahead of wti_daily (usable earlier than the archive)."""
    ref = pd.read_parquet(DATA / 'processed/wti_daily.parquet')
    ref['date'] = ref['date'].astype('datetime64[ns]')
    both = table.merge(ref, on='date', suffixes=('', '_ref'))
    rows = []
    for column in ['target_wti', *table.columns[2:]]:
        a, b = both[column], both[f'{column}_ref']
        same = np.isclose(a, b, rtol=1e-9, atol=1e-6)
        lagged = same | np.isclose(a, b.shift(1), rtol=1e-9, atol=1e-6) | np.isclose(a, b.shift(2), rtol=1e-9, atol=1e-6)
        ahead = ~same & np.isclose(a, b.shift(-1), rtol=1e-9, atol=1e-6)
        differ = ~lagged & ~ahead
        rows.append({'column': column, 'rows': len(both), 'same_row': round(same.mean(), 4), 'within_2_rows': round(lagged.mean(), 4),
                     'ahead_of_wti_daily': int(ahead.sum()), 'other_differences': int(differ.sum()),
                     'median_abs_difference': float(np.round((a - b)[differ].abs().median(), 4)) if differ.any() else 0.0,
                     'differences_from': f'{both.date[differ].min():%Y-%m-%d}' if differ.any() else '',
                     'differences_to': f'{both.date[differ].max():%Y-%m-%d}' if differ.any() else ''})
    return pd.DataFrame(rows)


def main():
    table, stale = build()
    assert not table.isna().any().any(), table.columns[table.isna().any()].tolist()
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT / 'wti_daily_long.parquet', index=False)
    table.to_csv(OUT / 'wti_daily_long.csv', index=False, date_format='%Y-%m-%d', lineterminator='\n')
    dictionary(table, stale).to_csv(OUT / 'features_long.csv', index=False, lineterminator='\n')
    comparison = compare(table)
    comparison.to_csv(OUT / 'overlap_comparison.csv', index=False, lineterminator='\n')
    print(f'wti_daily_long: {len(table)} rows, {table.shape[1] - 2} features, '
          f'{table.date.min():%Y-%m-%d} to {table.date.max():%Y-%m-%d}, no missing values')
    print(comparison.to_string(index=False))


if __name__ == '__main__':
    main()

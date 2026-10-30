"""Export the trimmed, model-ready WTI table that the group works from.

Reads the full audited build (.local/rebuild/release) and writes data/processed/:
  wti_daily.parquet / wti_daily.csv  one row per trading day with a WTI quote
  features.csv                       what every column means
  wti_publication.csv                verified or assumed label-use dates

Run after prepare_data.py and validate_prepared.py have passed.
"""
from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from data_layout import PROCESSED, TEAM_DATA, output_path

START, END = '2011-08-04', '2026-10-23'
TARGET = 'target_wti'
RENAME = {'forecast_date': 'date', 'y_wti_spot_usd_per_barrel': TARGET,
          'x_wti_spot': 'x_wti_spot_last_published'}

DESCRIPTIONS = {
    'date': 'Trading day being forecast. Every feature on this row was public before 00:00 New York time that day.',
    TARGET: 'WTI Cushing spot price on this date (FRED DCOILWTICO). The value to forecast; never use as a feature.',
    'wti_spot_last_published': 'Latest WTI spot quote published by EIA before this date (lagged, not the same-day target).',
    'brent_spot': 'Latest Brent spot quote published by EIA.',
    'commercial_crude_stocks': 'US commercial crude oil stocks, excluding SPR.',
    'spr_crude_stocks': 'Strategic Petroleum Reserve crude stocks.',
    'gasoline_stocks': 'US total motor gasoline stocks.',
    'distillate_stocks': 'US distillate fuel oil stocks.',
    'jet_fuel_stocks': 'US kerosene-type jet fuel stocks.',
    'total_petroleum_stocks_ex_spr': 'US total crude and petroleum product stocks, excluding SPR.',
    'crude_production': 'US domestic crude oil production.',
    'crude_imports': 'US crude oil imports.',
    'crude_exports': 'US crude oil exports.',
    'crude_net_imports': 'US crude oil imports minus exports.',
    'crude_refinery_inputs': 'Crude oil input to US refineries.',
    'total_products_supplied': 'Total petroleum products supplied (demand proxy).',
    'gasoline_products_supplied': 'Motor gasoline products supplied (demand proxy).',
    'distillate_products_supplied': 'Distillate fuel oil products supplied (demand proxy).',
    'jet_fuel_products_supplied': 'Jet fuel products supplied (demand proxy).',
    'ny_gasoline_spot': 'New York Harbor conventional gasoline spot price.',
    'gulf_gasoline_spot': 'US Gulf Coast conventional gasoline spot price.',
    'la_rbob_spot': 'Los Angeles RBOB gasoline spot price.',
    'ny_heating_oil_spot': 'New York Harbor No. 2 heating oil spot price.',
    'gulf_diesel_spot': 'US Gulf Coast ultra-low sulfur diesel spot price.',
    'gulf_jet_fuel_spot': 'US Gulf Coast kerosene-type jet fuel spot price.',
    'mont_belvieu_propane_spot': 'Mont Belvieu, TX propane spot price.',
    'treasury_2y_yield': '2-year US Treasury constant-maturity yield.',
    'treasury_10y_yield': '10-year US Treasury constant-maturity yield.',
    'real_treasury_10y_yield': '10-year US Treasury inflation-indexed (real) yield.',
    'breakeven_inflation_10y': '10-year breakeven inflation rate.',
    'oil_implied_volatility': 'CBOE crude oil volatility index (OVX).',
    'vix_close': 'CBOE VIX closing level.',
    'sp500_close': 'S&P 500 index closing level.',
    'yahoo_dollar_index_close': 'US Dollar Index (DX-Y.NYB) closing level.',
}


def load_full():
    return pd.read_parquet(output_path(PROCESSED, 'daily/extended.parquet'))


def build_table(full):
    window = full['forecast_date'].between(START, END) & full['target_status'].eq('observed')
    features = [c for c in full if c.startswith('x_')]
    table = full.loc[window, ['forecast_date', 'y_wti_spot_usd_per_barrel', *features]].rename(columns=RENAME)
    table.columns = [c.removeprefix('x_') for c in table.columns]
    # The build blanks EIA values older than 21 days (only 2024-06-21..07-03 in this window).
    # Carrying the last published value forward is causal: it is what a forecaster had.
    # Never interpolate or backward-fill; both leak later reports.
    features = table.columns[2:]
    table[features] = table[features].ffill()
    assert not table.isna().any().any(), 'missing values remain after forward-fill'
    return table.reset_index(drop=True)


def build_dictionary(table):
    series = pd.read_csv(output_path(PROCESSED, 'metadata/series_dictionary.csv'))
    eia = series[series['source'].eq('EIA_WPSR_ARCHIVE')].set_index('feature')
    market = series[series['eligibility'].eq('market_assumption')].set_index('feature')
    rows = [{'column': 'date', 'group': 'index', 'units': '', 'frequency': 'daily',
             'source': '', 'available': '', 'description': DESCRIPTIONS['date']},
            {'column': TARGET, 'group': 'target', 'units': 'USD/barrel', 'frequency': 'daily',
             'source': 'FRED DCOILWTICO', 'available': 'not available at forecast time',
             'description': DESCRIPTIONS[TARGET]}]
    for column in table.columns[2:]:
        key = 'wti_spot' if column == 'wti_spot_last_published' else column
        group, meta = ('eia', eia.loc[key]) if key in eia.index else ('market', market.loc[key])
        source = 'EIA weekly report' if group == 'eia' else f"{meta['source']} {meta['series_id']}"
        frequency = 'weekly' if group == 'eia' else 'daily'
        if group == 'eia' and meta['frequency'] != 'weekly':
            frequency = 'daily quotes, released weekly'
        rows.append({'column': column, 'group': group, 'units': meta['units'], 'frequency': frequency,
                     'source': source, 'available': meta['availability_policy'],
                     'description': DESCRIPTIONS[column]})
    return pd.DataFrame(rows)


def build_publication(table, full):
    """Label-use dates, distinguishing archive evidence from a weekly assumption."""
    evidence = full[['forecast_date', 'target_verified_available_date',
                     'target_publication_source_file']].rename(columns={'forecast_date': 'date'})
    result = table[['date', TARGET]].merge(evidence, on='date', validate='one_to_one')
    dates = pd.to_datetime(result['date'])
    friday = dates + pd.to_timedelta((4 - dates.dt.weekday) % 7, unit='D')
    business_day = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    assumed = pd.to_datetime([day + 5 * business_day for day in friday])
    verified = result['target_verified_available_date'].notna()
    result['available_date'] = result['target_verified_available_date'].where(verified, assumed)
    result['evidence'] = verified.map({True: 'matching_dated_archive', False: 'assumed_weekly_release'})
    result['source_file'] = result['target_publication_source_file'].fillna('')
    assert result['available_date'].gt(dates).all()
    return result[['date', TARGET, 'available_date', 'evidence', 'source_file']]


def main():
    full = load_full()
    table = build_table(full)
    dictionary = build_dictionary(table)
    assert list(dictionary['column']) == list(table.columns)
    TEAM_DATA.mkdir(parents=True, exist_ok=True)
    table.to_parquet(TEAM_DATA / 'wti_daily.parquet', index=False)
    table.to_csv(TEAM_DATA / 'wti_daily.csv', index=False, date_format='%Y-%m-%d', lineterminator='\n')
    dictionary.to_csv(TEAM_DATA / 'features.csv', index=False, lineterminator='\n')
    build_publication(table, full).to_csv(TEAM_DATA / 'wti_publication.csv', index=False,
                                        date_format='%Y-%m-%d', lineterminator='\n')
    print(f"wti_daily: {len(table)} rows, {table.shape[1] - 2} features, "
          f"{table['date'].min().date()} to {table['date'].max().date()}, no missing values")


if __name__ == '__main__':
    main()

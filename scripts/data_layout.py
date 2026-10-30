"""Single path registry for data CSVs, local rebuild support and deliverables.

Legacy download manifests remain byte-for-byte intact. Their relative paths are
translated in memory; raw observations and historical download hashes never change.
"""
from pathlib import Path
from functools import lru_cache
import os

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
LOCAL = ROOT / '.local/rebuild'
SNAPSHOT = ROOT / 'data/rebuild'
# Full audited build stays local; the team gets a trimmed export (export_team_data.py).
PROCESSED = LOCAL / 'release'
TEAM_DATA = ROOT / 'data/processed'
INPUTS = SNAPSHOT if SNAPSHOT.is_dir() else LOCAL
MANIFESTS = INPUTS / 'manifests'
SUPPORT = INPUTS / 'source_support'
INTERIM = LOCAL / 'download_tables'

FRED_NAMES = {
    'DCOILWTICO_wti_spot.csv': 'wti_spot_daily.csv',
    'DCOILBRENTEU_brent_spot.csv': 'brent_spot_daily.csv',
    'DTWEXBGS_broad_dollar_index.csv': 'trade_weighted_dollar_daily.csv',
    'DGS2_us_treasury_2y.csv': 'treasury_yield_2y_daily.csv',
    'DGS10_us_treasury_10y.csv': 'treasury_yield_10y_daily.csv',
}
MARKET_NAMES = {'OVXCLS.csv': 'oil_volatility_ovx_daily.csv',
                'T10YIE.csv': 'breakeven_inflation_10y_daily.csv',
                'DFII10.csv': 'real_treasury_yield_10y_daily.csv'}
YAHOO_NAMES = {name + '.csv': name + '_daily.csv' for name in (
    'wti_nearby_futures', 'brent_futures', 'sp500', 'vix',
    'energy_select_sector_etf', 'gold_futures', 'natural_gas_futures')}
YAHOO_NAMES['dollar_index_futures.csv'] = 'dollar_index_daily.csv'
EIA_NAMES = {
    'PET.WCESTUS1.W_commercial_crude_stocks_ex_spr.csv': 'commercial_crude_stocks_weekly.csv',
    'PET.W_EPC0_SAX_YCUOK_MBBL.W_cushing_crude_stocks.csv': 'cushing_crude_stocks_weekly.csv',
    'PET.WCRFPUS2.W_us_crude_field_production.csv': 'crude_production_weekly.csv',
    'PET.WCRRIUS2.W_us_refinery_crude_inputs.csv': 'refinery_crude_inputs_weekly.csv',
    'PET.WPULEUS3.W_us_refinery_utilization.csv': 'refinery_utilization_weekly.csv',
    'PET.WCRIMUS2.W_us_crude_imports.csv': 'crude_imports_weekly.csv',
    'PET.WCREXUS2.W_us_crude_exports.csv': 'crude_exports_weekly.csv',
}
REPORT_NAMES = {'table1.csv': 'table_01_petroleum_balance.csv',
                'table11.csv': 'table_11_crude_and_gasoline_prices.csv',
                'table12.csv': 'table_12_other_product_prices.csv'}
HELD_OUT = {
    'data/raw/cftc/wti_positions_weekly.csv',
    'data/raw/fred/trade_weighted_dollar_daily.csv',
    'data/raw/futures/brent_futures_daily.csv',
    'data/raw/futures/energy_select_sector_etf_daily.csv',
    'data/raw/futures/gold_futures_daily.csv',
    'data/raw/futures/natural_gas_futures_daily.csv',
    'data/raw/futures/wti_nearby_futures_daily.csv',
    'data/raw/eia_api/cushing_crude_stocks_weekly.csv',
    'data/raw/eia_api/refinery_utilization_weekly.csv',
}


def hold_unused(path):
    return (INPUTS / 'held-out' / path).relative_to(ROOT).as_posix() if path in HELD_OUT else path


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


@lru_cache(maxsize=32768)
def relocate_reference(value):
    """Translate known legacy paths only; idempotent for the new layout."""
    p = value.replace('\\', '/')
    for folder in ('source_support', 'held-out', 'manifests'):
        prefix = f'.local/rebuild/{folder}/'
        if p.startswith(prefix):
            return relative(INPUTS / folder / p.removeprefix(prefix))
    fixed = {
        'data/raw/download_manifest.json': relative(MANIFESTS / 'public_download.json'),
        'data/raw/eia/download_manifest.json': relative(MANIFESTS / 'eia_api_download.json'),
        'data/raw/cftc/wti_067651_disaggregated_futures_only.csv': 'data/raw/cftc/wti_positions_weekly.csv',
        'data/raw/supplement/validation_summary.json': relative(LOCAL / 'checks/supplement_validation.json'),
    }
    if p in fixed:
        return hold_unused(fixed[p])
    for old, names, new in [('fred', FRED_NAMES, 'fred'), ('yfinance', YAHOO_NAMES, 'futures'),
                             ('eia', EIA_NAMES, 'eia_api')]:
        prefix = 'data/raw/' + old + '/'
        if p.startswith(prefix) and p[len(prefix):] in names:
            return hold_unused('data/raw/' + new + '/' + names[p[len(prefix):]])
    if p.startswith('data/raw/supplement/market/'):
        rest = p.removeprefix('data/raw/supplement/market/')
        date, name = rest.split('/')
        if name in MARKET_NAMES:
            return f'data/raw/fred/snapshots/{date}/{MARKET_NAMES[name]}'
    if p.startswith('data/raw/supplement/wpsr/') and p.endswith('.csv'):
        rest = p.removeprefix('data/raw/supplement/wpsr/')
        folder, name = rest.split('/')
        if name in REPORT_NAMES:
            return f'data/raw/eia_reports/{folder}/{REPORT_NAMES[name]}'
    if p.startswith('data/raw/supplement/manifest_') and p.endswith('.json'):
        return relative(MANIFESTS / 'supplement' / Path(p).name)
    if p.startswith('data/raw/') and Path(p).suffix.lower() in {'.json', '.html', '.zip'}:
        return relative(SUPPORT / p.removeprefix('data/raw/'))
    if p.startswith('data/interim/'):
        return relative(INTERIM / Path(p).name)
    if p in {'data/processed/supplement_daily_market.csv',
             'data/processed/supplement_eia_release_history.csv',
             'data/processed/eia_weekly_petroleum_fundamentals.csv'}:
        return relative(INTERIM / Path(p).name)
    return hold_unused(p)


def source_path(path):
    p = Path(path)
    value = relative(p) if p.is_absolute() else p.as_posix()
    return ROOT / relocate_reference(value)


def relocate_manifest(value):
    if isinstance(value, dict):
        return {k: relocate_manifest(v) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate_manifest(v) for v in value]
    if isinstance(value, str) and value.replace('\\', '/').startswith(('data/raw/', 'data/interim/', 'data/processed/', '.local/rebuild/')):
        return relocate_reference(value)
    return value


# Logical names are stable internal identifiers. Physical names describe purpose.
DATA_OUTPUTS = {
    'daily/core': 'forecasting/wti_spot_core_daily',
    'daily/extended': 'forecasting/wti_spot_extended_daily',
    'targets/wti_spot_daily': 'targets/wti_spot_observed_daily',
    'targets/wti_spot_release_vintages': 'targets/wti_spot_publication_history',
    'normalized/fred_daily': 'reference_only/fred_series_daily',
    'normalized/yahoo_ohlcv': 'reference_only/yahoo_prices_daily',
    'reference/eia_api_weekly': 'reference_only/eia_latest_fundamentals_weekly',
    'reference/cftc_wti_weekly': 'reference_only/wti_positions_weekly',
    'metadata/series_dictionary': 'dataset_info/series_dictionary',
    'metadata/column_dictionary': 'dataset_info/column_dictionary',
    'audit/daily_coverage': 'dataset_info/forecasting_coverage',
    'audit/issues': 'dataset_info/data_issues',
    'audit/quarantine': 'dataset_info/excluded_reports',
    'audit/target_extreme_observations': 'dataset_info/wti_extreme_prices',
    'audit/quality_summary.json': 'dataset_info/quality_summary.json',
    'audit/validation_report.json': 'dataset_info/validation_report.json',
    'metadata/release_manifest.json': 'dataset_info/release_manifest.json',
}
LOCAL_OUTPUTS = {
    'core/eia_release_history': 'feature_observations/eia_report_observations',
    'auxiliary/market_observations': 'feature_observations/market_observations',
    'audit/daily_alignment': 'audit/forecasting_alignment',
    'audit/vintage_comparison': 'audit/historical_vs_latest_values',
    'audit/source_inventory': 'audit/source_inventory',
    'audit/source_quality': 'audit/source_quality',
    'metadata/build_manifest.json': 'build_manifest.json',
    'build_status.json': 'build_status.json',
}


def build_workspace(out=PROCESSED):
    out = Path(out).resolve()
    return LOCAL / 'build' if out == PROCESSED.resolve() else out.parent / ('.' + out.name + '_rebuild')


def output_path(out, logical):
    key, suffix = logical, ''
    if logical.endswith(('.csv', '.parquet')):
        key, suffix = logical.rsplit('.', 1)
        suffix = '.' + suffix
    if key in DATA_OUTPUTS:
        return Path(out) / (DATA_OUTPUTS[key] + suffix)
    if key in LOCAL_OUTPUTS:
        return build_workspace(out) / (LOCAL_OUTPUTS[key] + suffix)
    raise KeyError('Unknown output artifact: ' + logical)


def output_reference(out, path):
    return os.path.relpath(path, out).replace('\\', '/')


SHARED_FILES = tuple(sorted(
    [name + ext for key, name in DATA_OUTPUTS.items() if not key.endswith('.json')
     for ext in (('.csv',) if key.startswith(('metadata/', 'audit/')) else ('.csv', '.parquet'))]
    + [DATA_OUTPUTS[k] for k in ('audit/quality_summary.json', 'audit/validation_report.json')]
))

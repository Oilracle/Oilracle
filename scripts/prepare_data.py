"""Offline, reproducible preparation of WTI spot forecasting data.

Run: uv run python scripts/prepare_data.py
Raw snapshots are never modified. No credentials, network, training or fitted
transformations are used. See the root README.md for the data contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from download_supplement import parse_prices, parse_table
from data_layout import (LOCAL, SNAPSHOT, MANIFESTS, PROCESSED, REPORT_NAMES, source_path,
                         relocate_manifest, output_path, output_reference, build_workspace)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
DEFAULT_OUT = PROCESSED
BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
MISSING = {'', '.', '-', '--', 'NA', 'N/A', 'NaN', 'nan', 'null', 'None', '–', '—'}
FRED = {
    'DCOILWTICO': ('wti_spot', 'USD/barrel', 'target_only', None),
    'DCOILBRENTEU': ('brent_spot_latest', 'USD/barrel', 'reference_only', None),
    'DTWEXBGS': ('broad_trade_weighted_dollar', 'index', 'reference_only', None),
    'DGS2': ('treasury_2y_yield', 'percent', 'market_assumption', None),
    'DGS10': ('treasury_10y_yield', 'percent', 'market_assumption', None),
    'OVXCLS': ('oil_implied_volatility', 'index', 'market_assumption', '2009-01-01'),
    'T10YIE': ('breakeven_inflation_10y', 'percent', 'market_assumption', '2003-01-01'),
    'DFII10': ('real_treasury_10y_yield', 'percent', 'market_assumption', '2003-01-01'),
}
YAHOO = {
    'CL=F': ('wti_nearby_futures', 'USD/barrel', False),
    'BZ=F': ('brent_futures', 'USD/barrel', False),
    '^GSPC': ('sp500', 'index', True),
    '^VIX': ('vix', 'index', True),
    'XLE': ('energy_select_sector_etf', 'USD/share', False),
    'GC=F': ('gold_futures', 'USD/troy_ounce', False),
    'NG=F': ('natural_gas_futures', 'USD/MMBtu', False),
    'DX-Y.NYB': ('yahoo_dollar_index', 'index', True),
}
EIA = {
    'PET.WCESTUS1.W': ('commercial_crude_stocks', 'million barrels', 'MBBL', .001),
    'PET.W_EPC0_SAX_YCUOK_MBBL.W': ('cushing_crude_stocks', 'million barrels', 'MBBL', .001),
    'PET.WCRFPUS2.W': ('crude_production', 'thousand barrels/day', 'MBBL/D', 1),
    'PET.WCRRIUS2.W': ('crude_refinery_inputs', 'thousand barrels/day', 'MBBL/D', 1),
    'PET.WPULEUS3.W': ('refinery_utilization', 'percent', '%', 1),
    'PET.WCRIMUS2.W': ('crude_imports', 'thousand barrels/day', 'MBBL/D', 1),
    'PET.WCREXUS2.W': ('crude_exports', 'thousand barrels/day', 'MBBL/D', 1),
}
COMMON = ['source', 'series_id', 'feature', 'observation_date', 'value', 'units',
          'release_date', 'available_date', 'availability_basis', 'eligibility',
          'source_file', 'source_row', 'source_missing', 'quality_flag']


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iso(value):
    return None if pd.isna(value) else pd.Timestamp(value).date().isoformat()


def shifted(dates: pd.Series, n: int) -> pd.Series:
    # Unique-date mapping is deterministic and avoids slow row-wise offsets.
    lookup = {d: pd.Timestamp(d) + n * BUSINESS_DAY for d in dates.dropna().unique()}
    return pd.to_datetime(dates.map(lookup))


def publication_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Newest observation known at each origin; old reprints cannot rewind it."""
    f = frame.loc[frame.value.notna()].sort_values(
        ['available_date', 'observation_date', 'release_date', 'source_file'], na_position='first')
    if f.empty:
        return f
    key = ['available_date', 'observation_date']
    conflicts = f.groupby(key).value.nunique().gt(1)
    if conflicts.any():
        raise ValueError('Conflicting equal-time observations must be quarantined before alignment')
    f = f.drop_duplicates(key).drop_duplicates('available_date', keep='last')
    f = f.loc[f.observation_date.ge(f.observation_date.cummax())]
    return f.reset_index(drop=True)


def align_feature(frame: pd.DataFrame, calendar: pd.DataFrame, max_age: int) -> pd.DataFrame:
    events = publication_events(frame).copy()
    calendar = calendar.copy()
    calendar['forecast_date'] = calendar.forecast_date.astype('datetime64[ns]')
    for c in ['observation_date', 'release_date', 'available_date']:
        events[c] = events[c].astype('datetime64[ns]')
    keep = ['observation_date', 'release_date', 'available_date', 'value',
            'source_file', 'source_row', 'quality_flag']
    aligned = pd.merge_asof(calendar, events[keep].sort_values('available_date'),
                            left_on='forecast_date', right_on='available_date', direction='backward')
    aligned['age_days'] = (aligned.forecast_date - aligned.observation_date).dt.days.astype('Int64')
    aligned['days_since_available'] = (aligned.forecast_date - aligned.available_date).dt.days.astype('Int64')
    aligned['max_age_days'] = max_age
    aligned['is_stale'] = aligned.age_days.gt(max_age).fillna(False)
    aligned['last_known_value'] = aligned['value']
    aligned.loc[aligned.is_stale, 'value'] = np.nan
    aligned['missing_reason'] = np.select(
        [aligned.available_date.isna(), aligned.is_stale],
        ['not_yet_available', 'stale'], default='')
    return aligned


class Builder:
    def __init__(self, out: Path, as_of: str | None):
        self.out = out.resolve()
        if self.out == RAW.resolve() or RAW.resolve() in self.out.parents:
            raise ValueError('Output must not be inside raw data')
        self.inventory, self.issues, self.quarantine, self.dictionary, self.outputs, self.schemas = {}, [], [], [], [], []
        self.base = self.load_manifest(RAW / 'download_manifest.json')
        self.api_manifest = self.load_manifest(RAW / 'eia/download_manifest.json')
        candidates = sorted(p for p in (MANIFESTS / 'supplement').glob('manifest_*.json')
                            if '_smoke' not in p.name)
        self.supplement = self.load_manifest(candidates[-1])
        if self.supplement['archive_errors']:
            raise ValueError('Supplement download has unresolved errors')
        timestamps = [m['retrieved_at_utc'] for m in (self.base, self.api_manifest, self.supplement)]
        self.as_of = pd.Timestamp(as_of or max(timestamps)[:10]).normalize()
        self.freeze_date = max(timestamps)[:10]
        if self.as_of > pd.Timestamp(self.freeze_date):
            raise ValueError('As-of cannot exceed the newest local snapshot date')
        self.register_sources()

    def load_manifest(self, path):
        path = source_path(path)
        payload = relocate_manifest(json.loads(path.read_text(encoding='utf-8')))
        self.track(path, role='input_manifest')
        return payload

    def track(self, path, expected=None, role='raw_source', url=''):
        path = Path(path).resolve()
        allowed_project_file = path.parent == ROOT and path.name in ('pyproject.toml', 'uv.lock')
        if (RAW.resolve() not in path.parents and LOCAL.resolve() not in path.parents
                and SNAPSHOT.resolve() not in path.parents
                and path.parent != ROOT / 'scripts' and not allowed_project_file):
            raise ValueError(f'Unexpected input location: {path}')
        key = path.relative_to(ROOT).as_posix()
        actual = digest(path)
        if expected and actual != expected:
            raise ValueError(f'Input hash mismatch: {key}')
        self.inventory[key] = {'source_file': key, 'role': role, 'bytes': path.stat().st_size,
                               'sha256': actual, 'source_url': url}
        return key

    def register_sources(self):
        for r in self.base['records']:
            self.track(ROOT / r['file'], r['sha256'], url=r['url'])
        for r in self.api_manifest['records']:
            for file_key, hash_key in [('csv_file', 'csv_sha256'), ('raw_json_file', 'raw_json_sha256')]:
                self.track(ROOT / r[file_key], r[hash_key], url=r['url'])
        for r in self.supplement['daily_records']:
            self.track(ROOT / r['file'], r['sha256'], url=r['url'])
        for r in self.supplement['archive_records']:
            for f in ([r] if 'files' not in r else r['files'] + [r['report']]):
                self.track(ROOT / f['file'], f['sha256'], url=f['url'])
        for r in self.supplement['archive_exclusions']:
            for f in r['files']:
                self.track(ROOT / f['file'], f['sha256'], role='quarantined_archive', url=f['url'])
            self.issue('eia_archive', '', None, 'date_conflict', r['reason'], 'exclude_report')
            self.quarantine.append({'dataset': 'eia_archive', 'feature': '', 'observation_date': None,
                'source_file': r['files'][0]['file'], 'source_row': None,
                'reason': r['reason'], 'action': 'excluded_from_all_daily_predictors'})
        for name in ['prepare_data.py', 'download_supplement.py', 'validate_prepared.py', 'data_layout.py']:
            self.track(ROOT / 'scripts' / name, role='pipeline_code')
        for name in ['pyproject.toml', 'uv.lock']:
            self.track(ROOT / name, role='environment_lock')

    def issue(self, dataset, feature, date, code, detail, action, severity='warning'):
        self.issues.append({'severity': severity, 'dataset': dataset, 'feature': feature,
                            'observation_date': iso(date), 'issue': code, 'detail': detail, 'action': action})

    def number(self, raw, dataset, field, path, dates):
        text = raw.astype('string').str.strip()
        missing = text.isin(MISSING) | text.isna()
        values = pd.to_numeric(text.str.replace(',', '', regex=False).mask(missing), errors='coerce')
        invalid = (~missing & values.isna()) | (values.notna() & ~np.isfinite(values))
        for i in values.index[invalid]:
            self.quarantine.append({'dataset': dataset, 'feature': field,
                'observation_date': iso(dates.loc[i]), 'source_file': path,
                'source_row': int(i) + 2, 'reason': 'invalid_numeric:' + str(text.loc[i]),
                'action': 'numeric_value_set_missing'})
        return values.mask(invalid).astype('float64'), missing, invalid

    def unique(self, frame, keys, dataset):
        exact = frame.duplicated(keys + ['value'], keep='first')
        if exact.any():
            self.issue(dataset, '', None, 'exact_duplicates', str(int(exact.sum())), 'keep_first')
        frame = frame.loc[~exact].copy()
        conflict = frame.duplicated(keys, keep=False)
        for row in frame.loc[conflict].to_dict('records'):
            self.quarantine.append({'dataset': dataset, 'feature': row.get('feature', ''),
                'observation_date': iso(row.get('observation_date')), 'source_file': row['source_file'],
                'source_row': row.get('source_row'), 'reason': 'conflicting_duplicate',
                'action': 'exclude_all_conflicting_rows'})
        return frame.loc[~conflict].sort_values(keys).reset_index(drop=True)

    def describe(self, source, code, feature, units, frequency, eligibility, policy, notes=''):
        self.dictionary.append({'source': source, 'series_id': code, 'feature': feature,
            'units': units, 'frequency': frequency, 'eligibility': eligibility,
            'availability_policy': policy, 'notes': notes})

    def save(self, name, frame, csv=True, parquet=True):
        path = output_path(self.out, name)
        dataset_name = output_reference(self.out, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        definitions = {
            'forecast_date': 'Research weekday. Information cutoff is 00:00 America/New_York.',
            'date': 'WTI spot observation date, not publication date.',
            'observation_date': 'Date measured; weekly fundamentals use week-ending dates.',
            'release_date': 'Verified archive report date; null if not independently established.',
            'available_date': 'Conservative allowed forecast-origin date; never the observation date by default.',
            'value': 'Numeric observation in units; in alignment tables stale values are masked.',
            'last_known_value': 'Last published value even if too stale; audit only, not a default predictor.',
            'quality_flag': 'Machine-readable flag; does not imply a valid extreme was removed.',
            'source_file': 'Project-relative immutable raw source path.',
            'source_row': '1-based CSV row including header, or row in ZIP member; null for label-based table extraction.',
            'source_locator': 'Feature and observation-date locator in archived raw table.',
            'source_missing': 'Provider supplied a missing token, distinct from invalid numeric text.',
            'units': 'Measurement units; percentages remain percentage points, not fractions.',
            'feature': 'Stable series name; consult series_dictionary for source and eligibility.',
            'age_days': 'Calendar days since observation, not days since publication.',
            'days_since_available': 'Calendar days since permitted first use.',
            'is_stale': 'True if age_days exceeds max_age_days; predictor is then null.',
            'target_status': 'observed, missing_in_source, or not_in_target_snapshot; never imputed.',
            'target_verified_available_date': 'First conservative archived availability date with the same quoted target value; null means unverified, not immediately available.',
            'target_publication_evidence': 'matching_dated_archive or unverified; model training must respect label availability separately from feature availability.',
            'target_publication_source_file': 'Raw dated report supporting target_verified_available_date.',
            'target_observed': 'Outcome exists. This is a selection/audit flag, not a predictor.',
            'available_feature_count': 'Count of nonmissing x_ columns at this forecast origin.',
            'all_features_available': 'Complete-case convenience flag; no rows are dropped.',
        }
        lookup = {r['feature']: r['units'] for r in self.dictionary}
        for c in frame:
            role = 'predictor' if c.startswith('x_') else 'metadata'
            if c in ('y_wti_spot_usd_per_barrel', 'wti_spot_usd_per_barrel'):
                role, description, unit = 'outcome', 'WTI Cushing spot outcome; not known at forecast origin. Never pass as predictor.', 'USD/barrel'
            else:
                description = definitions.get(c, c.replace('_', ' '))
                unit = lookup.get(c[2:], '') if c.startswith('x_') else ''
                if c.startswith('x_'):
                    description = 'As-of aligned ' + c[2:] + '; .local/rebuild/build/audit/forecasting_alignment supplies row-level lineage.'
            self.schemas.append({'dataset': dataset_name, 'column': c, 'dtype': str(frame[c].dtype),
                                 'role': role, 'units': unit, 'description': description})
        for extension, enabled in [('csv', csv), ('parquet', parquet)]:
            if not enabled:
                continue
            output = path.with_suffix('.' + extension)
            if extension == 'csv':
                frame.to_csv(output, index=False, date_format='%Y-%m-%d', lineterminator='\n')
            else:
                frame.to_parquet(output, index=False, engine='pyarrow')
            self.outputs.append({'file': output_reference(self.out, output),
                                 'rows': len(frame), 'columns': len(frame.columns),
                                 'sha256': digest(output), 'bytes': output.stat().st_size})

    def fred(self):
        records = [r for r in self.base['records'] if r['source'] == 'FRED'] + self.supplement['daily_records']
        frames = []
        for r in records:
            code = r['series']
            feature, units, eligibility, cutoff = FRED[code]
            path = Path(r['file']).as_posix()
            raw = pd.read_csv(ROOT / r['file'], dtype='string', keep_default_na=False)
            dates = pd.to_datetime(raw.observation_date, errors='raise')
            values, missing, invalid = self.number(raw[code], 'fred', code, path, dates)
            f = pd.DataFrame({'observation_date': dates, 'value': values})
            f['source'], f['series_id'], f['feature'], f['units'] = 'FRED', code, feature, units
            f['release_date'], f['available_date'] = pd.NaT, pd.NaT
            f['eligibility'] = eligibility
            f['availability_basis'] = 'unverified_do_not_use_as_predictor'
            f['quality_flag'] = np.select([invalid, missing], ['invalid_numeric', 'source_missing'], default='')
            f['source_missing'], f['source_file'], f['source_row'] = missing, path, raw.index + 2
            if eligibility == 'market_assumption':
                f['available_date'] = shifted(dates, 2)
                f['availability_basis'] = 'assumed_observation_plus_2_US_federal_business_days'
            if cutoff:
                f.loc[dates.lt(cutoff), 'eligibility'] = 'reference_only_pre_cutoff'
            frames.append(self.unique(f[COMMON], ['series_id', 'observation_date'], 'fred'))
            note = 'Latest downloaded vintage, not historical FRED vintages.'
            if code == 'DTWEXBGS':
                note += ' Weight/base revisions and publication timing unverified; no daily-model inclusion.'
            if code.startswith('DCOIL'):
                note += ' Daily observation frequency does not imply same-day publication.'
            if cutoff:
                note += f' Observations before {cutoff} excluded from predictors.'
            self.describe('FRED', code, feature, units, 'daily', eligibility,
                          '2 US federal business days (assumption)' if eligibility == 'market_assumption' else 'not assigned', note)
        result = pd.concat(frames, ignore_index=True)
        self.save('normalized/fred_daily', result)
        return result

    def yahoo(self):
        frames, auxiliary = [], []
        for r in [r for r in self.base['records'] if r['source'].startswith('yfinance')]:
            ticker = r['series']
            feature, units, use = YAHOO[ticker]
            raw = pd.read_csv(ROOT / r['file'], dtype='string', keep_default_na=False)
            path = Path(r['file']).as_posix()
            dates = pd.to_datetime(raw.date, errors='raise')
            if dates.dt.tz is not None:
                raise ValueError('Yahoo date timezone needs explicit exchange-local handling')
            f = pd.DataFrame({'observation_date': dates})
            invalid = pd.Series(False, index=raw.index)
            for field in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                name = field.lower().replace(' ', '_')
                f[name], _, bad = self.number(raw[field], 'yahoo', ticker + ':' + name, path, dates)
                invalid |= bad
            f['ticker'], f['feature'], f['units'] = ticker, feature, units
            f['source_file'], f['source_row'] = path, raw.index + 2
            f['eligibility'] = 'market_assumption_close_only' if use else 'reference_only'
            prices = f[['open', 'high', 'low', 'close']]
            tol = prices.abs().max(axis=1).clip(lower=1) * 1e-7
            f['ohlc_valid'] = (prices.notna().all(axis=1) & f.high.add(tol).ge(prices.max(axis=1))
                               & f.low.sub(tol).le(prices.min(axis=1)))
            f['volume_valid'] = f.volume.ge(0) & f.volume.mod(1).eq(0)
            f['zero_volume'] = f.volume.eq(0)
            f['nonpositive_close'] = f.close.le(0)
            f['quality_flag'] = np.select([invalid, ~f.ohlc_valid, ~f.volume_valid],
                ['invalid_numeric', 'ohlc_inconsistent_or_missing', 'invalid_volume'], default='')
            if f.observation_date.duplicated().any():
                raise ValueError(f'Duplicate Yahoo dates for {ticker}; review before selection')
            if (~f.ohlc_valid).any():
                self.issue('yahoo', feature, None, 'ohlc_inconsistent_or_missing',
                           str(int((~f.ohlc_valid).sum())), 'retain_reference_exclude_auxiliary_close')
            if (~f.volume_valid).any():
                self.issue('yahoo', feature, None, 'invalid_volume',
                           str(int((~f.volume_valid).sum())), 'retain_flag_volume_not_used')
            frames.append(f)
            policy = '2 US federal business days for close only' if use else 'not assigned'
            note = 'Adj Close never used as a predictor. Zero index volume is not measured trading activity.'
            if not use:
                note += ' Split-adjusted ETF history or continuous futures roll methodology is not point-in-time verified.'
            self.describe('Yahoo', ticker, feature + '_close', units, 'daily',
                          'market_assumption' if use else 'reference_only', policy, note)
            if use:
                a = f.loc[f.ohlc_valid & dates.ge('2004-01-01')].copy()
                # The modeled calendar starts in 2011; this also avoids using early VIX backcasts.
                a['source'], a['series_id'] = 'Yahoo', ticker
                a['feature'], a['value'] = feature + '_close', a['close']
                a['source_missing'] = a.value.isna()
                a['release_date'] = pd.NaT
                a['available_date'] = shifted(a.observation_date, 2)
                a['eligibility'] = 'market_assumption'
                a['availability_basis'] = 'assumed_close_plus_2_US_federal_business_days'
                auxiliary.append(a[COMMON])
        result = pd.concat(frames, ignore_index=True).sort_values(['ticker', 'observation_date'])
        self.save('normalized/yahoo_ohlcv', result)
        return result, pd.concat(auxiliary, ignore_index=True)

    def api(self):
        frames = []
        for r in self.api_manifest['records']:
            code = r['series_id']
            feature, units, raw_unit, multiplier = EIA[code]
            path = Path(r['csv_file']).as_posix()
            raw = pd.read_csv(ROOT / r['csv_file'], dtype='string', keep_default_na=False)
            if set(raw.units) != {raw_unit}:
                raise ValueError(f'Unexpected units for {code}: {set(raw.units)}')
            dates = pd.to_datetime(raw.observation_date, errors='raise')
            values, missing, invalid = self.number(raw.value, 'eia_api', feature, path, dates)
            f = pd.DataFrame({'observation_date': dates, 'original_value': values,
                              'original_units': raw.units, 'value': values * multiplier})
            f['source'], f['series_id'], f['feature'], f['units'] = 'EIA_API', code, feature, units
            f['release_date'], f['available_date'] = pd.NaT, pd.NaT
            f['source_file'], f['source_row'], f['source_missing'] = path, raw.index + 2, missing
            f['eligibility'], f['availability_basis'] = 'reference_only', 'unknown_historical_release_and_vintage'
            f['quality_flag'] = np.select([invalid, missing], ['invalid_numeric', 'source_missing'], default='')
            frames.append(self.unique(f, ['series_id', 'observation_date'], 'eia_api'))
            self.describe('EIA_API', code, feature, units, 'weekly', 'reference_only', 'not assigned',
                          f'Current-vintage API. Source unit {raw_unit}; multiplier {multiplier}. Not backfilled into core.')
        result = pd.concat(frames, ignore_index=True)
        self.save('reference/eia_api_weekly', result)
        return result

    def archive(self):
        rows = []
        for report in self.supplement['archive_records']:
            if 'files' not in report:
                continue
            for record in report['files']:
                path = ROOT / record['file']
                parser = parse_table if path.name == REPORT_NAMES['table1.csv'] else parse_prices
                rows.extend(parser(path.read_bytes(), report['release_date'], record['file']))
        result = pd.DataFrame(rows)
        for c in ['observation_date', 'release_date', 'available_date']:
            result[c] = pd.to_datetime(result[c], errors='raise')
        result['source'], result['series_id'] = 'EIA_WPSR_ARCHIVE', result.feature
        result['source_row'] = pd.Series(pd.NA, index=result.index, dtype='Int64')
        result['source_missing'], result['eligibility'] = False, 'dated_archive_core'
        result['quality_flag'] = ''
        transition = result.feature.eq('commercial_crude_stocks') & result.observation_date.between('2016-10-01', '2016-10-31')
        result.loc[transition, 'quality_flag'] = 'lease_stock_coverage_transition_review'
        if transition.any():
            self.issue('eia_archive', 'commercial_crude_stocks', None, 'definition_break',
                       'Lease-stock coverage changes in October 2016.', 'retain_values_flag_transition')
        # Label-based extraction rather than fragile row numbers; raw tables remain available.
        result['source_locator'] = result.feature + '@' + result.observation_date.dt.strftime('%Y-%m-%d')
        result = self.unique(result, ['feature', 'release_date', 'observation_date'], 'eia_archive')
        if not (result.observation_date < result.release_date).all():
            raise ValueError('EIA observation must precede report publication')
        if not (result.release_date < result.available_date).all():
            raise ValueError('EIA availability must follow publication')
        for feature, f in result.groupby('feature'):
            self.describe('EIA_WPSR_ARCHIVE', feature, feature, f.units.iloc[0],
                'daily_quote_weekly_release' if feature.endswith('_spot') else 'weekly',
                'dated_archive_core', 'report release + 1 US federal business day',
                'No API backfill. Archived files may have been corrected before download. '
                + ('Demand proxy, not observed final consumption.' if 'products_supplied' in feature else ''))
        self.save('core/eia_release_history', result)
        return result

    def cftc(self):
        # Rebuild selected position fields from the original ZIPs instead of relying
        # on the old CSV extractor's silent keep-last duplicate policy.
        columns = ['Market_and_Exchange_Names', 'Report_Date_as_YYYY-MM-DD',
                   'CFTC_Contract_Market_Code', 'Open_Interest_All',
                   'Prod_Merc_Positions_Long_All', 'Prod_Merc_Positions_Short_All',
                   'Swap_Positions_Long_All', 'Swap__Positions_Short_All', 'Swap__Positions_Spread_All',
                   'M_Money_Positions_Long_All', 'M_Money_Positions_Short_All', 'M_Money_Positions_Spread_All',
                   'Other_Rept_Positions_Long_All', 'Other_Rept_Positions_Short_All',
                   'Other_Rept_Positions_Spread_All', 'Tot_Rept_Positions_Long_All',
                   'Tot_Rept_Positions_Short_All', 'NonRept_Positions_Long_All', 'NonRept_Positions_Short_All']
        frames = []
        for r in self.base['records']:
            if not r['file'].endswith('.zip'):
                continue
            with zipfile.ZipFile(ROOT / r['file']) as archive:
                members = [p for p in archive.namelist() if p.lower().endswith('.txt')]
                if len(members) != 1:
                    raise ValueError('Ambiguous CFTC ZIP')
                with archive.open(members[0]) as stream:
                    for chunk in pd.read_csv(stream, dtype='string', usecols=columns,
                                             keep_default_na=False, chunksize=30000):
                        codes = chunk.CFTC_Contract_Market_Code.str.strip().str.zfill(6)
                        f = chunk.loc[codes.eq('067651')].copy()
                        f['source_file'] = Path(r['file']).as_posix()
                        f['zip_member'], f['source_row'] = members[0], f.index + 2
                        frames.append(f)
        raw = pd.concat(frames, ignore_index=True)
        raw = raw.rename(columns={c: re.sub('_+', '_', c.lower()) for c in columns})
        raw = raw.rename(columns={'report_date_as_yyyy-mm-dd': 'observation_date',
                                  'cftc_contract_market_code': 'contract_code'})
        raw['contract_code'] = raw.contract_code.str.strip().str.zfill(6)
        raw['observation_date'] = pd.to_datetime(raw.observation_date, errors='raise')
        numeric = [re.sub('_+', '_', c.lower()) for c in columns[3:]]
        for c in numeric:
            raw[c] = pd.to_numeric(raw[c], errors='raise').astype('Int64')
        duplicate_keys = raw.duplicated(['observation_date', 'contract_code'], keep=False)
        if duplicate_keys.any():
            check = raw.loc[duplicate_keys].groupby(['observation_date', 'contract_code'])[numeric].nunique()
            if check.gt(1).any().any():
                raise ValueError('Conflicting original CFTC ZIP records')
            self.issue('cftc', '', None, 'identical_duplicate_reports',
                       str(int(raw.duplicated(['observation_date', 'contract_code']).sum())), 'keep_first')
        raw = raw.drop_duplicates(['observation_date', 'contract_code']).sort_values('observation_date')
        raw['release_date'], raw['available_date'] = pd.NaT, pd.NaT
        raw['eligibility'] = 'reference_only'
        raw['observation_predates_launch_day'] = raw.observation_date.lt('2009-09-04')
        raw['availability_basis'] = 'actual_publication_dates_not_verified'
        raw['long_balance_valid'] = (raw.tot_rept_positions_long_all + raw.nonrept_positions_long_all).eq(raw.open_interest_all)
        raw['short_balance_valid'] = (raw.tot_rept_positions_short_all + raw.nonrept_positions_short_all).eq(raw.open_interest_all)
        if not (raw.long_balance_valid & raw.short_balance_valid).all():
            raise ValueError('CFTC open-interest reconciliation failed')
        if raw[numeric].lt(0).any().any():
            raise ValueError('Negative absolute CFTC position count')
        old_record = next(r for r in self.base['records'] if r['file'].endswith('cftc/wti_positions_weekly.csv'))
        old = pd.read_csv(ROOT / old_record['file'], usecols=columns)
        old_dates = pd.to_datetime(old['Report_Date_as_YYYY-MM-DD'])
        if set(old_dates) != set(raw.observation_date):
            raise ValueError('CFTC original ZIP and earlier extract date sets disagree')
        prior = old.assign(observation_date=old_dates).set_index('observation_date')
        rebuilt = raw.set_index('observation_date')
        for original, clean in zip(columns[3:], numeric):
            if not np.array_equal(pd.to_numeric(prior.loc[rebuilt.index, original]), rebuilt[clean]):
                raise ValueError(f'CFTC earlier extract differs from original ZIP: {original}')
        for c in numeric:
            self.describe('CFTC', '067651', c, 'contracts', 'weekly', 'reference_only',
                          'not assigned', 'All maturities, disaggregated futures only. Historical release delays and pre-launch backcasts prevent direct daily-model use.')
        self.save('reference/cftc_wti_weekly', raw.reset_index(drop=True))
        return raw

    def target(self, fred, archive):
        f = fred.loc[fred.series_id.eq('DCOILWTICO')].copy()
        f = f.rename(columns={'observation_date': 'date', 'value': 'wti_spot_usd_per_barrel'})
        f['is_observed'] = f.wti_spot_usd_per_barrel.notna()
        f['is_nonpositive'] = f.wti_spot_usd_per_barrel.le(0)
        f['role'] = 'outcome_not_a_predictor'
        f['vintage'] = 'latest_local_FRED_snapshot'
        vintages = archive.loc[archive.feature.eq('wti_spot'),
            ['observation_date', 'value', 'release_date', 'available_date', 'source_file', 'source_locator']].copy()
        vintages = vintages.rename(columns={'observation_date': 'date', 'value': 'wti_spot_usd_per_barrel'})
        self.save('targets/wti_spot_release_vintages', vintages)
        # Do not assign a guessed lag to the current-vintage outcome. A matching
        # dated quote supplies evidence; a mismatch stays unverified.
        matched = f[['date', 'wti_spot_usd_per_barrel']].merge(vintages, on='date', suffixes=('_latest', '_archive'))
        same = np.isclose(matched.wti_spot_usd_per_barrel_latest,
                          matched.wti_spot_usd_per_barrel_archive, atol=1e-9, rtol=0, equal_nan=False)
        matched = matched.loc[same].sort_values(['available_date', 'source_file']).drop_duplicates('date')
        matched = matched[['date', 'available_date', 'source_file']].rename(columns={
            'available_date': 'target_verified_available_date', 'source_file': 'target_publication_source_file'})
        f = f.merge(matched, on='date', how='left', validate='one_to_one')
        f['target_publication_evidence'] = np.where(f.target_verified_available_date.notna(),
                                                   'matching_dated_archive', 'unverified')
        f = f[['date', 'wti_spot_usd_per_barrel', 'is_observed', 'is_nonpositive',
               'role', 'vintage', 'target_verified_available_date', 'target_publication_evidence',
               'target_publication_source_file', 'source_file', 'source_row', 'quality_flag']]
        f = f.loc[f.date.le(self.as_of)].sort_values('date').reset_index(drop=True)
        self.save('targets/wti_spot_daily', f)
        for row in f.loc[f.is_nonpositive].itertuples():
            self.issue('target', 'wti_spot', row.date, 'nonpositive_price',
                       str(row.wti_spot_usd_per_barrel), 'retain_no_log_transform', 'info')
        observed = f.loc[f.is_observed].copy()
        observed['previous_observed_price'] = observed.wti_spot_usd_per_barrel.shift()
        observed['previous_observed_date'] = observed.date.shift()
        observed['price_change_usd_per_barrel'] = observed.wti_spot_usd_per_barrel.diff()
        observed['large_absolute_move'] = observed.price_change_usd_per_barrel.abs().ge(20)
        observed['action'] = 'retain_original_value_no_winsorization'
        self.save('audit/target_extreme_observations', observed.loc[observed.is_nonpositive | observed.large_absolute_move], parquet=False)
        return f

    def reconcile(self, archive, fred, api):
        comparisons = []
        for feature, current in [('wti_spot', fred.loc[fred.series_id.eq('DCOILWTICO')]),
                                 ('brent_spot', fred.loc[fred.series_id.eq('DCOILBRENTEU')])]:
            comparisons.append((feature, current))
        comparisons.extend((feature, group) for feature, group in api.groupby('feature')
                           if feature in set(archive.feature))
        records = []
        for feature, current in comparisons:
            old = archive.loc[archive.feature.eq(feature),
                               ['observation_date', 'release_date', 'value', 'units', 'source_file']]
            compare = old.merge(current[['observation_date', 'value', 'source_file']],
                                on='observation_date', how='left', suffixes=('_archive', '_latest'), validate='many_to_one')
            compare['feature'] = feature
            compare['difference_latest_minus_archive'] = compare.value_latest - compare.value_archive
            compare['differs_beyond_rounding'] = compare.difference_latest_minus_archive.abs().gt(.000001)
            compare['action'] = 'retain_separate_vintages_never_overwrite_archive'
            records.append(compare)
        result = pd.concat(records, ignore_index=True)
        self.save('audit/vintage_comparison', result)
        return result

    def panels(self, archive, auxiliary, target):
        start = archive.available_date.min()
        calendar = pd.DataFrame({'forecast_date': pd.bdate_range(start, self.as_of)})
        # Weekdays are an explicit research grid, not an exchange calendar. Missing
        # target dates/holidays remain rows with unavailable outcomes.
        target_lookup = target[['date', 'wti_spot_usd_per_barrel', 'target_verified_available_date',
                                'target_publication_evidence', 'target_publication_source_file']].assign(target_in_source=True)
        base = calendar.merge(target_lookup,
                              left_on='forecast_date', right_on='date', how='left', validate='one_to_one').drop(columns='date')
        base = base.rename(columns={'wti_spot_usd_per_barrel': 'y_wti_spot_usd_per_barrel'})
        base['target_observed'] = base.y_wti_spot_usd_per_barrel.notna()
        base['target_status'] = np.select([base.target_observed, base.target_in_source.fillna(False).astype(bool)],
                                          ['observed', 'missing_in_source'], default='not_in_target_snapshot')
        base = base.drop(columns='target_in_source')
        core, extended, audit = base.copy(), base.copy(), []
        for block, group, max_age in [('core', archive, 21), ('auxiliary', auxiliary, 7)]:
            for feature, frame in group.groupby('feature', sort=True):
                frame = frame.loc[frame.available_date.le(self.as_of)]
                aligned = align_feature(frame, calendar, max_age)
                column = 'x_' + feature
                extended[column] = aligned.value.to_numpy()
                if block == 'core':
                    core[column] = aligned.value.to_numpy()
                aligned['feature'], aligned['block'], aligned['units'] = feature, block, frame.units.iloc[0]
                aligned['eligibility'] = frame.eligibility.iloc[0]
                aligned['availability_basis'] = frame.availability_basis.iloc[0]
                audit.append(aligned)
        for panel in (core, extended):
            cols = [c for c in panel if c.startswith('x_')]
            panel['available_feature_count'] = panel[cols].notna().sum(axis=1)
            panel['all_features_available'] = panel[cols].notna().all(axis=1)
        alignment = pd.concat(audit, ignore_index=True)
        valid = alignment.available_date.notna()
        if not alignment.loc[valid, 'available_date'].le(alignment.loc[valid, 'forecast_date']).all():
            raise ValueError('Future data entered daily alignment')
        self.save('daily/core', core)
        self.save('daily/extended', extended)
        self.save('audit/daily_alignment', alignment, csv=False)
        return core, extended, alignment

    def quality(self, datasets, panels):
        rows = []
        for name, data in datasets.items():
            for feature, f in data.groupby('feature'):
                observed = f.loc[f.value.notna()]
                rows.append({'dataset': name, 'feature': feature, 'rows': len(f),
                    'observed': len(observed), 'missing': int(f.value.isna().sum()),
                    'missing_fraction': float(f.value.isna().mean()), 'zero_count': int(f.value.eq(0).sum()),
                    'negative_count': int(f.value.lt(0).sum()), 'min': f.value.min(), 'max': f.value.max(),
                    'first_date': iso(observed.observation_date.min()),
                    'last_date': iso(observed.observation_date.max()),
                    'max_observed_gap_days': observed.observation_date.drop_duplicates().sort_values().diff().dt.days.max(),
                    'units': f.units.iloc[0]})
        self.save('audit/source_quality', pd.DataFrame(rows), parquet=False)
        coverage = []
        for name, panel in panels.items():
            for c in [c for c in panel if c.startswith(('x_', 'y_'))]:
                available = panel[c].notna()
                runs = (~available).groupby(available.cumsum()).sum()
                coverage.append({'dataset': name, 'column': c, 'rows': len(panel),
                    'available': int(available.sum()), 'missing': int((~available).sum()),
                    'missing_fraction': float((~available).mean()),
                    'max_consecutive_missing_grid_days': int(runs.max()),
                    'first_available_date': iso(panel.loc[available, 'forecast_date'].min()),
                    'last_available_date': iso(panel.loc[available, 'forecast_date'].max())})
        self.save('audit/daily_coverage', pd.DataFrame(coverage), parquet=False)

    def run(self):
        self.out.mkdir(parents=True, exist_ok=True)
        build_workspace(self.out).mkdir(parents=True, exist_ok=True)
        status_path = output_path(self.out, 'build_status.json')
        status_path.write_text(json.dumps({'status': 'running', 'as_of_date': iso(self.as_of)}) + '\n', encoding='utf-8')
        output_path(self.out, 'audit/validation_report.json').parent.mkdir(parents=True, exist_ok=True)
        (output_path(self.out, 'audit/validation_report.json')).write_text(
            json.dumps({'status': 'pending', 'reason': 'new build requires independent validation'}) + '\n', encoding='utf-8')
        print('Verified source files:', len(self.inventory), flush=True)
        fred = self.fred()
        yahoo, yahoo_aux = self.yahoo()
        api = self.api()
        archive = self.archive()
        cftc = self.cftc()
        target = self.target(fred, archive)
        auxiliary = pd.concat([fred.loc[fred.eligibility.eq('market_assumption')], yahoo_aux], ignore_index=True)
        auxiliary = auxiliary.loc[auxiliary.value.notna()].sort_values(['feature', 'observation_date']).reset_index(drop=True)
        self.save('auxiliary/market_observations', auxiliary)
        comparison = self.reconcile(archive, fred, api)
        core, extended, alignment = self.panels(archive, auxiliary, target)
        yahoo_quality = []
        for field in ['open', 'high', 'low', 'close', 'adj_close', 'volume']:
            q = yahoo[['feature', 'observation_date', 'units', field]].rename(columns={field: 'value'}).copy()
            q['feature'] = q.feature + ':' + field
            if field == 'volume':
                q['units'] = 'vendor_reported_volume_not_comparable_between_symbols'
            yahoo_quality.append(q)
        cftc_metrics = [c for c in cftc if c.endswith('_all')]
        cq = cftc.melt(id_vars='observation_date', value_vars=cftc_metrics, var_name='feature', value_name='value')
        cq['units'] = 'contracts'
        self.quality({'fred': fred, 'yahoo': pd.concat(yahoo_quality), 'cftc_reference': cq,
                      'eia_api_reference': api, 'eia_archive': archive, 'market_auxiliary': auxiliary},
                     {'core': core, 'extended': extended})
        self.save('metadata/series_dictionary', pd.DataFrame(self.dictionary), parquet=False)
        self.save('audit/source_inventory', pd.DataFrame(self.inventory.values()).sort_values('source_file'), parquet=False)
        self.save('audit/issues', pd.DataFrame(self.issues, columns=['severity', 'dataset', 'feature',
            'observation_date', 'issue', 'detail', 'action']), parquet=False)
        self.save('audit/quarantine', pd.DataFrame(self.quarantine, columns=['dataset', 'feature',
            'observation_date', 'source_file', 'source_row', 'reason', 'action']), parquet=False)
        self.save('metadata/column_dictionary', pd.DataFrame(self.schemas), parquet=False)
        summary = {
            'as_of_date': iso(self.as_of), 'source_freeze_date': self.freeze_date,
            'forecast_origin': '00:00 America/New_York on forecast_date',
            'target': 'WTI Cushing spot, USD/barrel, latest-local-vintage evaluation outcome',
            'target_observed_rows': int(target.is_observed.sum()),
            'target_first': iso(target.loc[target.is_observed, 'date'].min()),
            'target_last': iso(target.loc[target.is_observed, 'date'].max()),
            'target_nonpositive_rows': int(target.is_nonpositive.sum()),
            'target_rows_with_verified_publication': int(target.target_verified_available_date.notna().sum()),
            'daily_start': iso(core.forecast_date.min()), 'daily_end': iso(core.forecast_date.max()),
            'daily_grid_rows': len(core), 'daily_observed_targets': int(core.target_observed.sum()),
            'core_features': sum(c.startswith('x_') for c in core),
            'extended_features': sum(c.startswith('x_') for c in extended),
            'core_complete_labeled_rows': int((core.all_features_available & core.target_observed).sum()),
            'extended_complete_labeled_rows': int((extended.all_features_available & extended.target_observed).sum()),
            'reference_cftc_rows': len(cftc), 'quarantined_records_or_reports': len(self.quarantine),
            'vintage_comparisons': len(comparison),
            'vintage_differences': int(comparison.differs_beyond_rounding.sum()),
            'stale_feature_cells_masked': int(alignment.is_stale.sum()),
            'python': platform.python_version(), 'pandas': pd.__version__, 'pyarrow': pyarrow.__version__,
            'policies': {'core_max_age_calendar_days': 21, 'auxiliary_max_age_calendar_days': 7,
                'target_imputation': False, 'predictor_backward_fill': False,
                'interpolation': False, 'winsorization': False, 'standardization': False,
                'log_transform': False, 'train_test_split': None, 'horizon_labels_created': False,
                'complete_case_rows_dropped': False, 'network_used': False},
        }
        self.out.mkdir(parents=True, exist_ok=True)
        (output_path(self.out, 'audit/quality_summary.json')).write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        manifest = {'built_at_utc': datetime.now(UTC).isoformat(), 'summary': summary,
                    'quality_summary_sha256': digest(output_path(self.out, 'audit/quality_summary.json')),
                    'inputs': list(self.inventory.values()), 'outputs': self.outputs}
        (output_path(self.out, 'metadata/build_manifest.json')).write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        # A second hash pass proves this run did not alter original inputs.
        for source, r in self.inventory.items():
            if digest(ROOT / source) != r['sha256']:
                raise ValueError('Raw source changed during processing: ' + source)
        status_path.write_text(json.dumps({'status': 'complete', 'as_of_date': iso(self.as_of),
                                          'validation': 'run scripts/validate_prepared.py for independent checks'}) + '\n', encoding='utf-8')
        print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--as-of', help='Override forecast grid end, never beyond the local snapshot date')
    args = parser.parse_args()
    Builder(args.output_dir, args.as_of).run()


if __name__ == '__main__':
    main()

"""Independent integrity, lineage, leakage and rerun tests for processed data."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import unittest
from pathlib import Path
from data_layout import output_path

import numpy as np
import pandas as pd

from prepare_data import Builder, DEFAULT_OUT, ROOT, align_feature, digest, publication_events, shifted


def events(observations, available, values):
    dates = pd.to_datetime(observations)
    availability = pd.to_datetime(available)
    return pd.DataFrame({'observation_date': dates, 'release_date': availability - pd.Timedelta(days=1),
                         'available_date': availability, 'value': values,
                         'source_file': 'synthetic_fixture', 'source_row': 2, 'quality_flag': ''})


class TemporalTests(unittest.TestCase):
    def test_no_backfill_and_exact_availability(self):
        f = events(['2020-01-01'], ['2020-01-03'], [10.])
        grid = pd.DataFrame({'forecast_date': pd.to_datetime(['2020-01-02', '2020-01-03'])})
        a = align_feature(f, grid, 7)
        self.assertTrue(pd.isna(a.value.iloc[0]))
        self.assertEqual(a.value.iloc[1], 10.)

    def test_future_mutation_cannot_change_past(self):
        f = events(['2020-01-01', '2020-01-06'], ['2020-01-03', '2020-01-08'], [10., 20.])
        grid = pd.DataFrame({'forecast_date': pd.date_range('2020-01-01', '2020-01-07')})
        before = align_feature(f, grid, 7)
        f.loc[1, 'value'] = -1e12
        pd.testing.assert_frame_equal(before, align_feature(f, grid, 7))

    def test_delayed_old_report_cannot_rewind_latest(self):
        f = events(['2020-01-03', '2019-12-27'], ['2020-01-08', '2020-01-09'], [20., 5.])
        selected = publication_events(f)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.value.iloc[0], 20.)

    def test_two_observation_weeks_released_together(self):
        f = events(['2022-06-17', '2022-06-24'], ['2022-06-30', '2022-06-30'], [100., 101.])
        selected = publication_events(f)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.value.iloc[0], 101.)

    def test_same_observation_later_revision_only_after_release(self):
        f = events(['2020-01-01', '2020-01-01'], ['2020-01-03', '2020-01-06'], [10., 12.])
        grid = pd.DataFrame({'forecast_date': pd.to_datetime(['2020-01-03', '2020-01-06'])})
        self.assertEqual(align_feature(f, grid, 7).value.tolist(), [10., 12.])

    def test_staleness_boundary_and_negative_price(self):
        f = events(['2020-04-20'], ['2020-04-22'], [-36.98])
        grid = pd.DataFrame({'forecast_date': pd.to_datetime(['2020-05-11', '2020-05-12'])})
        a = align_feature(f, grid, 21)
        self.assertEqual(a.value.iloc[0], -36.98)
        self.assertTrue(pd.isna(a.value.iloc[1]))
        self.assertEqual(a.last_known_value.iloc[1], -36.98)

    def test_equal_time_conflict_rejected(self):
        f = events(['2020-01-01', '2020-01-01'], ['2020-01-03', '2020-01-03'], [1., 2.])
        with self.assertRaises(ValueError):
            publication_events(f)

    def test_missing_does_not_become_zero(self):
        builder = Builder.__new__(Builder)
        builder.quarantine = []
        raw = pd.Series(['.', '', '0', '-36.98', 'not_a_number', 'inf'])
        dates = pd.Series(pd.date_range('2020-01-01', periods=len(raw)))
        values, missing, invalid = builder.number(raw, 'test', 'test', 'fixture', dates)
        self.assertEqual(missing.tolist(), [True, True, False, False, False, False])
        self.assertEqual(invalid.tolist(), [False, False, False, False, True, True])
        self.assertEqual(values.iloc[2], 0.)
        self.assertEqual(values.iloc[3], -36.98)
        self.assertTrue(values.iloc[[0, 1, 4, 5]].isna().all())

    def test_holiday_and_weekend_delay(self):
        result = shifted(pd.Series(pd.to_datetime(['2025-12-24', '2025-12-26'])), 2)
        self.assertEqual(result.dt.strftime('%Y-%m-%d').tolist(), ['2025-12-29', '2025-12-30'])

    def test_empty_events_remain_missing(self):
        f = events([], [], [])
        grid = pd.DataFrame({'forecast_date': pd.date_range('2020-01-01', periods=2)})
        self.assertTrue(align_feature(f, grid, 7).value.isna().all())


def check_roundtrip(path):
    p = pd.read_parquet(path)
    c = pd.read_csv(path.with_suffix('.csv'), dtype={name: 'string' for name in p
                    if pd.api.types.is_string_dtype(p[name].dtype)}, keep_default_na=True)
    assert list(p.columns) == list(c.columns) and len(p) == len(c), str(path)
    for column in p:
        if pd.api.types.is_datetime64_any_dtype(p[column]):
            actual = pd.to_datetime(c[column]).astype('datetime64[ns]')
            expected = p[column].astype('datetime64[ns]')
            pd.testing.assert_series_equal(expected, actual, check_names=False)
        elif pd.api.types.is_numeric_dtype(p[column]) and not pd.api.types.is_bool_dtype(p[column]):
            np.testing.assert_allclose(p[column].to_numpy(dtype=float, na_value=np.nan),
                                       pd.to_numeric(c[column]).to_numpy(dtype=float, na_value=np.nan),
                                       equal_nan=True, rtol=1e-12, atol=1e-12)
        else:
            expected = p[column].astype('string').fillna('')
            actual = c[column].astype('string').fillna('')
            pd.testing.assert_series_equal(expected, actual, check_names=False)


def validate(out, rerun):
    tests = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TemporalTests))
    if not tests.wasSuccessful():
        raise AssertionError('Temporal unit tests failed')
    status = json.loads((output_path(out, 'build_status.json')).read_text())
    assert status['status'] == 'complete'
    manifest = json.loads((output_path(out, 'metadata/build_manifest.json')).read_text())
    assert digest(output_path(out, 'audit/quality_summary.json')) == manifest['quality_summary_sha256']
    assert all(digest(ROOT / r['source_file']) == r['sha256'] for r in manifest['inputs'])
    for r in manifest['outputs']:
        assert digest(out / r['file']) == r['sha256'], r['file']
    assert manifest['summary']['core_features'] == 24
    assert manifest['summary']['extended_features'] == 32
    core = pd.read_parquet(output_path(out, 'daily/core.parquet'))
    extended = pd.read_parquet(output_path(out, 'daily/extended.parquet'))
    target = pd.read_parquet(output_path(out, 'targets/wti_spot_daily.parquet'))
    archive = pd.read_parquet(output_path(out, 'core/eia_release_history.parquet'))
    auxiliary = pd.read_parquet(output_path(out, 'auxiliary/market_observations.parquet'))
    audit = pd.read_parquet(output_path(out, 'audit/daily_alignment.parquet'))
    fred = pd.read_parquet(output_path(out, 'normalized/fred_daily.parquet'))
    api = pd.read_parquet(output_path(out, 'reference/eia_api_weekly.parquet'))
    cftc = pd.read_parquet(output_path(out, 'reference/cftc_wti_weekly.parquet'))
    assert not core.forecast_date.duplicated().any() and core.forecast_date.is_monotonic_increasing
    assert not target.date.duplicated().any()
    assert core.forecast_date.dt.dayofweek.lt(5).all()
    assert not core.filter(regex='^x_').columns.isin(['x_wti_spot_latest', 'x_broad_trade_weighted_dollar']).any()
    assert not any('futures' in c or 'etf' in c for c in extended.filter(regex='^x_'))
    assert api.available_date.isna().all() and cftc.available_date.isna().all()
    assert set(archive.eligibility) == {'dated_archive_core'}
    assert set(auxiliary.eligibility) == {'market_assumption'}
    assert archive.release_date.lt(archive.available_date).all()
    assert archive.observation_date.lt(archive.release_date).all()
    assert not audit.duplicated(['forecast_date', 'feature']).any()
    known = audit.available_date.notna()
    assert audit.loc[known, 'available_date'].le(audit.loc[known, 'forecast_date']).all()
    assert audit.loc[audit.is_stale, 'value'].isna().all()
    assert audit.loc[~known, 'value'].isna().all()
    assert audit.loc[audit.value.notna(), 'age_days'].le(audit.loc[audit.value.notna(), 'max_age_days']).all()
    assert core.target_observed.equals(core.y_wti_spot_usd_per_barrel.notna())
    assert core.loc[core.target_status.ne('observed'), 'y_wti_spot_usd_per_barrel'].isna().all()
    shared = [c for c in core if c not in ('available_feature_count', 'all_features_available')]
    pd.testing.assert_frame_equal(core[shared], extended[shared])
    gold = fred.loc[fred.series_id.eq('DCOILWTICO')].set_index('observation_date').value
    np.testing.assert_allclose(target.wti_spot_usd_per_barrel, gold.loc[target.date], equal_nan=True)
    expected_target = target.set_index('date').wti_spot_usd_per_barrel.reindex(core.forecast_date)
    np.testing.assert_allclose(core.y_wti_spot_usd_per_barrel, expected_target, equal_nan=True)
    negative = target.loc[target.date.eq('2020-04-20'), 'wti_spot_usd_per_barrel']
    assert len(negative) == 1 and negative.iloc[0] == -36.98
    assert core.loc[core.forecast_date.eq('2020-04-20'), 'y_wti_spot_usd_per_barrel'].iloc[0] == -36.98
    target_known = target.target_verified_available_date.notna()
    assert target.loc[target_known, 'target_verified_available_date'].gt(target.loc[target_known, 'date']).all()
    target_vintages = pd.read_parquet(output_path(out, 'targets/wti_spot_release_vintages.parquet'))
    evidence = target.loc[target_known].merge(target_vintages, on='date', suffixes=('_latest', '_archive'))
    same_value = np.isclose(evidence.wti_spot_usd_per_barrel_latest, evidence.wti_spot_usd_per_barrel_archive,
                           atol=1e-9, rtol=0)
    evidence = evidence.loc[same_value].groupby('date').available_date.min()
    assert len(evidence) == target_known.sum()
    assert (target.loc[target_known].set_index('date').target_verified_available_date.sort_index() == evidence.sort_index()).all()
    for day, release in [('2022-06-17', '2022-06-29'), ('2023-11-03', '2023-11-15')]:
        rows = archive.loc[archive.observation_date.eq(day) & ~archive.feature.str.endswith('_spot')]
        assert len(rows) == 15 and rows.release_date.eq(release).all()
    assert not archive.source_file.str.contains('2017-02-23|2026-06-24|2026-10-07').any()
    # Unit conversion checked against raw source, not a second copy of derived data.
    row = api.loc[api.feature.eq('commercial_crude_stocks')].iloc[0]
    original = pd.read_csv(ROOT / row.source_file)
    assert row.value == float(original.iloc[int(row.source_row) - 2].value) / 1000
    assert set(cftc.contract_code) == {'067651'}
    assert (cftc.long_balance_valid & cftc.short_balance_valid).all()
    # Independent brute-force as-of selection, not a second merge_asof call.
    observations = pd.concat([archive, auxiliary], ignore_index=True)
    groups = {name: f for name, f in observations.groupby('feature')}
    rng = np.random.default_rng(5210)
    sample = audit.iloc[rng.choice(len(audit), size=512, replace=False)]
    for row in sample.itertuples():
        candidates = groups[row.feature]
        candidates = candidates.loc[candidates.available_date.le(row.forecast_date) & candidates.value.notna()]
        if candidates.empty:
            assert pd.isna(row.value)
            continue
        candidates = candidates.loc[candidates.observation_date.eq(candidates.observation_date.max())]
        selected = candidates.sort_values(['available_date', 'release_date', 'source_file']).iloc[-1]
        assert row.observation_date == selected.observation_date
        assert row.available_date == selected.available_date
        assert row.last_known_value == selected.value
        expected = np.nan if (row.forecast_date - selected.observation_date).days > row.max_age_days else selected.value
        assert (pd.isna(expected) and pd.isna(row.value)) or expected == row.value
    # The wide matrices must exactly match their independently auditable long table.
    for name, panel, blocks in [('core', core, ['core']), ('extended', extended, ['core', 'auxiliary'])]:
        wide = audit.loc[audit.block.isin(blocks)].pivot(index='forecast_date', columns='feature', values='value')
        for feature in wide:
            np.testing.assert_allclose(panel['x_' + feature], wide[feature].reindex(panel.forecast_date), equal_nan=True)
    pairs = [out / r['file'] for r in manifest['outputs'] if r['file'].endswith('.parquet')
             and (out / r['file']).with_suffix('.csv').exists()]
    for path in pairs:
        check_roundtrip(path)
    deterministic = None
    if rerun:
        subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_data.py'), '--output-dir', str(out),
                        '--as-of', manifest['summary']['as_of_date']], check=True, capture_output=True, text=True)
        newer = json.loads((output_path(out, 'metadata/build_manifest.json')).read_text())
        before = {r['file']: r['sha256'] for r in manifest['outputs']}
        after = {r['file']: r['sha256'] for r in newer['outputs']}
        assert before == after, 'Identical inputs produced differing data outputs'
        deterministic = True
    result = {'status': 'passed', 'temporal_unit_tests': tests.testsRun,
              'raw_input_hashes_verified': len(manifest['inputs']),
              'output_hashes_verified': len(manifest['outputs']),
              'csv_parquet_pairs_verified': len(pairs), 'independent_asof_samples': len(sample),
              'deterministic_rerun_verified': deterministic,
              'as_of_date': manifest['summary']['as_of_date'],
              'scope': 'Data integrity and temporal alignment; not a guarantee against provider archive corrections.'}
    (output_path(out, 'audit/validation_report.json')).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--check-rerun', action='store_true')
    args = parser.parse_args()
    validate(args.output_dir.resolve(), args.check_rerun)

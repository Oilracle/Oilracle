"""Verify portable submission data without fetching data or fitting models."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_layout import INPUTS, output_path
from prepare_data import Builder
from export_team_data import build_publication, build_table
from m3_ml.common.news import news_availability, news_features, news_metadata
from m3_ml.common.walkforward import forecast_origins


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pair(csv, parquet):
    left = pd.read_csv(csv)
    right = pd.read_parquet(parquet)
    for column in right:
        if pd.api.types.is_datetime64_any_dtype(right[column]):
            left[column] = pd.to_datetime(left[column])
    pd.testing.assert_frame_equal(left, right, check_dtype=False, rtol=1e-12, atol=1e-12)
    assert not right.date.duplicated().any() and right.date.is_monotonic_increasing
    assert not right.isna().any().any()
    return right


def inventory():
    records = []
    for path in sorted((ROOT / 'data').rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[1] == 'metadata' or any(p.startswith('.') for p in relative.parts):
            continue
        category = relative.parts[1] if len(relative.parts) > 2 else 'documentation'
        records.append({'path': relative.as_posix(), 'category': category,
                        'bytes': path.stat().st_size, 'sha256': digest(path)})
    return records


def audit(rebuilt=None):
    table = pair(ROOT / 'data/processed/wti_daily.csv', ROOT / 'data/processed/wti_daily.parquet')
    assert len(table) == 3803 and table.date.min() == pd.Timestamp('2011-08-04')
    assert table.date.max() == pd.Timestamp('2026-10-23')
    features = pd.read_csv(ROOT / 'data/processed/features.csv')
    assert list(features.column) == list(table.columns)
    negative = table.loc[table.date.eq('2020-04-20'), 'target_wti']
    assert len(negative) == 1 and np.isclose(negative.iloc[0], -36.98)
    raw = pd.read_csv(ROOT / 'data/raw/fred/wti_spot_daily.csv', parse_dates=['observation_date'])
    raw = raw.set_index('observation_date').DCOILWTICO
    np.testing.assert_allclose(table.target_wti, raw.reindex(table.date), atol=1e-12)
    publication = pd.read_csv(ROOT / 'data/processed/wti_publication.csv',
                              parse_dates=['date', 'available_date'])
    pd.testing.assert_series_equal(table.date, publication.date)
    np.testing.assert_allclose(table.target_wti, publication.target_wti)
    assert publication.available_date.gt(publication.date).all()
    assert set(publication.evidence) == {'matching_dated_archive', 'assumed_weekly_release'}
    matched = publication.evidence.eq('matching_dated_archive')
    assert publication.loc[matched, 'source_file'].notna().all()
    assert all((ROOT / p).is_file() for p in publication.loc[matched, 'source_file'].unique())

    long = pair(ROOT / 'data/long/wti_daily_long.csv', ROOT / 'data/long/wti_daily_long.parquet')
    assert len(long) == 8268 and long.date.min() == pd.Timestamp('1991-02-15')
    assert long.date.max() == pd.Timestamp('2026-10-06')
    long_features = pd.read_csv(ROOT / 'data/long/features_long.csv')
    assert list(long_features.column) == list(long.columns)
    series = pd.read_csv(ROOT / 'data/time_series/daily_spot.csv', parse_dates=['date'])
    assert not series.date.duplicated().any() and series.date.is_monotonic_increasing
    np.testing.assert_allclose(series.target_wti, raw.reindex(series.date), atol=1e-12)

    news = news_metadata()
    x_news = news_features(table.date)
    origin_counts = {}
    for mode in ('fixed_origin', 'daily_update'):
        origins = forecast_origins(table.date, mode)
        assert np.isfinite(x_news.iloc[origins].to_numpy()).all()
        origin_counts[mode] = len(origins)
    availability = news_availability(table.date)

    assert INPUTS == ROOT / 'data/rebuild', 'Submission still relies on hidden local inputs'
    builder = Builder(ROOT / '.local/submission_audit_unused', None)
    assert builder.inventory, 'No source hashes verified'
    assert all(not p.startswith('.local/') for p in builder.inventory), 'Hidden source dependency'

    corpus = ROOT / 'data/kronos/corpus'
    cm = pd.read_csv(ROOT / 'm3_ml/kronos/corpus_manifest.csv').fillna('')
    corpus_rows = 0
    for row in cm.to_dict('records'):
        if not row['rows'] or int(row['rows']) == 0:
            continue
        path = corpus / (row['ticker'].replace('=', '_').replace('^', '_') + '.parquet')
        assert digest(path) == row['sha256'], f'Corpus hash mismatch: {path.name}'
        frame = pd.read_parquet(path)
        assert len(frame) == int(row['rows'])
        assert not frame.index.has_duplicates and frame.index.is_monotonic_increasing
        corpus_rows += len(frame)
    for stage, cutoff in [(1, '2023-08-31'), (2, '2026-09-30')]:
        folder = ROOT / f'm3_ml/kronos/checkpoints/stage{stage}'
        metadata = json.loads((folder / 'metadata.json').read_text())
        assert metadata['train_end'] == cutoff
        assert metadata['protocol_id'] == 'qf5210-frozen-october-2026'
        for name in ['predictor.pt', 'tokenizer.pt', 'normalization.npz', 'train_log.json']:
            assert (folder / name).stat().st_size > 0

    archive_count = 0
    for path in (ROOT / 'data/rebuild/source_support').rglob('*.zip'):
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            for member in archive.namelist():
                parts = Path(member.replace('\\', '/')).parts
                assert not any(p in {'.git', '.env', '..'} for p in parts)
        archive_count += 1

    records = inventory()
    sizes = Counter()
    counts = Counter()
    for row in records:
        sizes[row['category']] += row['bytes']
        counts[row['category']] += 1
    result = {
        'status': 'passed', 'main_rows': len(table), 'main_features': len(table.columns) - 2,
        'main_start': '2011-08-04', 'main_end': '2026-10-23', 'negative_price_preserved': True,
        'csv_parquet_pairs': 2, 'long_rows': len(long), 'time_series_rows': len(series),
        'publication_evidence_counts': publication.evidence.value_counts().to_dict(),
        'news_policy_id': news['news_policy_id'], 'news_sources_verified': len(news['sources']),
        'news_availability_rows': len(availability), 'complete_news_origins': origin_counts,
        'raw_inputs_verified': len(builder.inventory),
        'portable_corpus_files': len(list(corpus.glob('*.parquet'))), 'corpus_rows': corpus_rows,
        'checkpoint_stages': 2, 'raw_source_archives_verified': archive_count,
        'file_count_by_category': dict(counts), 'bytes_by_category': dict(sizes),
        'network_used': False, 'model_fitting_performed': False,
        'scope': 'Snapshot integrity and portable dependencies; not a real-time-vintage guarantee.',
    }
    if rebuilt:
        full = pd.read_parquet(output_path(rebuilt, 'daily/extended.parquet'))
        pd.testing.assert_frame_equal(build_table(full), table, check_dtype=False, rtol=1e-12, atol=1e-12)
        actual = build_publication(table, full).fillna('')
        expected = publication.fillna('')
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
        report = json.loads(output_path(rebuilt, 'audit/validation_report.json').read_text())
        assert report['status'] == 'passed' and report['deterministic_rerun_verified']
        result['independent_rebuild'] = report
        result['reproduced_main_table_and_publication_schedule'] = True
    return result, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuilt', type=Path, help='Compare a separately rebuilt and validated data release')
    parser.add_argument('--write', action='store_true', help='Save inventory and report to data/metadata')
    args = parser.parse_args()
    report, records = audit(args.rebuilt)
    if args.write:
        out = ROOT / 'data/metadata'
        out.mkdir(exist_ok=True)
        pd.DataFrame(records).to_csv(out / 'file_inventory.csv', index=False, lineterminator='\n')
        (out / 'submission_data_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

"""Validate the shared data package offline, without raw data or credentials.

--write-manifest is for maintainers after a successful full local validation.
Default mode is read-only and suitable for a fresh team checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from data_layout import output_path, PROCESSED, SHARED_FILES

ROOT = Path(__file__).resolve().parents[1]
OUT = PROCESSED
MANIFEST = output_path(OUT, 'metadata/release_manifest.json')
FILES = SHARED_FILES


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records():
    return [{'file': name, 'bytes': (OUT / name).stat().st_size,
             'sha256': digest(OUT / name)} for name in FILES]


def inspect_tables():
    summary = json.loads((output_path(OUT, 'audit/quality_summary.json')).read_text(encoding='utf-8'))
    report = json.loads((output_path(OUT, 'audit/validation_report.json')).read_text(encoding='utf-8'))
    require(report['status'] == 'passed', 'Full local validation has not passed')
    require(report['deterministic_rerun_verified'] is True, 'Deterministic rebuild not verified')
    require(report['as_of_date'] == summary['as_of_date'], 'Mismatched snapshot dates')
    frames = {}
    for name in FILES:
        if not name.endswith('.parquet') or name.replace('.parquet', '.csv') not in FILES:
            continue
        frame = pd.read_parquet(OUT / name)
        csv = pd.read_csv(OUT / name.replace('.parquet', '.csv'),
                          dtype={c: 'string' for c in frame
                                 if pd.api.types.is_string_dtype(frame[c].dtype)})
        require(list(frame.columns) == list(csv.columns), f'Column mismatch: {name}')
        # CSV represents both empty strings and null strings as empty fields.
        # Parquet is authoritative for that distinction; numeric nulls stay null.
        for column in frame:
            dtype = frame[column].dtype
            expected = frame[column]
            if pd.api.types.is_datetime64_any_dtype(dtype):
                actual = pd.to_datetime(csv[column]).astype(dtype)
            elif pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
                actual = csv[column].astype(dtype)
            else:
                actual = csv[column].astype('string').fillna('')
                expected = expected.astype('string').fillna('')
            pd.testing.assert_series_equal(expected, actual, check_exact=False, rtol=1e-12, atol=1e-12)
        frames[name] = frame
    core, extended = frames['forecasting/wti_spot_core_daily.parquet'], frames['forecasting/wti_spot_extended_daily.parquet']
    for name, frame in [('core', core), ('extended', extended)]:
        dates = frame['forecast_date']
        require(len(frame) == summary['daily_grid_rows'], f'{name}: wrong row count')
        require(dates.notna().all() and dates.is_unique and dates.is_monotonic_increasing,
                f'{name}: invalid date order')
        require(str(dates.min().date()) == summary['daily_start'] and
                str(dates.max().date()) == summary['daily_end'], f'{name}: wrong coverage')
        require(sum(c.startswith('x_') for c in frame) == summary[f'{name}_features'],
                f'{name}: wrong feature count')
        require('y_wti_spot_usd_per_barrel' in frame, f'{name}: missing WTI target')
        features = frame.filter(regex='^x_')
        require(frame['available_feature_count'].eq(features.notna().sum(axis=1)).all(),
                f'{name}: incorrect availability counts')
        require(frame['all_features_available'].eq(features.notna().all(axis=1)).all(),
                f'{name}: incorrect completeness flags')
    shared = core.columns.difference(['available_feature_count', 'all_features_available'])
    pd.testing.assert_frame_equal(core[shared], extended[shared])
    return summary, len(frames)


def verify_local_build():
    """Reject manifest publication if local inputs/outputs changed after the build."""
    build = json.loads((output_path(OUT, 'metadata/build_manifest.json')).read_text(encoding='utf-8'))
    status = json.loads((output_path(OUT, 'build_status.json')).read_text(encoding='utf-8'))
    require(status['status'] == 'complete', 'Local build incomplete')
    require(digest(output_path(OUT, 'audit/quality_summary.json')) == build['quality_summary_sha256'],
            'Local quality summary changed; rebuild and revalidate')
    for record in build['inputs']:
        require(digest(ROOT / record['source_file']) == record['sha256'],
                f"Local input changed: {record['source_file']}; rebuild and revalidate")
    for record in build['outputs']:
        require(digest(OUT / record['file']) == record['sha256'],
                f"Local output changed: {record['file']}; rebuild and revalidate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true')
    args = parser.parse_args()
    if args.write_manifest:
        verify_local_build()
        summary, pairs = inspect_tables()
        manifest = {'schema_version': 1, 'as_of_date': summary['as_of_date'],
                    'scope': 'Shared deliverables only; raw-source validation requires the frozen local snapshot.',
                    'files': records()}
        MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    else:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        require(manifest['schema_version'] == 1, 'Unknown release manifest version')
        require(manifest['files'] == records(), 'Release files differ from their manifest')
        summary, pairs = inspect_tables()
        require(manifest['as_of_date'] == summary['as_of_date'], 'Mismatched release date')
    print(json.dumps({'status': 'passed', 'files_verified': len(FILES),
                      'csv_parquet_pairs_verified': pairs, 'as_of_date': summary['as_of_date'],
                      'raw_sources_revalidated': False, 'manifest_written': args.write_manifest}, indent=2))


if __name__ == '__main__':
    main()

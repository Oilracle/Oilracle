"""Validate downloaded supplement provenance and temporal constraints; no network."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math

import pandas as pd
from data_layout import source_path, relocate_manifest, relocate_reference, MANIFESTS, LOCAL

from download_supplement import OUT, RAW, ROOT, next_business_date, parse_prices, parse_table


def parser_checks():
    # The first two Friday columns are weekly averages, not daily observations.
    # The last five dates also test December/January year inference.
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(['STUB_1', 'STUB_2', 'STUB_3', 'Fri 12/12', 'Fri 12/19',
                     'Mon 12/22', 'Tue 12/23', 'Wed 12/24', 'Thu 12/25', 'Fri 12/26',
                     'Mon 12/29', 'Tue 12/30', 'Wed 12/31', 'Thu 1/1', 'Fri 1/2'])
    writer.writerow(['Crude Oil', 'WTI - Cushing', '', 999, 999, 999, 999, 999,
                     999, 999, 1, 2, 3, '-', 5])
    parsed = parse_prices(buffer.getvalue().encode('cp1252'), '2015-01-07', 'test')
    assert [x['value'] for x in parsed] == [1, 2, 3, 5]
    assert parsed[0]['observation_date'] == '2014-12-29'
    assert parsed[-1]['observation_date'] == '2015-01-02'
    assert all(x['available_date'] == '2015-01-08' for x in parsed)
    assert next_business_date('2025-12-24') == '2025-12-26'
    assert next_business_date('2025-12-26') == '2025-12-29'
    sample = source_path(RAW / 'wpsr' / '2011-08-03' / 'table1.csv').read_bytes()
    short = parse_table(sample, '2011-08-03', 'test')
    long = parse_table(sample.replace(b'7/29/11', b'7/29/2011'), '2011-08-03', 'test')
    assert short == long


def main():
    parser_checks()
    paths = sorted(p for p in (MANIFESTS / 'supplement').glob('manifest_*.json') if '_smoke' not in p.name)
    assert paths, 'Full download manifest missing'
    manifest = relocate_manifest(json.loads(paths[-1].read_text(encoding='utf-8')))
    assert not manifest['archive_errors'], manifest['archive_errors']
    records = manifest['daily_records'] + manifest['outputs']
    for excluded in manifest['archive_exclusions']:
        records.extend(excluded['files'])
    for archive in manifest['archive_records']:
        if 'files' in archive:
            records.extend(archive['files'])
            records.append(archive['report'])
        else:
            records.append(archive)
    for record in records:
        path = ROOT / record['file']
        content = path.read_bytes()
        assert len(content) == record['bytes'], record['file']
        assert hashlib.sha256(content).hexdigest() == record['sha256'], record['file']

    market = pd.read_csv(OUT / 'supplement_daily_market.csv')
    history = pd.read_csv(OUT / 'supplement_eia_release_history.csv')
    for frame in (market, history):
        frame['source_file'] = frame.source_file.map(relocate_reference)
    assert market.feature.nunique() == 3
    assert not market.duplicated(['feature', 'observation_date']).any()
    assert market.loc[market.series_id == 'OVXCLS', 'observation_date'].min() >= '2009-01-01'
    assert (market.available_date > market.observation_date).all()
    assert not history.duplicated(['feature', 'release_date', 'observation_date']).any()
    assert (history.observation_date < history.release_date).all()
    assert (history.release_date < history.available_date).all()
    releases = {a['release_date'] for a in manifest['archive_records'] if 'release_date' in a}
    assert set(history.release_date) == releases
    for excluded in manifest['archive_exclusions']:
        for record in excluded['files']:
            assert not history.source_file.eq(record['file']).any()
    assert history.feature.nunique() == 24
    for frame in (market, history):
        assert frame.value.map(math.isfinite).all()
        assert not frame[['feature', 'value', 'units', 'source_file']].isna().any().any()
        assert frame.source_file.map(lambda s: (ROOT / s).exists()).all()
    weekly = history.loc[~history.feature.str.endswith('_spot')]
    counts = weekly.groupby('source_file').feature.nunique()
    assert counts.eq(15).all(), counts.loc[counts.ne(15)].to_dict()
    assert not weekly.duplicated(['release_date', 'feature', 'observation_date']).any()
    for observation, release in [('2022-06-17', '2022-06-29'), ('2023-11-03', '2023-11-15')]:
        delayed = weekly.loc[weekly.observation_date.eq(observation)]
        assert len(delayed) == 15
        assert delayed.release_date.eq(release).all()

    # Demonstrate backward-only joining: no feature exists before first publication.
    sample = history.loc[history.feature == 'gasoline_stocks'].sort_values('available_date')
    sample = sample.assign(available_date=pd.to_datetime(sample.available_date))
    grid = pd.DataFrame({'forecast_date': pd.date_range('2011-07-25', periods=20)})
    merged = pd.merge_asof(grid, sample, left_on='forecast_date', right_on='available_date')
    known = merged.available_date.notna()
    assert (~known).any() and known.any()
    assert (merged.loc[known, 'available_date'] <= merged.loc[known, 'forecast_date']).all()
    assert merged.loc[~known, 'value'].isna().all()

    summary = {
        'manifest': paths[-1].relative_to(ROOT).as_posix(),
        'checks': ['parser excludes weekly averages', 'year rollover and missing prices',
                   'holiday availability', 'source and output hashes', 'finite numeric values',
                   'unique keys', 'observation before release before availability',
                   '15 weekly features per report', 'backward-only join before first release'],
        'verified_files': len(records), 'dated_reports': len(manifest['archive_records']) - 1,
        'unique_release_dates': len(releases),
        'quarantined_reports': len(manifest['archive_exclusions']),
        'market_rows': len(market), 'eia_rows': len(history),
        'eia_features': int(history.feature.nunique()),
        'first_release': min(releases), 'last_release': max(releases),
        'market_coverage': market.groupby('series_id').observation_date.agg(['min', 'max']).to_dict('index'),
        'eia_coverage': history.groupby('feature').observation_date.agg(['min', 'max']).to_dict('index'),
        'limitations': 'Structural and provenance checks do not prove providers never retrospectively corrected archives.',
    }
    destination = LOCAL / 'checks/supplement_validation.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

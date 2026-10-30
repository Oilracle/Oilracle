"""Public supplementary inputs with explicit information-availability policies.

EIA weekly values come from individual dated WPSR reports, not today's revised
time-series endpoint. Run with uv run python scripts/download_supplement.py.
No credentials are needed. Cache archived reports; retain dated market snapshots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html as html_module
import io
import json
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from data_layout import source_path, MANIFESTS, INTERIM

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw' / 'supplement'
OUT = INTERIM
ARCHIVE_URL = 'https://www.eia.gov/petroleum/supply/weekly/archive/'
TODAY = datetime.now(UTC).date().isoformat()
DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
DAILY = {
    'OVXCLS': ('oil_etf_implied_volatility', 'index', '2009-01-01'),
    'T10YIE': ('us_10y_breakeven_inflation', 'percent', '2003-01-01'),
    'DFII10': ('us_10y_real_yield', 'percent', '2003-01-01'),
}
STOCKS = {
    'Commercial (Excluding SPR)': 'commercial_crude_stocks',
    'Strategic Petroleum Reserve (SPR)': 'spr_crude_stocks',
    'Total Motor Gasoline': 'gasoline_stocks',
    'Distillate Fuel Oil': 'distillate_stocks',
    'Kerosene-Type Jet Fuel': 'jet_fuel_stocks',
    'Total Stocks (Excluding SPR)': 'total_petroleum_stocks_ex_spr',
}
SUPPLY = {
    'Domestic Production': 'crude_production',
    'Net Imports (Including SPR)': 'crude_net_imports',
    'Imports': 'crude_imports',
    'Exports': 'crude_exports',
    'Refinery Inputs': 'crude_refinery_inputs',
    'Crude Oil Input to Refineries': 'crude_refinery_inputs',
}
DEMAND = {
    'Total': 'total_products_supplied',
    'Finished Motor Gasoline': 'gasoline_products_supplied',
    'Kerosene-Type Jet Fuel': 'jet_fuel_products_supplied',
    'Distillate Fuel Oil': 'distillate_products_supplied',
}


class ArchiveDateConflict(ValueError):
    """Quarantine a source whose independent date labels disagree."""


def fetch(url: str) -> bytes:
    curl = shutil.which('curl.exe') or shutil.which('curl')
    if curl is None:
        raise RuntimeError('curl is required for public downloads')
    result = subprocess.run(
        [curl, '-L', '--fail', '--silent', '--show-error', '--retry', '2',
         '--connect-timeout', '15', '--max-time', '60', url],
        capture_output=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(f'Public download failed: curl {result.returncode}, {url}')
    return result.stdout


def cached(url: str, path: Path) -> bytes:
    path = source_path(path)
    if path.exists():
        return path.read_bytes()
    content = fetch(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


def file_record(path: Path, url: str) -> dict:
    path = source_path(path)
    content = path.read_bytes()
    return {'file': path.relative_to(ROOT).as_posix(), 'url': url,
            'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}


def links(html: str, base: str) -> list[str]:
    return list(dict.fromkeys(urljoin(base, u) for u in
                re.findall(r'href=[\"\']([^\"\']+)', html, flags=re.I)))


def date_string(value) -> str:
    return pd.Timestamp(value).date().isoformat()


def next_business_date(value, n=1) -> str:
    return date_string(pd.Timestamp(value) + n * DAY)


def daily_download() -> tuple[pd.DataFrame, list[dict]]:
    frames, records = [], []
    for code, (feature, units, cutoff) in DAILY.items():
        print('Daily market:', code, flush=True)
        url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=' + code
        path = source_path(RAW / 'market' / TODAY / f'{code}.csv')
        content = cached(url, path)
        raw = pd.read_csv(io.BytesIO(content))
        frame = pd.DataFrame({'observation_date': pd.to_datetime(raw['observation_date']),
                              'value': pd.to_numeric(raw[code], errors='coerce')})
        frame = frame.dropna().sort_values('observation_date')
        assert not frame.observation_date.duplicated().any()
        record = file_record(path, url)
        record.update(series=code, feature=feature, units=units,
                      raw_first=date_string(frame.observation_date.min()),
                      raw_last=date_string(frame.observation_date.max()))
        if cutoff:
            frame = frame.loc[frame.observation_date >= cutoff].copy()
        frame = frame.loc[frame.observation_date < TODAY].copy()
        frame['feature'] = feature
        frame['series_id'] = code
        frame['units'] = units
        # Forecast origin is the start of available_date (US Eastern).
        # Extra delay avoids treating a close as information known at that day's open.
        frame['available_date'] = frame.observation_date.map(
            lambda d: next_business_date(d, 2))
        frame['availability_basis'] = 'market_observation_plus_2_US_business_days'
        frame['source_file'] = path.relative_to(ROOT).as_posix()
        frame['observation_date'] = frame.observation_date.map(date_string)
        record.update(usable_rows=len(frame),
                      usable_first=frame.observation_date.min(),
                      usable_last=frame.observation_date.max(),
                      minimum_observation_date=cutoff,
                      caveat='Latest provider history; not a tick-by-tick vintage archive. '
                             'OVX pre-2009 excluded because official launch year was 2008.')
        records.append(record)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), records


def parse_table(content: bytes, release: str, source_file: str) -> list[dict]:
    rows = list(csv.reader(io.StringIO(content.decode('cp1252'))))
    result = []
    section, observation = None, None
    for row in rows:
        if not row:
            continue
        if row[0].strip() == 'STUB_1':
            flow = len(row) > 1 and row[1].strip() == 'STUB_2'
            section = 'flow' if flow else 'stock'
            header_date = row[2 if flow else 1].strip()
            fmt = '%m/%d/%Y' if len(header_date.split('/')[-1]) == 4 else '%m/%d/%y'
            observation = datetime.strptime(header_date, fmt).date().isoformat()
            assert observation < release, (observation, release)
            continue
        feature = None
        if section == 'stock' and row[0].strip() in STOCKS:
            feature = STOCKS[row[0].strip()]
            value, units = row[1], 'million barrels'
        elif section == 'flow' and len(row) >= 3:
            label = re.sub(r'^\s*\(\d+\)\s*', '', row[1]).strip()
            if row[0].strip() == 'Products Supplied':
                feature = DEMAND.get(label)
            elif row[0].strip() == 'Crude Oil Supply':
                feature = SUPPLY.get(label)
            value, units = row[2], 'thousand barrels/day'
        if feature:
            try:
                number = float(value.replace(',', ''))
            except ValueError:
                continue
            result.append({'observation_date': observation, 'release_date': release,
                           'available_date': next_business_date(release),
                           'feature': feature, 'value': number, 'units': units,
                           'availability_basis': 'dated_WPSR_archive_plus_1_US_business_day',
                           'source_file': source_file})
    names = {r['feature'] for r in result}
    required = {'gasoline_stocks', 'distillate_stocks', 'gasoline_products_supplied',
                'distillate_products_supplied'}
    if not required <= names:
        raise ValueError(f'{release}: missing required fields {sorted(required - names)}')
    if len(names) != len(result):
        raise ValueError(f'{release}: duplicate fields')
    return result


def parse_prices(content: bytes, release: str, source_file: str) -> list[dict]:
    """Keep latest week's daily quotes, never weekly averages or monthly history."""
    mapping = {
        ('Crude Oil', 'WTI - Cushing', ''): 'wti_spot',
        ('Crude Oil', 'Brent', ''): 'brent_spot',
        ('Motor Gasoline', 'Conventional', 'New York Harbor'): 'ny_gasoline_spot',
        ('Motor Gasoline', 'Conventional', 'U.S. Gulf Coast'): 'gulf_gasoline_spot',
        ('RBOB Regular', 'Los Angeles', ''): 'la_rbob_spot',
        ('Heating Oils', 'No. 2 Heating Oil', 'New York Harbor'): 'ny_heating_oil_spot',
        ('No. 2 Distillate', 'Ultra-Low Sulfur Diesel Fuel', 'U.S. Gulf Coast'): 'gulf_diesel_spot',
        ('Kerosene-Type Jet Fuel', 'U.S. Gulf Coast', ''): 'gulf_jet_fuel_spot',
        ('Propane', 'Mont Belvieu', ''): 'mont_belvieu_propane_spot',
    }
    result, columns = [], []
    for row in csv.reader(io.StringIO(content.decode('cp1252'))):
        if len(row) < 4:
            continue
        if row[0] == 'STUB_1':
            columns = []
            # Daily section has three stub columns; monthly section has four.
            if row[3] == 'STUB_4':
                continue
            for i in range(max(5, len(row) - 5), len(row)):
                match = re.search(r'(\d{1,2})/(\d{1,2})', row[i])
                if not match:
                    continue
                month, day = map(int, match.groups())
                year = int(release[:4])
                stamp = pd.Timestamp(year=year, month=month, day=day)
                if date_string(stamp) >= release:
                    stamp = stamp.replace(year=year - 1)
                if not 0 < (pd.Timestamp(release) - stamp).days <= 31:
                    raise ValueError(f'{release}: invalid price date {stamp}')
                columns.append((i, date_string(stamp)))
            continue
        key = tuple(x.strip() for x in row[:3])
        feature = mapping.get(key)
        if not feature or not columns:
            continue
        for i, observation in columns:
            try:
                number = float(row[i].replace(',', ''))
            except ValueError:
                continue
            result.append({'observation_date': observation, 'release_date': release,
                           'available_date': next_business_date(release),
                           'feature': feature, 'value': number,
                           'units': 'USD/barrel' if key[0] == 'Crude Oil' else 'USD/gallon',
                           'availability_basis': 'dated_WPSR_archive_plus_1_US_business_day',
                           'source_file': source_file})
    if not result:
        raise ValueError(f'{release}: no recognized daily prices in {source_file}')
    return result


def one_report(url: str) -> tuple[list[dict], dict]:
    archive_id = re.search(r'wpsr_(\d{4}_\d{2}_\d{2}(?:_data)?)', url).group(1)
    release = archive_id[:10].replace('_', '-')
    folder = RAW / 'wpsr' / archive_id.replace('_', '-', 2)
    page = cached(url, folder / 'report.html').decode('cp1252')
    plain = html_module.unescape(re.sub(r'<[^>]+>', ' ', page))
    dated = re.search(r'Release Date:\s*([A-Za-z]+\.?\s+\d{1,2},\s*\d{4})', plain)
    if dated:
        page_release = date_string(pd.to_datetime(dated.group(1)))
        if page_release != release and not archive_id.endswith('_data'):
            raise ArchiveDateConflict(f'{archive_id}: page release {page_release} differs from folder')
        release = page_release
    elif archive_id.endswith('_data'):
        raise ValueError(f'{archive_id}: delayed report needs explicit publication date')
    values, files = [], []
    fallback = False
    for table in (1, 11, 12):
        linked = links(page, url)
        csv_urls = [u for u in linked if u.endswith(f'/csv/table{table}.csv')]
        if not csv_urls:
            csv_urls = [u for u in linked if u.endswith(f'/table{table}.csv')]
        if not csv_urls and 'PHP Error Encountered' in page:
            # The official index establishes the dated folder. Some historical PHP
            # pages fail in today's server while their static CSVs still work.
            csv_urls = [urljoin(url, f'csv/table{table}.csv')]
            fallback = True
        if len(csv_urls) != 1:
            if table in (11, 12) and not csv_urls:
                continue
            raise ValueError(f'{release}: expected one archived table{table} CSV')
        csv_url = csv_urls[0]
        if '/archive/' not in csv_url:
            raise ValueError(f'{release}: table URL is not an archive')
        csv_path = source_path(folder / f'table{table}.csv')
        content = cached(csv_url, csv_path)
        parse = parse_table if table == 1 else parse_prices
        values.extend(parse(content, release, csv_path.relative_to(ROOT).as_posix()))
        files.append(file_record(csv_path, csv_url))
    record = {'release_date': release, 'archive_id': archive_id, 'files': files,
              'report': file_record(folder / 'report.html', url), 'rows': len(values),
              'static_csv_fallback_after_php_error': fallback}
    time.sleep(0.10)
    return values, record


def archive_download(limit: int | None) -> tuple[pd.DataFrame, list[dict], list[dict], list[dict]]:
    index_path = RAW / 'wpsr' / ('index_' + TODAY + '.html')
    html = cached(ARCHIVE_URL, index_path).decode('cp1252')
    report_urls = sorted(u for u in links(html, ARCHIVE_URL)
                         if re.search(r'wpsr_\d{4}_\d{2}_\d{2}(?:_data)?\.', u))
    report_urls = [u for u in report_urls
                   if '2007-01-01' <= re.search(r'wpsr_(\d{4}_\d{2}_\d{2})', u)
                   .group(1).replace('_', '-') <= TODAY]
    if limit:
        report_urls = report_urls[:limit]
    print(f'Processing {len(report_urls)} dated EIA reports', flush=True)
    rows, records, errors, exclusions = [], [file_record(index_path, ARCHIVE_URL)], [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(one_report, url): url for url in report_urls}
        for count, future in enumerate(as_completed(futures), 1):
            try:
                values, record = future.result()
                rows.extend(values)
                records.append(record)
            except ArchiveDateConflict as error:
                url = futures[future]
                archive_id = re.search(r'wpsr_(\d{4}_\d{2}_\d{2})', url).group(1)
                folder = RAW / 'wpsr' / archive_id.replace('_', '-')
                files = [file_record(folder / 'report.html', url)]
                files.extend(file_record(source_path(folder / f'table{table}.csv'),
                                         urljoin(url, f'csv/table{table}.csv'))
                             for table in (1, 11, 12)
                             if source_path(folder / f'table{table}.csv').exists())
                exclusions.append({'url': url, 'reason': str(error), 'files': files})
                print('Quarantined:', str(error), flush=True)
            except Exception as error:
                errors.append({'url': futures[future], 'error': str(error)})
                print('Archive issue:', str(error), flush=True)
            if count % 50 == 0 or count == len(report_urls):
                print(f'EIA reports checked: {count}/{len(report_urls)}; issues: {len(errors)}', flush=True)
    return pd.DataFrame(rows), records, errors, exclusions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Smoke test only; separate output files')
    args = parser.parse_args()
    (MANIFESTS / 'supplement').mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    market, market_records = daily_download()
    weekly, archive_records, errors, exclusions = archive_download(args.limit)
    suffix = '_smoke' if args.limit else ''
    market_path = OUT / f'supplement_daily_market{suffix}.csv'
    weekly_path = OUT / f'supplement_eia_release_history{suffix}.csv'
    market.sort_values(['feature', 'observation_date']).to_csv(market_path, index=False)
    if not weekly.empty:
        weekly = weekly.sort_values(['release_date', 'feature']).reset_index(drop=True)
        assert not weekly.duplicated(['release_date', 'feature', 'observation_date']).any()
        weekly.to_csv(weekly_path, index=False)
    manifest = {
        'retrieved_at_utc': datetime.now(UTC).isoformat(),
        'forecast_origin': 'start of available_date in America/New_York',
        'daily_records': market_records, 'archive_records': archive_records,
        'archive_errors': errors,
        'archive_exclusions': exclusions,
        'outputs': [file_record(p, 'derived from sources in this manifest')
                    for p in [market_path, weekly_path] if p.exists()],
        'limitations': [
            'available_date is a conservative model-use date, not an exact release timestamp.',
            'Dated official archives avoid using later time-series revisions; retroactive '
            'corrections to archive files cannot be ruled out without contemporaneous snapshots.',
            'Only current-week fundamentals and latest five daily price columns are extracted. Older '
            'comparative columns are not backfilled to their observation dates.',
            'EIA commercial crude stocks change lease-stock coverage in October 2016; '
            'do not treat the level discontinuity as a genuine inventory shock.',
            'Monthly revised/backcast indexes and CFTC without verified release dates are '
            'excluded from this supplementary modeling dataset.',
        ],
    }
    manifest_path = MANIFESTS / 'supplement' / f'manifest_{TODAY}{suffix}.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    if not weekly.empty:
        print(weekly.groupby('feature').agg(rows=('value', 'size'),
              first=('observation_date', 'min'), last=('observation_date', 'max')).to_string())
    print('Market observations:', len(market), 'Archive issues:', len(errors),
          'Quarantined reports:', len(exclusions), flush=True)
    print('Manifest:', manifest_path.relative_to(ROOT), flush=True)
    if errors:
        raise SystemExit('Archive incomplete; errors recorded, rerun to resume cached downloads.')


if __name__ == '__main__':
    main()

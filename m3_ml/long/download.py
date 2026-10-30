"""Download longer-history oil data into data/long/ (does not touch data/raw or data/processed).

  uv run python -I ml/long/download.py [eia] [cftc]

EIA API v2 (key from .env, never written out): the 9 weekly core variables not yet
in data/raw/eia_api, WTI futures contracts 1-4 (EIA stops on 2024-04-05) and 3
product spot prices. Rows keep the EIA API layout used in data/raw/eia_api.

CFTC legacy Commitments of Traders, futures only: the 1986-2016 file and one file
per year since 2017. The zips go to .local/long/cftc/ (zips are not tracked, as in
data/raw); WTI rows (CFTC code 067651) are extracted to data/long/cftc/.
Every file is listed with its URL, size and SHA-256 in a sources.csv.
"""
import hashlib
import io
import os
import sys
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'long'
ZIPS = ROOT / '.local' / 'long' / 'cftc'
EIA_API = 'https://api.eia.gov/v2/seriesid'
CFTC = 'https://www.cftc.gov/files/dea/history'
WTI_CODE = '067651'

EIA_SERIES = {  # legacy series id -> file name
    'PET.WTESTUS1.W': 'total_stocks_ex_spr_weekly',
    'PET.WCSSTUS1.W': 'spr_crude_stocks_weekly',
    'PET.WGTSTUS1.W': 'gasoline_stocks_weekly',
    'PET.WDISTUS1.W': 'distillate_stocks_weekly',
    'PET.WKJSTUS1.W': 'jet_fuel_stocks_weekly',
    'PET.WRPUPUS2.W': 'total_products_supplied_weekly',
    'PET.WGFUPUS2.W': 'gasoline_products_supplied_weekly',
    'PET.WDIUPUS2.W': 'distillate_products_supplied_weekly',
    'PET.WKJUPUS2.W': 'jet_fuel_products_supplied_weekly',
    'PET.RCLC1.D': 'wti_futures_contract1_daily',
    'PET.RCLC2.D': 'wti_futures_contract2_daily',
    'PET.RCLC3.D': 'wti_futures_contract3_daily',
    'PET.RCLC4.D': 'wti_futures_contract4_daily',
    'PET.EER_EPMRU_PF4_RGC_DPG.D': 'gulf_conventional_gasoline_spot_daily',
    'PET.EER_EPJK_PF4_RGC_DPG.D': 'gulf_jet_fuel_spot_daily',
    'PET.EER_EPLLPA_PF4_Y44MB_DPG.D': 'mont_belvieu_propane_spot_daily',
}
CFTC_FILES = ['deacot1986_2016.zip'] + [f'deacot{y}.zip' for y in range(2017, 2027)]


def session():
    s = requests.Session()
    s.mount('https://', HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504),
                                                      allowed_methods=('GET',))))
    s.headers['User-Agent'] = 'NUS-QF5210-oil-price-forecasting/0.1'
    return s


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def eia_series(s, series_id, key):
    rows, offset = [], 0
    while True:
        params = {'api_key': key, 'offset': offset, 'length': 5000,
                  'sort[0][column]': 'period', 'sort[0][direction]': 'asc'}
        try:
            r = s.get(f'{EIA_API}/{series_id}', params=params, timeout=(30, 180))
        except requests.RequestException as error:
            raise RuntimeError(f'EIA request for {series_id} failed ({type(error).__name__})') from None
        if r.status_code != 200:
            raise RuntimeError(f'EIA returned HTTP {r.status_code} for {series_id}: {r.text.replace(key, "<redacted>")[:300]}')
        body = r.json()['response']
        rows += body['data']
        if not body['data'] or len(rows) >= int(body['total']):
            break
        offset += len(body['data'])
    frame = pd.DataFrame(rows).rename(columns={'period': 'observation_date'})
    frame['observation_date'] = pd.to_datetime(frame['observation_date'], errors='raise')
    frame['value'] = pd.to_numeric(frame['value'], errors='coerce')
    frame.insert(0, 'series_id', series_id)
    frame = frame.sort_values('observation_date').drop_duplicates('observation_date', keep='last').reset_index(drop=True)
    if frame['value'].notna().sum() == 0:
        raise ValueError(f'EIA series {series_id} has no numeric observations')
    return frame


def download_eia(s, key, today):
    folder = OUT / 'eia_api'
    folder.mkdir(parents=True, exist_ok=True)
    sources = []
    for series_id, name in EIA_SERIES.items():
        frame = eia_series(s, series_id, key)
        data = frame.to_csv(index=False, date_format='%Y-%m-%d', lineterminator='\n').encode('utf-8')
        assert key.encode() not in data
        (folder / f'{name}.csv').write_bytes(data)
        valid = frame.loc[frame['value'].notna(), 'observation_date']
        sources.append({'file': f'{name}.csv', 'series_id': series_id, 'url': f'{EIA_API}/{series_id} (api_key from .env)',
                        'downloaded': today, 'coverage': f'{valid.iloc[0]:%Y-%m-%d}..{valid.iloc[-1]:%Y-%m-%d}',
                        'rows': int(len(valid)), 'units': frame['units'].iloc[-1] if 'units' in frame else '',
                        'bytes': len(data), 'sha256': sha256(data)})
        print(f'{name}: {sources[-1]["coverage"]}, {len(valid)} rows', flush=True)
    pd.DataFrame(sources).to_csv(folder / 'sources.csv', index=False, lineterminator='\n')


def download_cftc(s, today):
    ZIPS.mkdir(parents=True, exist_ok=True)
    (OUT / 'cftc').mkdir(parents=True, exist_ok=True)
    sources, wti = [], []
    for name in CFTC_FILES:
        r = s.get(f'{CFTC}/{name}', timeout=(30, 300))
        r.raise_for_status()
        (ZIPS / name).write_bytes(r.content)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            members = [m for m in z.namelist() if m.lower().endswith('.txt')]
            assert len(members) == 1, (name, z.namelist())
            table = pd.read_csv(z.open(members[0]), dtype=str, encoding='latin-1')
        table.columns = table.columns.str.strip()
        rows = table[table['CFTC Contract Market Code'].str.strip() == WTI_CODE].copy()
        rows.insert(0, 'source_file', name)
        wti.append(rows)
        sources.append({'file': name, 'url': f'{CFTC}/{name}', 'downloaded': today, 'stored_at': f'.local/long/cftc/{name}',
                        'bytes': len(r.content), 'sha256': sha256(r.content), 'wti_rows': len(rows)})
        print(f'{name}: {len(r.content) / 2**20:.1f} MB, {len(rows)} WTI rows', flush=True)
    out = pd.concat(wti, ignore_index=True)
    out['report_date'] = pd.to_datetime(out['As of Date in Form YYYY-MM-DD'].str.strip())
    out = out.sort_values('report_date')
    duplicated = out.duplicated('report_date', keep=False)
    assert not duplicated.any(), out.loc[duplicated, ['source_file', 'report_date']]
    data = out.drop(columns='report_date').to_csv(index=False, lineterminator='\n').encode('utf-8')
    (OUT / 'cftc' / 'wti_legacy_futures_only_weekly.csv').write_bytes(data)
    sources.append({'file': 'wti_legacy_futures_only_weekly.csv', 'url': f'extracted: CFTC code {WTI_CODE}',
                    'downloaded': today, 'stored_at': 'data/long/cftc/wti_legacy_futures_only_weekly.csv',
                    'bytes': len(data), 'sha256': sha256(data), 'wti_rows': len(out)})
    pd.DataFrame(sources).to_csv(OUT / 'cftc' / 'sources.csv', index=False, lineterminator='\n')
    print(f'WTI legacy positions: {out["report_date"].min():%Y-%m-%d}..{out["report_date"].max():%Y-%m-%d}, {len(out)} reports')


def main():
    load_dotenv(ROOT / '.env')
    key = os.getenv('EIA_API_KEY', '').strip()
    if not key:
        raise RuntimeError('EIA_API_KEY is missing from .env')
    s, today = session(), date.today().isoformat()
    parts = sys.argv[1:] or ['eia', 'cftc']
    if 'eia' in parts:
        download_eia(s, key, today)
    if 'cftc' in parts:
        download_cftc(s, today)


if __name__ == '__main__':
    main()

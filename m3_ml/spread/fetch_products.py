"""Fetch the NY Harbor gasoline and heating oil daily spot prices for the 3-2-1
crack spread (study 4) from the EIA API v2 (legacy series-id route).

  uv run python -m m3_ml.spread.fetch_products

These are the EIA series that FRED republishes as DGASNYH and DHOILNYH (FRED was
unreachable on 2026-10-18). The key is read from .env (EIA_API_KEY) and is never
printed or written: responses and error messages are redacted.
Writes data/raw/eia_api/spot_prices/*.csv and sources.csv (hashes, coverage).
"""
import hashlib
import os
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/raw/eia_api/spot_prices'
API_ROOT = 'https://api.eia.gov/v2/seriesid'
SERIES = {
    'PET.EER_EPMRU_PF4_Y35NY_DPG.D': ('ny_conventional_gasoline_spot_daily.csv', 'DGASNYH'),
    'PET.EER_EPD2F_PF4_Y35NY_DPG.D': ('ny_heating_oil_spot_daily.csv', 'DHOILNYH'),
}


def session():
    retry = Retry(total=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=('GET',))
    s = requests.Session()
    s.mount('https://', HTTPAdapter(max_retries=retry))
    s.headers['User-Agent'] = 'NUS-QF5210-oil-price-forecasting/0.1'
    return s


def fetch(s, series_id, key):
    rows, offset = [], 0
    while True:
        params = {'api_key': key, 'offset': offset, 'length': 5000,
                  'sort[0][column]': 'period', 'sort[0][direction]': 'asc'}
        try:
            r = s.get(f'{API_ROOT}/{series_id}', params=params, timeout=(30, 180))
        except requests.RequestException as error:
            raise RuntimeError(f'EIA request for {series_id} failed ({type(error).__name__})') from None
        if r.status_code != 200:
            raise RuntimeError(f'EIA returned HTTP {r.status_code} for {series_id}: {r.text.replace(key, "<redacted>")[:300]}')
        data = r.json().get('response', {})
        page = data.get('data', [])
        rows.extend(page)
        if not page or len(rows) >= int(data.get('total', len(rows))):
            break
        offset += len(page)
    if not rows:
        raise ValueError(f'EIA series {series_id} returned no observations')
    frame = pd.DataFrame(rows).rename(columns={'period': 'observation_date'})
    frame.insert(0, 'series_id', series_id)
    frame['observation_date'] = pd.to_datetime(frame['observation_date'])
    frame['value'] = pd.to_numeric(frame['value'], errors='coerce')
    return frame.sort_values('observation_date').drop_duplicates('observation_date', keep='last')


def main():
    load_dotenv(ROOT / '.env')
    key = os.getenv('EIA_API_KEY', '').strip()
    if not key:
        raise RuntimeError('EIA_API_KEY is missing from .env')
    OUT.mkdir(parents=True, exist_ok=True)
    s = session()
    sources = []
    for series_id, (name, fred_id) in SERIES.items():
        frame = fetch(s, series_id, key)
        path = OUT / name
        frame.to_csv(path, index=False, date_format='%Y-%m-%d', lineterminator='\n')
        valid = frame.dropna(subset=['value'])
        sources.append({'file': name, 'series_id': series_id, 'fred_equivalent': fred_id,
                        'url': f'{API_ROOT}/{series_id} (api_key from .env)', 'downloaded': date.today().isoformat(),
                        'coverage': f'{valid.observation_date.min().date()}..{valid.observation_date.max().date()}',
                        'rows': len(valid), 'units': valid['units'].iloc[0] if 'units' in valid else '',
                        'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'notes': 'Daily spot price FOB; EIA publishes daily spot prices with a weekly update.'})
        print(f'{name}: {len(valid)} rows, {sources[-1]["coverage"]}, units {sources[-1]["units"]}')
    pd.DataFrame(sources).to_csv(OUT / 'sources.csv', index=False, lineterminator='\n')


if __name__ == '__main__':
    main()

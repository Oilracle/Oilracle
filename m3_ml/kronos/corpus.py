"""Download and clean the Kronos pretraining corpus (PLAN.md section 3).

  uv run python -m m3_ml.kronos.corpus download

Daily OHLCV from Yahoo Finance (yfinance, auto_adjust=False: split-adjusted,
no dividend adjustment) from 2000-01-01 to today, one parquet per ticker in
data/kronos/corpus in the submission, with a legacy local-cache fallback. Training
truncates every series at its cutoff (see load_corpus). m3_ml/kronos/corpus_manifest.csv
records rows, dates and file hashes for every ticker, including failures.
"""
import argparse
import hashlib
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from .universe import GROUP_A, WTI_INPUT, point_in_time_sp500
from m3_ml.common.protocol import TRAIN_END, FIT_END, TEST_END

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / 'data/kronos/corpus'
if not CORPUS.is_dir():
    CORPUS = ROOT / '.local/kronos/corpus'
MANIFEST = Path(__file__).resolve().parent / 'corpus_manifest.csv'
START = '2000-01-01'
CUTOFFS = (TRAIN_END, FIT_END)
FIELDS = ['open', 'high', 'low', 'close', 'volume']
CHUNK = 40


def universe():
    stocks = sorted(set(point_in_time_sp500(CUTOFFS[0])) | set(point_in_time_sp500(CUTOFFS[1])))
    return GROUP_A + [s for s in stocks if s not in GROUP_A]


def path(ticker):
    return CORPUS / f'{ticker.replace("=", "_").replace("^", "_")}.parquet'


def clean(frame, ticker):
    """Drop rows with missing or non-positive OHLC (the 2020-04-20 WTI futures
    price is the one allowed negative value); missing volume becomes 0."""
    frame = frame.rename(columns=str.lower)[FIELDS].copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame = frame[~frame.index.duplicated(keep='last')].sort_index()
    ohlc = frame[['open', 'high', 'low', 'close']]
    ok = ohlc.notna().all(axis=1)
    ok &= (ohlc > 0).all(axis=1) if ticker != WTI_INPUT else ohlc.notna().all(axis=1)
    frame = frame[ok]
    frame['volume'] = frame['volume'].fillna(0).clip(lower=0)
    return frame


def download():
    CORPUS.mkdir(parents=True, exist_ok=True)
    tickers = universe()
    rows = []
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        data = yf.download(chunk, start=START, auto_adjust=False, actions=False, group_by='ticker',
                           threads=True, progress=False)
        for t in chunk:
            try:
                frame = clean(data[t], t) if t in data.columns.get_level_values(0) else pd.DataFrame()
            except Exception as exc:
                frame, error = pd.DataFrame(), repr(exc)
            else:
                error = '' if len(frame) else 'no data'
            if len(frame):
                frame.to_parquet(path(t))
                digest = hashlib.sha256(path(t).read_bytes()).hexdigest()
                rows.append({'ticker': t, 'group': 'A' if t in GROUP_A else 'B', 'rows': len(frame),
                             'first': frame.index.min().date(), 'last': frame.index.max().date(),
                             'sha256': digest, 'error': ''})
            else:
                rows.append({'ticker': t, 'group': 'A' if t in GROUP_A else 'B', 'rows': 0,
                             'first': '', 'last': '', 'sha256': '', 'error': error})
        print(f'{min(i + CHUNK, len(tickers))}/{len(tickers)} tickers', flush=True)
        time.sleep(2)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST, index=False, lineterminator='\n')
    print(manifest.groupby('group')['rows'].agg(['count', lambda r: int((r > 0).sum()), 'sum']))

def refresh_wti(through=TEST_END):
    """Refresh only the small CL futures suffix needed for causal daily updates."""
    cutoff = pd.Timestamp(through).normalize()
    cached = pd.read_parquet(path(WTI_INPUT))
    start = cached.index.max() + pd.Timedelta(days=1)
    if start > cutoff:
        print('WTI futures cache already covers', cutoff.date())
        return
    downloaded = yf.download(WTI_INPUT, start=start.date(),
                             end=(cutoff + pd.Timedelta(days=1)).date(),
                             auto_adjust=False, actions=False, group_by='ticker',
                             threads=False, progress=False)
    if isinstance(downloaded.columns, pd.MultiIndex):
        downloaded = downloaded[WTI_INPUT]
    suffix = clean(downloaded, WTI_INPUT)
    if suffix.empty:
        raise ValueError('No new WTI futures bars returned; daily-update cache was not refreshed.')
    merged = pd.concat([cached, suffix]).sort_index()
    merged = merged[~merged.index.duplicated(keep='last')]
    merged.to_parquet(path(WTI_INPUT))
    manifest = pd.read_csv(MANIFEST)
    row = dict(ticker=WTI_INPUT, group='A', rows=len(merged),
               first=str(merged.index.min().date()), last=str(merged.index.max().date()),
               sha256=hashlib.sha256(path(WTI_INPUT).read_bytes()).hexdigest(), error='')
    manifest = manifest[manifest.ticker != WTI_INPUT]
    manifest = pd.concat([pd.DataFrame([row]), manifest], ignore_index=True)
    manifest.to_csv(MANIFEST, index=False, lineterminator='\n')
    print('Refreshed WTI futures through', merged.index.max().date())



def load_corpus(cutoff):
    """Every ticker truncated at `cutoff` (inclusive): {ticker: DataFrame}."""
    manifest = pd.read_csv(MANIFEST)
    out = {}
    for t in manifest.loc[manifest['rows'] > 0, 'ticker']:
        frame = pd.read_parquet(path(t))
        frame = frame[frame.index <= pd.Timestamp(cutoff)]
        if len(frame):
            out[t] = frame
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['download', 'refresh-wti'])
    parser.add_argument('--through', default=TEST_END,
                        help='Last desired WTI futures session for the bounded suffix refresh.')
    args = parser.parse_args()
    download() if args.stage == 'download' else refresh_wti(args.through)

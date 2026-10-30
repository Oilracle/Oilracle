"""Check official news snapshots; --write archives originals and refreshes them.

Run from the project root: python -m m3_ml.common.refresh_news [--write].
"""
import argparse
import hashlib
import io
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from m3_ml.common.data import ROOT
from m3_ml.common.news import AI_GPR_SERIES, NEWS

URLS = {
    'ai_gpr/ai_gpr_data_daily.csv': 'https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv',
    'gpr/data_gpr_daily_recent.xls': 'https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls',
    'epu/All_Daily_Policy_Data.csv': 'https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv',
}


def inspect_source(name, content):
    if name.startswith('gpr/'):
        frame = pd.read_excel(io.BytesIO(content), usecols=['date', 'GPRD'])
        dates, values = pd.to_datetime(frame.date), frame[['GPRD']]
    else:
        frame = pd.read_csv(io.BytesIO(content))
        if name.startswith('ai_gpr/'):
            dates, values = pd.to_datetime(frame.Date), frame[list(AI_GPR_SERIES)]
        else:
            dates = pd.to_datetime(frame[['year', 'month', 'day']])
            values = frame[['daily_policy_index']]
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(f'invalid source dates: {name}')
    import numpy as np
    if not np.isfinite(values).all(axis=None) or (values < 0).any(axis=None):
        raise ValueError(f'invalid news values: {name}')
    return f'{dates.min():%Y-%m-%d}..{dates.max():%Y-%m-%d}'


def main(write=False):
    sources = pd.read_csv(NEWS / 'sources.csv')
    now = datetime.now(ZoneInfo('Asia/Singapore'))
    pending = {}
    for index, row in sources.iterrows():
        name = row['file']
        if row['url'] != URLS[name]:
            raise ValueError(f'unrecognized official source URL: {name}')
        response = requests.get(URLS[name], timeout=40)
        response.raise_for_status()
        content = response.content
        coverage = inspect_source(name, content)
        digest = hashlib.sha256(content).hexdigest()
        old = (NEWS / name).read_bytes()
        if hashlib.sha256(old).hexdigest() != row['sha256']:
            raise ValueError(f'local source does not match manifest: {name}')
        if coverage[:10] > row['coverage'][:10] or coverage[-10:] < row['coverage'][-10:]:
            raise ValueError(f'official source coverage shrank: {name}')
        print(f'{name}: {coverage}; changed={content != old}', flush=True)
        if content != old:
            pending[name] = content
            sources.loc[index, ['downloaded', 'coverage', 'bytes', 'sha256']] = [
                now.strftime('%Y-%m-%d'), coverage, len(content), digest]
    if write and pending:
        archive = ROOT / '.local/news_source_archive' / now.strftime('%Y%m%dT%H%M%S')
        archive.mkdir(parents=True, exist_ok=False)
        (archive / 'sources.csv').write_bytes((NEWS / 'sources.csv').read_bytes())
        for name, content in pending.items():
            backup = archive / name
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes((NEWS / name).read_bytes())
            (NEWS / name).write_bytes(content)
        sources.to_csv(NEWS / 'sources.csv', index=False, lineterminator='\n')
        print(f'Updated {len(pending)} files; originals archived in {archive}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    main(parser.parse_args().write)

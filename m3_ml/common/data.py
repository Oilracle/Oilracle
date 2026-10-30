"""Team data, observable WTI anchors and audited label-publication evidence.

The raw target is not a contemporaneously observable feature. Price forecasts
use the newest quotation admitted by the portable publication schedule, with
an explicit weekly-release assumption where archive matches are unavailable.
"""
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

from .protocol import HORIZONS, TEST_START, TEST_END, VALIDATION_START, VALIDATION_END

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / 'data/processed/wti_daily.parquet'
PUBLICATION = ROOT / 'data/processed/wti_publication.csv'
PRICE = 'target_wti'
VALIDATION = (VALIDATION_START, VALIDATION_END)
TEST = (TEST_START, TEST_END)


def load_table():
    table = pd.read_parquet(TABLE).sort_values('date').reset_index(drop=True)
    table['date'] = pd.to_datetime(table['date'])
    return table

@lru_cache(maxsize=1)
def _publication_evidence():
    evidence = pd.read_csv(PUBLICATION, parse_dates=['date', 'available_date'])
    if evidence['date'].duplicated().any():
        raise ValueError('publication evidence has duplicate quotation dates')
    return evidence.set_index('date')


def label_available_dates(table_or_dates):
    """Publication dates aligned to input rows; unknown evidence stays NaT.

    Accept a DataFrame containing date (and optionally target_wti), or a date
    Series/sequence. Portable evidence distinguishes dated archive matches from
    the explicitly retrospective weekly-release assumption for unmatched rows.
    """
    is_frame = isinstance(table_or_dates, pd.DataFrame)
    dates = table_or_dates['date'] if is_frame else pd.Series(table_or_dates)
    aligned = _publication_evidence().reindex(pd.DatetimeIndex(pd.to_datetime(dates)))
    result = pd.Series(aligned['available_date'].to_numpy(), index=dates.index)
    if is_frame and PRICE in table_or_dates:
        matched = np.isclose(table_or_dates[PRICE].to_numpy(dtype=float),
                             aligned[PRICE].to_numpy(dtype=float), equal_nan=False)
        result = result.where(matched)
    return result


def observable_history(table, origin):
    """Raw quotes public under the evidence policy; preserve quotation indices.

    This is a sparse history, NOT a compressed trading calendar. Time-series
    callers must retain gaps when advancing state to a forecast target.
    """
    return table.loc[label_available_dates(table).le(pd.Timestamp(origin))].copy()


def observable_price(table):
    """Latest quotation admitted by the portable publication schedule.

    Release events are processed in publication order, but a late release of
    an older quote cannot rewind the anchor's observation date. This includes
    explicitly assumed weekly-release dates, not a claim of real-time vintages.
    Only the leading history before our first scheduled quote uses the input
    table's already-published (pre-table) WTI feature as a cold-start anchor.
    """
    evidence = _publication_evidence().sort_index()
    valid = evidence['available_date'].notna()
    events = evidence.loc[valid, ['available_date']].copy()
    events['quote_row'] = np.flatnonzero(valid.to_numpy())
    events = events.sort_values(['available_date', 'quote_row'])
    event_dates = events['available_date'].to_numpy(dtype='datetime64[ns]')
    newest = np.maximum.accumulate(events['quote_row'].to_numpy(dtype=int))
    lookup = np.searchsorted(event_dates, pd.to_datetime(table['date']).to_numpy(), side='right') - 1
    result = table['wti_spot_last_published'].astype(float).copy()
    has_quote = lookup >= 0
    if has_quote.any():
        rows = newest[lookup[has_quote]]
        result.iloc[np.flatnonzero(has_quote)] = evidence[PRICE].to_numpy(dtype=float)[rows]
    return result


def log_price(price):
    """Log price; NaN at non-positive quotes (including 2020-04-20)."""
    return np.log(price.where(price > 0))


def safe_log_price(price):
    """Causal positive-price carry for feature engineering only."""
    return np.log(price.where(price > 0).ffill())


def targets(table, horizons=HORIZONS):
    """Future raw quote log price minus the observable origin's log anchor.

    The horizon counts quoted table rows, not publication events. A hidden
    same-origin target is never used as a baseline or explanatory variable.
    """
    lp = log_price(table[PRICE])
    anchor = log_price(observable_price(table))
    return pd.DataFrame({h: lp.shift(-h) - anchor for h in horizons}, index=table.index)

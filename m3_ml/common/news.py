"""News-based daily risk indices as features (data/raw/news, see sources.csv).

The article date is NOT the index release date. Under the release-calendar
assumption approved on 2026-10-29, AI-GPR month-end statistics become available
on the next month's first day, keyword GPR Sunday statistics on the following
Monday (US federal holidays deferred), and EPU statistics the next day.
Monthly/weekly states expire at the next scheduled release: absent source
observations cannot silently extend the previous release. EPU retains a
maximum seven-calendar-day carry. No raw news observations are filled.

Features per series, on log(1 + index) to tame the skew and the zero days of
the sparse oil sub-indexes:
  _ma7, _ma30   trailing 7- and 30-day means
  _shock        _ma7 minus the trailing 90-day mean (news surprise)

Caveats: the AI-GPR history was scored backwards with a current LLM and was
revised in 2026, and EPU revises its last 30 days at every update, so neither
is a real-time vintage. The GPR file's 'event' column is a hindsight
annotation and is never read.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

NEWS = Path(__file__).resolve().parents[2] / 'data/raw/news'

# Chosen a priori: overall and oil geopolitical risk, oil threats vs acts, the
# two oil regions that are not mostly zero (Middle East, Russia), the
# keyword-based GPR as an LLM-free cross-check, and US policy uncertainty.
AI_GPR_SERIES = {'GPR_AI': 'ai_gpr', 'GPR_OIL': 'ai_gpr_oil', 'GPR_OIL_THREATS': 'ai_gpr_oil_threats',
                 'GPR_OIL_ACTS': 'ai_gpr_oil_acts', 'GPR_OIL_MiddleEast': 'ai_gpr_oil_mideast',
                 'GPR_OIL_Russia': 'ai_gpr_oil_russia'}
MIN_DAYS = {7: 4, 30: 20, 90: 60}  # minimum observed days per trailing window
NEWS_POLICY_ID = 'news-release-calendar-v1'
FEDERAL_BUSINESS_DAY = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())


def news_metadata():
    """Record the exact source snapshot and assumptions used in this run."""
    sources = pd.read_csv(NEWS / 'sources.csv').to_dict('records')
    for source in sources:
        actual = hashlib.sha256((NEWS / source['file']).read_bytes()).hexdigest()
        if actual != source['sha256']:
            raise ValueError(f"news source hash mismatch: {source['file']}")
    return {'news_policy_id': NEWS_POLICY_ID,
            'availability_status': 'assumed_release_calendar_not_verified_historical_timestamps',
            'ai_gpr': 'previous_month_end_statistics_available_next_month_day_1',
            'gpr_keyword': 'previous_sunday_statistics_available_monday_federal_holidays_deferred',
            'epu': 'next_calendar_day_max_7_day_carry',
            'vintage_status': 'retrospective_current_vintage',
            'sources': sources}


def load_news():
    """All selected series on a calendar-day index (NaN where not published)."""
    ai = pd.read_csv(NEWS / 'ai_gpr/ai_gpr_data_daily.csv', parse_dates=['Date']).set_index('Date')
    gpr = pd.read_excel(NEWS / 'gpr/data_gpr_daily_recent.xls', usecols=['date', 'GPRD'])
    gpr = gpr.assign(date=pd.to_datetime(gpr['date'])).set_index('date')['GPRD']
    epu = pd.read_csv(NEWS / 'epu/All_Daily_Policy_Data.csv')
    epu = epu.set_index(pd.to_datetime(epu[['year', 'month', 'day']]))['daily_policy_index']
    frame = pd.concat([ai[list(AI_GPR_SERIES)].rename(columns=AI_GPR_SERIES),
                       gpr.rename('gpr_keyword'), epu.rename('epu')], axis=1)
    days = pd.date_range(frame.index.min(), frame.index.max())
    return frame.reindex(days)


def _calendar_news(news):
    news = news.copy()
    news.index = pd.DatetimeIndex(pd.to_datetime(news.index))
    if news.index.hasnans or news.index.has_duplicates or news.columns.has_duplicates:
        raise ValueError('news dates and columns must be unique and non-missing')
    if news.empty or not news.index.equals(news.index.normalize()):
        raise ValueError('news requires nonempty calendar-day observations')
    unknown = set(news) - set(AI_GPR_SERIES.values()) - {'gpr_keyword', 'epu'}
    if unknown:
        raise ValueError(f'unknown news series: {sorted(unknown)}')
    values = news.to_numpy(dtype=float)
    if np.isinf(values).any() or np.any(values < 0):
        raise ValueError('news indices must be nonnegative and finite or missing')
    return news.sort_index().reindex(pd.date_range(news.index.min(), news.index.max()))


def _news_features_and_audit(dates, news=None):
    """Use release events, including missing releases, without skipping origins."""
    dates = pd.Series(dates)
    origins = pd.DatetimeIndex(pd.to_datetime(dates))
    if origins.hasnans or not origins.equals(origins.normalize()):
        raise ValueError('forecast origins must be non-missing calendar dates')
    news = _calendar_news(load_news() if news is None else news)
    logged = np.log1p(news)
    means = {k: logged.rolling(k, min_periods=MIN_DAYS[k]).mean() for k in MIN_DAYS}
    out = pd.concat([means[7].add_suffix('_ma7'), means[30].add_suffix('_ma30'),
                     (means[7] - means[90]).add_suffix('_shock')], axis=1)
    # Include future expected releases with NaN values. Otherwise merge-asof
    # would carry the last available month/week forever when a file is stale.
    days = pd.date_range(min(news.index.min(), origins.min()) - pd.Timedelta(days=35),
                         max(news.index.max(), origins.max()))
    blocks, audits = [], []
    for series in news:
        columns = [f'{series}_{suffix}' for suffix in ('ma7', 'ma30', 'shock')]
        if series in AI_GPR_SERIES.values():
            endpoints = days[days.is_month_end]
            releases = endpoints + pd.Timedelta(days=1)
        elif series == 'gpr_keyword':
            endpoints = days[days.dayofweek == 6]
            releases = pd.DatetimeIndex([FEDERAL_BUSINESS_DAY.rollforward(d + pd.Timedelta(days=1))
                                         for d in endpoints])
        else:
            endpoints = news.index[news[series].notna()]
            releases = endpoints + pd.Timedelta(days=1)
        scheduled = pd.DataFrame({'available_date': releases, 'data_through': endpoints})
        values = out[columns].reindex(endpoints).reset_index(drop=True)
        # A partial rolling window beyond the source tail is not a new release.
        observed = news[series].reindex(endpoints).notna().to_numpy()
        values.loc[~observed, :] = np.nan
        scheduled = pd.concat([scheduled, values], axis=1)
        requested = pd.DataFrame({'origin_date': origins, '_position': np.arange(len(dates))})
        matched = pd.merge_asof(requested.sort_values('origin_date'), scheduled,
                                left_on='origin_date', right_on='available_date',
                                direction='backward').sort_values('_position')
        if series == 'epu':
            stale = (matched.origin_date - matched.available_date).dt.days.gt(7)
            matched.loc[stale, columns] = np.nan
        finite = np.isfinite(matched[columns]).all(axis=1)
        block = matched[columns].set_axis(dates.index)
        blocks.append(block)
        audit = matched[['origin_date', 'data_through', 'available_date']].copy()
        audit['series'] = series
        audit['age_days'] = (audit.origin_date - audit.data_through).dt.days
        audit['complete'] = finite.to_numpy()
        audit['news_policy_id'] = NEWS_POLICY_ID
        audits.append(audit)
    return pd.concat(blocks, axis=1).reindex(columns=out.columns), pd.concat(audits, ignore_index=True)


def news_features(dates, news=None):
    """Release-aligned features; unavailable scheduled releases remain NaN."""
    return _news_features_and_audit(dates, news)[0]


def news_availability(dates, news=None):
    """Audit dates describe the scheduled snapshot; complete=False means unusable."""
    return _news_features_and_audit(dates, news)[1]


def require_complete_features(X, dates, origins):
    """Fail before fitting, with source/column/date details; never drop origins."""
    missing = ~np.isfinite(X.iloc[origins])
    if missing.any(axis=None):
        details = []
        for column in missing.columns[missing.any()]:
            affected = dates.iloc[np.asarray(origins)[missing[column].to_numpy()]]
            details.append(f'{column}: {affected.min():%Y-%m-%d}..'
                           f'{affected.max():%Y-%m-%d} ({len(affected)} origins)')
        raise ValueError('forecast origin has incomplete causal features; ' + '; '.join(details)
                         + '. Supply published source data; do not fill or drop origins.')


if __name__ == '__main__':
    import argparse
    from .data import load_table
    from .walkforward import forecast_origins
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-output', type=Path)
    args = parser.parse_args()
    table = load_table()
    features, audit = _news_features_and_audit(table.date)
    news_metadata()  # verify recorded hashes, even for an audit-only run
    for mode in ('fixed_origin', 'daily_update'):
        origins = forecast_origins(table.date, mode)
        require_complete_features(features, table.date, origins)
        print(f'{mode}: {len(origins)} complete origins; {NEWS_POLICY_ID}')
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(args.audit_output, index=False, date_format='%Y-%m-%d', lineterminator='\n')

"""Pretraining universe (PLAN.md section 3).

Group A is fixed in advance. Group B is the S&P 500 at each cutoff, rebuilt by
undoing every index change after the cutoff (sp500_changes.csv) starting from
the current list (sp500_current.csv); see point_in_time_sp500.

Sources (read 2026-10-17): current constituents from Wikipedia "List of S&P 500
companies" (revision 1380159481, 503 rows); changes since 2020 from "Historical
components of the S&P 500" (revision 1379179365, 141 rows).
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

ENERGY = ['CL=F', 'BZ=F', 'NG=F', 'HO=F', 'RB=F']           # weight x3 in sampling
GROUP_A = ENERGY + [
    # metals and agriculture
    'GC=F', 'SI=F', 'HG=F', 'PL=F', 'PA=F', 'ZC=F', 'ZW=F', 'ZS=F', 'ZM=F', 'ZL=F',
    'KC=F', 'SB=F', 'CC=F', 'CT=F', 'LE=F', 'HE=F',
    # equity index and Treasury futures
    'ES=F', 'NQ=F', 'YM=F', 'RTY=F', 'ZN=F', 'ZB=F', 'ZF=F', 'ZT=F',
    # FX, dollar index, volatility indices
    'EURUSD=X', 'JPY=X', 'GBPUSD=X', 'AUDUSD=X', 'CAD=X', 'CHF=X', 'NZDUSD=X', 'MXN=X', 'CNY=X',
    'DX-Y.NYB', '^VIX', '^OVX',
    # broad, bond, commodity and energy ETFs
    'SPY', 'QQQ', 'IWM', 'DIA', 'EFA', 'EEM', 'VWO', 'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'TIP',
    'GLD', 'SLV', 'USO', 'BNO', 'UNG', 'DBC', 'DBA', 'XLE', 'XOP', 'OIH',
    # sector and country ETFs
    'XLF', 'XLK', 'XLV', 'XLI', 'XLB', 'XLU', 'XLP', 'XLY', 'XME', 'KRE', 'SMH', 'VNQ',
    'EWZ', 'EWJ', 'EWG', 'EWU', 'EWC', 'EWA', 'FXI',
]
WTI_INPUT = 'CL=F'


def yahoo_symbol(ticker):
    """Wikipedia uses BRK.B; Yahoo uses BRK-B."""
    return ticker.replace('.', '-')


# Rows of the change table that are ticker renames, not index changes.
RENAMES = {('DAY', 'CDAY'), ('WTW', 'WLTW')}
# Older symbols in the change table -> symbol in the current list.
ALIASES = {'SATS': 'ECHO', 'CDAY': 'DAY', 'WLTW': 'WTW'}


def point_in_time_sp500(cutoff):
    """S&P 500 members on `cutoff`: start from today's list and undo, newest
    first, every change dated after the cutoff (remove the added ticker, put
    back the removed one). Renames are not undone."""
    members = set(pd.read_csv(HERE / 'sp500_current.csv')['ticker'])
    changes = pd.read_csv(HERE / 'sp500_changes.csv', parse_dates=['date'], keep_default_na=False)
    later = changes[changes['date'] > pd.Timestamp(cutoff)].sort_values('date', ascending=False, kind='stable')
    for row in later.itertuples():
        if (row.added_ticker, row.removed_ticker) in RENAMES:
            continue
        if row.added_ticker:
            members.discard(ALIASES.get(row.added_ticker, row.added_ticker))
        if row.removed_ticker:
            members.add(row.removed_ticker)
    return sorted(yahoo_symbol(t) for t in members)

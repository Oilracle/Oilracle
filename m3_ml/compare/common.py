"""Frozen October comparison contracts; legacy root outputs are historical only."""
from pathlib import Path

import numpy as np
import pandas as pd

from m3_ml.common.data import observable_price
from m3_ml.common.protocol import (
    FIT_END, HORIZONS, MODES, PROTOCOL_ID, TEST_END, TEST_START, TRAIN_END,
    VALIDATION_END, VALIDATION_START,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'outputs' / 'frozen_october_2026'
H_MAX = max(HORIZONS)
PROVENANCE = {'protocol_id': PROTOCOL_ID, 'train_end': TRAIN_END,
              'validation_start': VALIDATION_START, 'validation_end': VALIDATION_END,
              'fit_end': FIT_END}
KEYS = ['mode', 'origin_date', 'target_row', 'horizon']


def load_prices():
    table = pd.read_csv(ROOT / 'data/processed/wti_daily.csv', parse_dates=['date']).sort_values('date').reset_index(drop=True)
    spot = pd.read_csv(ROOT / 'data/time_series/daily_spot.csv', parse_dates=['date']).sort_values('date').reset_index(drop=True)
    if not table['date'].equals(spot['date']) or not np.allclose(spot['target_wti'], table['target_wti']):
        raise ValueError('Daily spot and processed quotation rows do not align')
    if table['date'].duplicated().any():
        raise ValueError('Quotation dates must be unique')
    return table, spot


def forecast_origins(table, mode):
    fit = np.flatnonzero(table['date'].le(FIT_END).to_numpy())
    if not len(fit) or table['date'].iloc[fit[-1]] != pd.Timestamp(FIT_END):
        raise ValueError('The Sep30 fixed origin is absent from the quotation table')
    if mode == 'fixed_origin':
        return fit[-1:]
    if mode != 'daily_update':
        raise ValueError(f'Unknown mode: {mode}')
    return np.flatnonzero((table['date'].ge(FIT_END) & table['date'].lt(TEST_END)).to_numpy())


def forecast_horizons(table, origin, mode):
    if mode == 'fixed_origin':
        target = np.flatnonzero(table['date'].between(TEST_START, TEST_END).to_numpy())
        return sorted(set(HORIZONS) | {int(t - origin) for t in target if t > origin})
    return HORIZONS


def forecast_row(table, method, mode, origin, horizon, anchor):
    target = origin + horizon
    target_date = table['date'].iloc[target] if target < len(table) else pd.NaT
    matured = pd.notna(target_date) and pd.Timestamp(TEST_START) <= target_date <= pd.Timestamp(TEST_END)
    return {**PROVENANCE, 'mode': mode, 'origin_date': table['date'].iloc[origin],
            'target_date': target_date, 'target_row': target, 'horizon': horizon,
            'method': method, 'origin_price': anchor,
            'actual': float(table['target_wti'].iloc[target]) if matured else np.nan,
            'status': 'realized' if matured else 'unscored_no_test_outcome'}


def read_forecasts(path, values):
    """Reject legacy/mixed-protocol files rather than silently intersecting them."""
    return validate_forecasts(pd.read_csv(path), path, values)


def validate_forecasts(frame, path, values, provenance=PROVENANCE):
    """Validate canonical rows, including explicitly normalized external sources."""
    required = set(KEYS + ['target_date', 'origin_price', 'method', *provenance, *values])
    missing = required - set(frame)
    if missing:
        raise ValueError(f'{path}: missing frozen-protocol columns {sorted(missing)}; regenerate outputs')
    if frame.empty:
        raise ValueError(f'{path}: empty forecast file')
    for name, value in provenance.items():
        if frame[name].isna().any() or not frame[name].astype(str).eq(value).all():
            raise ValueError(f'{path}: incompatible {name}; regenerate from the approved frozen protocol')
    for name in ('origin_date', 'target_date'):
        frame[name] = pd.to_datetime(frame[name], errors='raise')
    if set(frame['mode']) != set(MODES):
        raise ValueError(f'{path}: both separate modes are required, got {set(frame["mode"])}')
    if frame.duplicated(KEYS + ['method']).any():
        raise ValueError(f'{path}: duplicate method/mode/origin/target/horizon rows')
    table, _ = load_prices()
    date_rows = pd.Series(np.arange(len(table)), index=table['date'])
    origin_rows = frame['origin_date'].map(date_rows)
    if origin_rows.isna().any() or not (origin_rows + frame['horizon']).eq(frame['target_row']).all():
        raise ValueError(f'{path}: horizon is not the quoted-row distance to target_row')
    anchors = observable_price(table)
    expected_anchor = origin_rows.map(anchors)
    if not np.isfinite(frame['origin_price']).all() or not np.allclose(frame['origin_price'], expected_anchor, atol=1e-6):
        raise ValueError(f'{path}: origin_price does not match the canonical publication-schedule anchor; regenerate outputs')
    expected_date = frame['target_row'].map(table['date'])
    if not (frame['target_date'].eq(expected_date) | (frame['target_date'].isna() & expected_date.isna())).all():
        raise ValueError(f'{path}: invented or misaligned target quotation dates')
    if not frame.loc[frame['mode'].eq('fixed_origin'), 'origin_date'].eq(pd.Timestamp(FIT_END)).all():
        raise ValueError(f'{path}: fixed-origin path must originate on Sep30')
    if not frame['origin_date'].between(FIT_END, TEST_END).all():
        raise ValueError(f'{path}: origin outside frozen October protocol')
    realized = frame['target_date'].between(TEST_START, TEST_END)
    for name in ('actual', 'actual_log_return'):
        if name in frame and frame.loc[~realized, name].notna().any():
            raise ValueError(f'{path}: outcomes outside October test targets must remain unscored')
    if 'actual' in frame:
        expected_actual = frame['target_row'].map(table['target_wti'])
        if frame.loc[realized, 'actual'].isna().any() or not np.allclose(frame.loc[realized, 'actual'], expected_actual[realized], atol=1e-6):
            raise ValueError(f'{path}: actual prices are missing or disagree with quotation targets')
    return frame

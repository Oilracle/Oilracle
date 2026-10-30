"""Publication-aware final forecasting shared by risk, residual and spread studies."""
import json

import numpy as np
import pandas as pd

from m3_ml.common.data import label_available_dates
from m3_ml.common.protocol import (FIT_END, HORIZONS, PROTOCOL_ID, TEST_END, TEST_START,
                                  TRAIN_END, VALIDATION_END, VALIDATION_START)
from m3_ml.common.walkforward import fixed_path_horizons


def path_horizons(dates, mode):
    if mode == 'daily_update':
        return HORIZONS
    return fixed_path_horizons(dates)


def selection_horizon(h):
    """Prespecified transfer: non-key path horizons use the next larger key's tuning."""
    return next((k for k in HORIZONS if k >= h), HORIZONS[-1])


def frame(dates, pos, h, mode, method, forecast, actual):
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    pos = np.asarray(pos, dtype=int)
    target = dates.reindex(pos + h).to_numpy()
    actual = np.asarray(actual, dtype=float).copy()
    scored = pd.Series(target).between(TEST_START, TEST_END).to_numpy() & np.isfinite(actual)
    actual[~scored] = np.nan
    return pd.DataFrame({'protocol_id': PROTOCOL_ID, 'mode': mode,
                         'origin_date': dates.iloc[pos].to_numpy(), 'target_date': target,
                         'target_row': pos + h, 'horizon': h, 'method': method,
                         'forecast': np.asarray(forecast, dtype=float), 'actual': actual,
                         'publication_policy': 'dated_archive_else_assumed_weekly_release',
                         'vintage_status': 'retrospective_current_vintage',
                         'status': np.where(scored, 'realized', 'unscored')})


def save_selection(path, selected):
    payload = {'protocol_id': PROTOCOL_ID, 'train_end': TRAIN_END,
               'validation_start': VALIDATION_START, 'validation_end': VALIDATION_END,
               'fit_end': FIT_END, 'test_start': TEST_START, 'test_end': TEST_END,
               'publication_policy': 'dated_archive_else_assumed_weekly_release',
               'vintage_status': 'retrospective_current_vintage',
               'path_parameter_policy': 'next_larger_key_horizon_validation_parameters',
               'selected': selected}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def load_selection(path):
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('protocol_id') != PROTOCOL_ID:
        raise ValueError(f'{path} is historical; rerun validation under {PROTOCOL_ID}')
    return payload['selected']


def available_return_mask(table, date):
    available = pd.to_datetime(label_available_dates(table))
    known = np.logical_and.accumulate(available.le(pd.Timestamp(date)).to_numpy())
    return known & np.roll(known, 1) & (np.arange(len(known)) > 0)

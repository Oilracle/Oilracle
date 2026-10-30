"""Check the complete frozen October handoff without fitting or selecting models."""
import hashlib
import json
import argparse
import ctypes
from pathlib import Path

import numpy as np
import pandas as pd

from m3_ml.common.data import load_table
from m3_ml.common.news import NEWS_POLICY_ID, news_metadata
from m3_ml.common.protocol import FIT_END, HORIZONS, MODES, PROTOCOL_ID, TEST_END, TEST_START
from m3_ml.common.walkforward import fixed_path_horizons, forecast_origins
from m3_ml.distribution import run as distribution
from m3_ml.distribution.quantiles import QUANTILE_COLUMNS
from m3_ml.distribution.tails import ALL_COLUMNS
from m3_ml.risk import events, volatility

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def wait_for_runner(runner_id):
    """Wait for this exact Windows training run, then require its final success."""
    status_path = ROOT / '.local/frozen_serial_status.json'
    current = json.loads(status_path.read_text())
    require(current['runner_pid'] == runner_id, 'status belongs to a different runner')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x00100000, False, runner_id)  # SYNCHRONIZE only
    if handle:
        print(f'Waiting for training runner {runner_id}', flush=True)
        try:
            while True:
                outcome = kernel.WaitForSingleObject(handle, 1000)
                if outcome == 0:
                    break
                require(outcome == 258, 'cannot wait for training process')
        finally:
            kernel.CloseHandle(handle)
    current = json.loads(status_path.read_text())
    require(current['runner_pid'] == runner_id and current['step'] == 'compare_analyze'
            and current['status'] == 'completed' and current['returncode'] == 0,
            f'training did not complete successfully: {current}')


def selection(path, expected):
    saved = json.loads(path.read_text(encoding='utf-8'))
    for key, value in {'protocol_id': PROTOCOL_ID, 'fit_end': FIT_END,
                       'validation_end': FIT_END}.items():
        require(saved.get(key) == value, f'{path}: invalid {key}')
    require(set(saved['selected']) == set(expected), f'{path}: incomplete selection grid')
    return saved


def forecasts(path, expected_methods, values=('forecast',), sides=None, omit_h1=()):
    frame = pd.read_csv(path, parse_dates=['origin_date', 'target_date'])
    require(frame.protocol_id.eq(PROTOCOL_ID).all(), f'{path}: wrong protocol')
    require(set(frame['method']) == set(expected_methods), f'{path}: incomplete methods')
    require(set(frame['mode']) == set(MODES), f'{path}: incomplete modes')
    require(np.isfinite(frame[list(values)].to_numpy()).all(), f'{path}: nonfinite forecasts')
    keys = ['method', 'mode', 'origin_date', 'horizon'] + (['side'] if sides else [])
    require(not frame.duplicated(keys).any(), f'{path}: duplicate forecasts')
    table = load_table()
    dates = table.date
    observations = []
    for method in expected_methods:
        for side in sides or (None,):
            group = frame[frame.method.eq(method)]
            if side is not None:
                group = group[group.side.eq(side)]
            for mode in MODES:
                part = group[group['mode'].eq(mode)]
                positions = forecast_origins(dates, mode)
                horizons = fixed_path_horizons(dates) if mode == 'fixed_origin' else HORIZONS
                expected = {(dates.iloc[o], h) for o in positions for h in horizons
                            if not (h == 1 and method in omit_h1)}
                require(set(zip(part.origin_date, part.horizon)) == expected,
                        f'{path}: incomplete {method}/{side}/{mode} origins/horizons')
                for h in HORIZONS:
                    if h == 1 and method in omit_h1:
                        continue
                    scored = part[part.horizon.eq(h)].actual.notna().sum()
                    expected_n = ({1: 17, 5: 13, 20: 0, 60: 0} if mode == 'daily_update'
                                  else {1: 1, 5: 1, 20: 0, 60: 0})[h]
                    require(scored == expected_n, f'{path}: {method}/{mode}/h{h} n={scored}')
                    observations.append(dict(method=method, side=side, mode=mode, h=h, n=int(scored)))
            fixed = group[group['mode'].eq('fixed_origin')].set_index('horizon')
            first = group[group['mode'].eq('daily_update') & group.origin_date.eq(FIT_END)].set_index('horizon')
            require(np.allclose(fixed.loc[first.index, list(values)], first[list(values)],
                                rtol=1e-10, atol=1e-10), f'{path}: first-origin modes disagree')
    require(frame.loc[~frame.target_date.between(TEST_START, TEST_END), 'actual'].isna().all(),
            f'{path}: outcomes outside final window were scored')
    if all(c in frame for c in QUANTILE_COLUMNS):
        require((np.diff(frame[QUANTILE_COLUMNS], axis=1) >= -1e-10).all(), f'{path}: crossed quantiles')
    return frame, dict(file=str(path.relative_to(ROOT)), rows=len(frame),
                       methods=sorted(expected_methods), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       realized_counts=observations)


def main():
    records = {}
    dates = load_table().date
    path_h = fixed_path_horizons(dates)
    dist_saved = selection(distribution.OUT / 'selected.json',
                           [f'{m}/h{h}' for m in distribution.METHODS for h in path_h
                            if not (m == 'c_mixture' and h == 1)])
    require(dist_saved.get('news') == news_metadata(), 'distribution news snapshot changed')
    dist, records['distribution'] = forecasts(distribution.OUT / 'test_quantiles.csv',
                                              [*distribution.METHODS, 'ensemble'], ALL_COLUMNS,
                                              omit_h1=('c_mixture',))
    require(dist.news_policy_id.eq(NEWS_POLICY_ID).all(), 'wrong final news policy')
    selection(volatility.OUT / 'volatility_selected.json',
              [f'{m}/h{h}' for m in volatility.METHODS for h in HORIZONS])
    vol, records['volatility'] = forecasts(volatility.OUT / 'volatility_test_forecasts.csv',
                                          volatility.METHODS, ('sigma2',))
    require(vol.sigma2.gt(0).all(), 'nonpositive variance')
    selection(events.OUT / 'events_selected.json',
              [f'{m}/h{h}/{s}' for m in events.MODELS for h in HORIZONS for s in events.SIDES])
    event, records['events'] = forecasts(events.OUT / 'events_test_probabilities.csv',
                                         [*events.MODELS, *[f'dist_{m}' for m in dist.method.unique()]],
                                         ('prob',), events.SIDES, ('dist_c_mixture',))
    require(event.prob.between(0, 1).all(), 'event probabilities outside [0, 1]')
    residual = ROOT / 'm3_ml/residual/outputs/frozen_october_2026'
    selection(residual / 'residual_params.json',
              [f'{m}/h{h}' for m in ('elastic_net', 'random_forest', 'lightgbm', 'scale', 'ewma') for h in HORIZONS])
    _, records['residual_point'] = forecasts(residual / 'residual_forecasts.csv',
                                             ('random_walk', 'seasonality', 'arima', 'base',
                                              'elastic_net', 'random_forest', 'lightgbm'))
    _, records['residual_quantiles'] = forecasts(residual / 'residual_quantiles.csv',
                                                 ('base_hs', 'elastic_net_hs', 'random_forest_hs',
                                                  'lightgbm_hs', 'lightgbm_scale', 'ewma'), QUANTILE_COLUMNS)
    report = {'protocol_id': PROTOCOL_ID, 'fit_end': FIT_END, 'test_start': TEST_START,
              'test_end': TEST_END, 'status': 'passed', 'artifacts': records,
              'scope': 'remaining distribution/risk/residual outputs; unified compare validates M1/M2/LSTM/tabular separately'}
    destination = ROOT / '.local/final_refit_audit.json'
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({name: {k: v for k, v in item.items() if k != 'realized_counts'}
                      for name, item in records.items()}, indent=2))
    print('Passed:', destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-runner', type=int)
    args = parser.parse_args()
    if args.wait_for_runner:
        wait_for_runner(args.wait_for_runner)
    main()

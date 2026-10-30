"""Frozen Kronos validation/test; real local checkpoints only, no model downloads.

Run stage-1 training, validate, stage-2 training, then test. All selection ends
September 30. Inference uses CUDA when available, with bounded path batches.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from m3_ml.common.data import load_table, label_available_dates, observable_price
from m3_ml.common.protocol import (TRAIN_END, VALIDATION_START, VALIDATION_END,
                                   FIT_END, TEST_START, TEST_END, HORIZONS, MODES, PROTOCOL_ID)
from m3_ml.distribution.quantiles import crps
from m3_ml.distribution.tails import ALL_TAUS, MAIN
from .config import (HORIZON, LOOKBACK, MODEL, PATHS_TEST, PATHS_VALIDATION,
                     SAMPLING_GRID, SEED, TOKENIZER, WORK)
from .corpus import path
from .dataset import normalize, to_arrays
from .generate import return_quantiles, sample_paths_cached
from .universe import WTI_INPUT
from .vendor import Kronos, KronosTokenizer

OUT = Path(__file__).resolve().parent / 'outputs' / 'frozen_october_2026'
BDAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
TARGET_SEQUENCES = 200


def resolve_device(device='auto'):
    return ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device


def load_models(stage, device='auto'):
    device = resolve_device(device)
    folder = WORK / f'stage{stage}'
    metadata = json.loads((folder / 'metadata.json').read_text())
    if metadata.get('protocol_id') != PROTOCOL_ID or metadata.get('stage') != stage:
        raise ValueError('Train genuine approved-protocol checkpoints before inference.')
    tokenizer, model = KronosTokenizer(**TOKENIZER), Kronos(**MODEL)
    tokenizer.load_state_dict(torch.load(folder / 'tokenizer.pt', map_location=device, weights_only=True))
    model.load_state_dict(torch.load(folder / 'predictor.pt', map_location=device, weights_only=True))
    return tokenizer.to(device).eval(), model.to(device).eval()


def contexts(origin_dates, stats=None, target_dates=None, return_offsets=False):
    """Past futures bars only; frozen normalization and explicit calendar scenarios.

    CL daily closes are assumed available after that exchange session's close;
    this is a market-bar assumption, NOT the spot WPSR publication rule. Future
    time stamps use observed target-row dates where supplied, then an explicitly
    assumed business-day scenario. No future OHLCV values enter the model.
    """
    frame = pd.read_parquet(path(WTI_INPUT)).sort_index()
    feats, stamps = to_arrays(frame)
    dates = frame.index
    xs, sc, sf, means, stds, offsets = [], [], [], [], [], []
    for j, d in enumerate(origin_dates):
        end = int(dates.searchsorted(pd.Timestamp(d), side='right'))
        if end < LOOKBACK:
            raise ValueError(f'Insufficient {WTI_INPUT} context before {d}.')
        x, (mean, std) = normalize(feats[end - LOOKBACK:end], stats)
        future = [] if target_dates is None else list(pd.to_datetime(target_dates[j]).dropna())
        # Missing exchange sessions are simulated, never filled with later bars.
        bridge = list(pd.date_range(dates[end - 1] + BDAY, pd.Timestamp(d), freq=BDAY))
        if not future:
            future = [pd.Timestamp(d) + BDAY]
        while len(future) < HORIZON:
            future.append(future[-1] + BDAY)
        future = bridge + future[:HORIZON]
        offsets.append(len(bridge))
        xs.append(x)
        sc.append(stamps[end - LOOKBACK:end])
        sf.append(future)
        means.append(mean[3])
        stds.append(std[3])
    length = max(map(len, sf))
    future_stamps = []
    for future in sf:
        while len(future) < length:
            future.append(future[-1] + BDAY)
        idx = pd.DatetimeIndex(future)
        future_stamps.append(np.column_stack([np.zeros(length), np.zeros(length), idx.weekday, idx.day, idx.month]))
    result = (np.stack(xs), np.stack(sc), np.stack(future_stamps).astype(np.int64),
              np.array(means), np.array(stds))
    return (*result, offsets) if return_offsets else result


def forecast(models, origin_dates, setting, paths, device='auto', stats=None,
             target_dates=None, horizons=HORIZONS, batch_sequences=TARGET_SEQUENCES):
    tokenizer, model = models
    if stats is None:
        raise ValueError('Inference requires checkpoint-fitted frozen normalization.')
    device = resolve_device(device)
    if paths < 1 or batch_sequences < 1:
        raise ValueError('paths and batch_sequences must be positive')
    x, sc, sf, mean, std, offsets = contexts(origin_dates, stats, target_dates, return_offsets=True)
    per_batch = max(1, batch_sequences // paths)
    parts = {h: [] for h in horizons}
    batches = (len(x) + per_batch - 1) // per_batch
    completed = 0
    print(f'inference device={device} origins={len(x)} paths/origin={paths} '
          f'total_paths={len(x) * paths} sequence_cap={batch_sequences}', flush=True)
    for batch, i in enumerate(range(0, len(x), per_batch)):
        sl = slice(i, min(i + per_batch, len(x)))
        count = sl.stop - sl.start
        path_batch = max(1, batch_sequences // count)
        chunks = []
        for start in range(0, paths, path_batch):
            chunk_paths = min(path_batch, paths - start)
            chunks.append(sample_paths_cached(
                tokenizer, model, x[sl], sc[sl], sf[sl], setting['T'], setting['top_p'],
                chunk_paths, device, seed=SEED + completed))
            completed += 1
            print(f'batch={batch + 1}/{batches} origins={i + 1}-{sl.stop}/{len(x)} '
                  f'paths/origin={start + chunk_paths}/{paths} '
                  f'completed_paths={i * paths + count * (start + chunk_paths)}/{len(x) * paths}',
                  flush=True)
        close = np.concatenate(chunks, axis=1) if len(chunks) > 1 else chunks[0]
        for j, offset in enumerate(offsets[i:sl.stop]):
            path = close[j:j + 1, :, offset:offset + HORIZON + 1]
            quantiles = return_quantiles(path, mean[i + j:i + j + 1], std[i + j:i + j + 1], horizons, ALL_TAUS)
            for h, q in quantiles.items():
                parts[h].append(q)
    return {h: np.vstack(v) for h, v in parts.items()}


def frozen_stats(stage):
    saved = np.load(WORK / f'stage{stage}' / 'normalization.npz')
    return saved['mean'], saved['std']


def request_calendar(table, origin):
    return table.date.iloc[origin + 1:origin + HORIZON + 1].tolist()


def validation_origins(table, h):
    dates = pd.to_datetime(table.date)
    available = label_available_dates(table)
    anchor = table.wti_spot_last_published.to_numpy(float)
    rows = []
    for i in range(len(table) - h):
        target = i + h
        if (dates.iloc[target] >= pd.Timestamp(VALIDATION_START) and
                dates.iloc[i + 1] >= pd.Timestamp(VALIDATION_START) and
                dates.iloc[target] <= pd.Timestamp(VALIDATION_END) and
                pd.notna(available.iloc[target]) and available.iloc[target] <= pd.Timestamp(FIT_END) and
                np.isfinite(anchor[i]) and anchor[i] > 0 and
                np.isfinite(table.target_wti.iloc[target]) and table.target_wti.iloc[target] > 0):
            rows.append(i)
    return np.array(rows, dtype=int)


def validate(device, batch_sequences=TARGET_SEQUENCES):
    table = load_table()
    table['wti_spot_last_published'] = observable_price(table).to_numpy()
    models, stats = load_models(1, device), frozen_stats(1)
    folds = {h: validation_origins(table, h) for h in HORIZONS}
    union = np.unique(np.concatenate(list(folds.values())))
    if not len(union):
        raise ValueError('No publication-verified historical validation targets.')
    calendars = [request_calendar(table, i) for i in union]
    where = {p: j for j, p in enumerate(union)}
    rows = []
    for grid, setting in enumerate(SAMPLING_GRID):
        print(f'validation grid={grid + 1}/{len(SAMPLING_GRID)} setting={setting} '
              f'origins={len(union)} paths/origin={PATHS_VALIDATION}', flush=True)
        quantiles = forecast(models, table.date.iloc[union], setting, PATHS_VALIDATION,
                             device, stats, calendars, batch_sequences=batch_sequences)
        for h, origins in folds.items():
            q = quantiles[h][[where[p] for p in origins]][:, MAIN]
            y = np.log(table.target_wti.iloc[origins + h].to_numpy() /
                       table.wti_spot_last_published.iloc[origins].to_numpy())
            rows.append(dict(grid=grid, **setting, h=h, n=len(origins),
                             crps=float(np.mean(crps(q, y))), protocol_id=PROTOCOL_ID))
        print(setting, 'historical validation complete', flush=True)
    scores = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUT / 'validation_scores.csv', index=False)
    best_grid = int(scores.groupby('grid').crps.mean().idxmin())
    best_setting = SAMPLING_GRID[best_grid]
    selected = dict(protocol_id=PROTOCOL_ID, train_end=TRAIN_END,
                    validation_start=VALIDATION_START, validation_end=VALIDATION_END,
                    sampler=dict(T=float(best_setting['T']), top_p=float(best_setting['top_p'])))
    (OUT / 'selected.json').write_text(json.dumps(selected, indent=2) + '\n')


def test(device, paths=None, mode=None, latest=False, batch_sequences=TARGET_SEQUENCES):
    table = load_table()
    table['wti_spot_last_published'] = observable_price(table).to_numpy()
    models, stats = load_models(2, device), frozen_stats(2)
    selected = json.loads((OUT / 'selected.json').read_text())
    if selected.get('protocol_id') != PROTOCOL_ID or selected.get('validation_end') != VALIDATION_END:
        raise ValueError('Sampling settings must be selected on approved historical validation.')
    fit = int(table.date.searchsorted(pd.Timestamp(FIT_END), side='right')) - 1
    if table.date.iloc[fit] != pd.Timestamp(FIT_END):
        raise ValueError('Missing fixed-origin quoted row.')
    stop = int(table.date.searchsorted(pd.Timestamp(TEST_END), side='right')) - 1
    rows = []
    modes = ('daily_update',) if latest else ((mode,) if mode else MODES)
    for current_mode in modes:
        origins = np.array([len(table) - 1] if latest else
                           ([fit] if current_mode == 'fixed_origin' else range(fit, stop)), dtype=int)
        calendars = [request_calendar(table, i) for i in origins]
        horizons = tuple(range(1, HORIZON + 1)) if current_mode == 'fixed_origin' else HORIZONS
        # One validation-selected sampler yields a single coherent path.
        settings = {h: selected['sampler'] for h in horizons}
        for encoded in sorted({json.dumps(v, sort_keys=True) for v in settings.values()}):
            setting = json.loads(encoded)
            hs = tuple(h for h in horizons if settings[h] == setting)
            q = forecast(models, table.date.iloc[origins], setting, paths or PATHS_TEST,
                         device, stats, calendars, hs, batch_sequences=batch_sequences)
            for h in hs:
                for j, i in enumerate(origins):
                    target = i + h
                    date = table.date.iloc[target] if target < len(table) else pd.NaT
                    actual = (float(table.target_wti.iloc[target]) if pd.notna(date) and
                              pd.Timestamp(TEST_START) <= date <= pd.Timestamp(TEST_END) else np.nan)
                    anchor = float(table.wti_spot_last_published.iloc[i])
                    prices = anchor * np.exp(q[h][j])
                    row = dict(mode=current_mode, origin_date=table.date.iloc[i], target_date=date,
                               target_row=target, horizon=h, method='kronos', origin_price=anchor,
                               forecast=float(np.interp(.5, ALL_TAUS, prices)), actual=actual,
                               protocol_id=PROTOCOL_ID, fit_end=FIT_END, train_end=TRAIN_END,
                               validation_start=VALIDATION_START, validation_end=VALIDATION_END,
                               calendar_scenario='observed quoted rows then assumed federal business days')
                    row.update({f'q{tau:g}': float(value) for tau, value in zip(ALL_TAUS, prices)})
                    rows.append(row)
    result = pd.DataFrame(rows).sort_values(['mode', 'origin_date', 'horizon'])
    if latest:
        result.to_csv(OUT / 'latest_forecast.csv', index=False, date_format='%Y-%m-%d')
        print('wrote', OUT / 'latest_forecast.csv', result.shape)
        return result
    result.to_csv(OUT / 'test_quantiles.csv', index=False, date_format='%Y-%m-%d')
    result.to_csv(OUT / 'test_predictions.csv', index=False, date_format='%Y-%m-%d')
    metrics = []
    for current_mode in result['mode'].unique():
        for h in HORIZONS:
            g = result[(result['mode'] == current_mode) & (result.horizon == h)].dropna(subset=['actual'])
            error = g.forecast - g.actual
            metrics.append(dict(mode=current_mode, horizon=h, method='kronos', n=len(g),
                                protocol_id=PROTOCOL_ID, fit_end=FIT_END,
                                mae=float(error.abs().mean()), rmse=float(np.sqrt((error ** 2).mean()))))
    pd.DataFrame(metrics).to_csv(OUT / 'test_metrics.csv', index=False)
    print('wrote', OUT / 'test_quantiles.csv', result.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['validate', 'test', 'forecast'])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--batch-sequences', type=int, default=TARGET_SEQUENCES,
                        help='Maximum concurrent origin/path sequences; does not change total Monte Carlo paths.')
    parser.add_argument('--paths', type=int, help='Test Monte Carlo paths; smaller positive count for smoke only.')
    parser.add_argument('--mode', choices=MODES)
    args = parser.parse_args()
    if args.paths is not None and args.paths < 1:
        parser.error('--paths must be positive')
    if args.batch_sequences < 1:
        parser.error('--batch-sequences must be positive')
    args.device = resolve_device(args.device)
    if args.stage == 'validate':
        validate(args.device, args.batch_sequences)
    else:
        test(args.device, args.paths, args.mode, latest=args.stage == 'forecast',
             batch_sequences=args.batch_sequences)

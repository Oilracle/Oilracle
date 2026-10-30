"""Publication-aware validation and frozen fixed-origin/daily-update forecasts.

Run: uv run python -m m3_ml.tabular.run_direct [--denoise] [--news]
Legacy outputs remain historical; new protocol artifacts have a dedicated folder.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from m3_ml.common.compute import bounded_cpu, report_compute
from m3_ml.common.data import PRICE, VALIDATION, load_table, observable_price, targets
from m3_ml.common.features import build_features
from m3_ml.common.metrics import summarize
from m3_ml.common.news import news_availability, news_metadata, require_complete_features
from m3_ml.common.protocol import (FIT_END, HORIZONS, MODES, PROTOCOL_ID, TEST_START, TEST_END,
                                   TRAIN_END, VALIDATION_START, VALIDATION_END)
from m3_ml.common.walkforward import (final_training_positions, fixed_path_horizons,
                                      forecast_origins, walk_forward)
from m3_ml.tabular.models import MODELS

OUT = Path(__file__).resolve().parent / 'outputs/frozen_october_2026'
REFIT_VALIDATION = 63


def forecast(make_model, params, X, y, dates, period, horizon, refit_every):
    """Historical validation only; final test uses frozen_forecast instead."""
    preds = {}
    folds = list(walk_forward(dates, X.notna().all(axis=1), y.notna(),
                              *period, horizon, refit_every))
    for fold, (_, train, origins) in enumerate(folds, 1):
        print(f'tabular h={horizon} fold={fold}/{len(folds)} '
              f'origin={dates.iloc[origins[0]]:%Y-%m-%d}', flush=True)
        model = make_model(**params).fit(X.iloc[train], y.iloc[train])
        preds.update(zip(origins, model.predict(X.iloc[origins])))
    mature = final_training_positions(dates, X.notna().all(axis=1), y.notna(), horizon, VALIDATION_END)
    return pd.Series(preds, dtype=float).reindex(mature).dropna()


def validation_rmse(preds, y, horizon, last_position):
    """Purge validation targets extending beyond the selection cutoff."""
    preds = preds[preds.index + horizon <= last_position]
    err = (preds - y.reindex(preds.index)).dropna()
    if err.empty:
        raise ValueError('no mature validation targets')
    return float(np.sqrt((err ** 2).mean()))


def frozen_forecast(make_model, params, X, y, dates, horizon, origins):
    """Fit estimator and its Pipeline preprocessing exactly once before October."""
    train = final_training_positions(dates, X.notna().all(axis=1), y.notna(), horizon)
    model = make_model(**params).fit(X.iloc[train], y.iloc[train])
    return pd.Series(model.predict(X.iloc[origins]), index=origins)


def make_frame(table, y, horizon, method, preds, mode, variant):
    pos = preds.index.to_numpy(dtype=int)
    target_row = pos + horizon
    dates = table['date']
    target_date = dates.reindex(target_row).to_numpy()
    realized = pd.Series(pd.to_datetime(target_date)).between(TEST_START, TEST_END).to_numpy()
    actual_return = y.reindex(pos).to_numpy()
    actual_return = np.where(realized, actual_return, np.nan)
    anchor = observable_price(table).iloc[pos].to_numpy()
    out = pd.DataFrame({'mode': mode, 'origin_date': dates.iloc[pos].to_numpy(),
                        'target_date': target_date, 'target_row': target_row,
                        'horizon': horizon, 'method': method, 'model': method, 'variant': variant,
                        'origin_price': anchor, 'forecast_log_return': preds.to_numpy(),
                        'actual_log_return': actual_return})
    out['forecast'] = anchor * np.exp(out['forecast_log_return'])
    out['actual'] = anchor * np.exp(out['actual_log_return'])
    out['status'] = np.where(out['actual'].notna(), 'realized', 'unreached')
    for name, value in {'protocol_id': PROTOCOL_ID, 'fit_end': FIT_END, 'train_end': TRAIN_END,
                        'validation_start': VALIDATION_START, 'validation_end': VALIDATION_END,
                        'publication_policy': 'dated_archive_else_assumed_weekly_release'}.items():
        out[name] = value
    return out


@bounded_cpu
def main(denoise=False, news=False):
    report_compute()
    variant = 'direct' + ('_denoised' if denoise else '') + ('_news' if news else '')
    table = load_table()
    dates = table['date']
    path_horizons = fixed_path_horizons(dates)
    X = build_features(table, denoise=denoise, news=news)
    Y = targets(table, path_horizons)
    validation_end = int(np.flatnonzero(dates <= VALIDATION_END)[-1])
    rows, chosen = [], {}
    fixed = forecast_origins(dates, 'fixed_origin')
    daily = forecast_origins(dates, 'daily_update')
    require_complete_features(X, dates, daily)
    news_info = news_metadata() if news else None
    for h in path_horizons:
        y = Y[h]
        origins = daily if h in HORIZONS else fixed
        train = final_training_positions(dates, X.notna().all(axis=1), y.notna(), h)
        results = {'random_walk': pd.Series(0.0, index=origins),
                   'historical_mean': pd.Series(y.iloc[train].mean(), index=origins)}
        for name, (make_model, param_grid) in MODELS.items():
            scores = []
            for candidate, p in enumerate(param_grid, 1):
                print(f'tabular variant={variant} method={name} h={h} '
                      f'candidate={candidate}/{len(param_grid)} params={p}', flush=True)
                scores.append(validation_rmse(forecast(make_model, p, X, y, dates, VALIDATION, h,
                                                        REFIT_VALIDATION), y, h, validation_end))
            best_index = int(np.argmin(scores))
            best = param_grid[best_index]
            chosen[f'{name}/h{h}'] = {'params': best, 'validation_rmse': scores[best_index]}
            print(f'tabular final method={name} h={h}', flush=True)
            results[name] = frozen_forecast(make_model, best, X, y, dates, h, origins)
            print(f'h={h:2d} {name:14s} best={best}', flush=True)
        for name, preds in results.items():
            rows.append(make_frame(table, y, h, name, preds.reindex(fixed), 'fixed_origin', variant))
            if h in HORIZONS:
                rows.append(make_frame(table, y, h, name, preds, 'daily_update', variant))
    out = pd.concat(rows, ignore_index=True)
    if news_info is not None:
        if news_info != news_metadata():
            raise ValueError('news source snapshot changed during fitting; rerun with fixed inputs')
        out['news_policy_id'] = news_info['news_policy_id']
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / f'{variant}_forecasts.csv', index=False, date_format='%Y-%m-%d', lineterminator='\n')
    metadata = {'protocol_id': PROTOCOL_ID, 'fit_end': FIT_END, 'train_end': TRAIN_END,
                'validation_start': VALIDATION_START, 'validation_end': VALIDATION_END,
                'publication_policy': 'dated_archive_else_assumed_weekly_release', 'selected': chosen}
    if news_info is not None:
        metadata['news'] = news_info
        news_availability(dates).to_csv(OUT / f'{variant}_availability.csv', index=False,
                                        date_format='%Y-%m-%d', lineterminator='\n')
    (OUT / f'{variant}_params.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    (OUT / 'protocol.json').write_text(json.dumps({k: v for k, v in metadata.items() if k not in ('selected', 'news')},
                                                indent=2) + '\n', encoding='utf-8')
    metrics = summarize(out)
    metrics['protocol_id'] = PROTOCOL_ID
    if news_info is not None:
        metrics['news_policy_id'] = news_info['news_policy_id']
    metrics.to_csv(OUT / f'{variant}_metrics.csv', lineterminator='\n')
    print(metrics.to_string())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--news', action='store_true')
    args = parser.parse_args()
    main(args.denoise, args.news)

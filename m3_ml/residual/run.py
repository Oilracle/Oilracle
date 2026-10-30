"""Select residual corrections on historical validation, then freeze October fits.

Run: uv run python -m m3_ml.residual.run
Outputs are isolated from historical results in outputs/frozen_october_2026/.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from m3_ml.common.compute import bounded_cpu, report_compute
from m3_ml.common.execution import ValidationCache, run_jobs
from m3_ml.common.data import HORIZONS, VALIDATION, load_table, observable_price, safe_log_price, targets
from m3_ml.common.features import build_features
from m3_ml.common.protocol import MODES, PROTOCOL_ID, VALIDATION_END
from m3_ml.common.walkforward import final_training_positions, forecast_origins as final_origins, walk_forward
from m3_ml.distribution.benchmarks import ewma_sigma
from m3_ml.distribution.quantiles import TAUS
from m3_ml.residual.base import build_base
from m3_ml.risk.protocol import available_return_mask, frame, path_horizons, save_selection, selection_horizon
from m3_ml.tabular.models import MODELS, elastic_net
from m3_ml.tabular.run_direct import REFIT_VALIDATION, validation_rmse

OUT = Path(__file__).resolve().parent / 'outputs' / 'frozen_october_2026'
SCALE_GRID = [{'alpha_ratio': a, 'l1_ratio': r} for a in (1.0, 0.1, 0.01) for r in (0.1, 0.5, 0.9)]
EWMA_GRID = [{'lam': lam, 'shape': shape} for lam in (0.94, 0.97, 0.99) for shape in ('normal', 'empirical')]
MIN_SHOCKS = 60


def _predict(make_model, params, X, y, dates, period, horizon, refit_every):
    preds = {}
    folds = list(walk_forward(dates, X.notna().all(axis=1), y.notna(), *period, horizon, refit_every))
    for fold, (_, train, origins) in enumerate(folds, 1):
        print(f'residual h={horizon} fold={fold}/{len(folds)} '
              f'origin={dates.iloc[origins[0]]:%Y-%m-%d}', flush=True)
        if len(train) >= MIN_SHOCKS:
            model = make_model(**params).fit(X.iloc[train], y.iloc[train])
            preds.update(zip(origins, model.predict(X.iloc[origins])))
    mature = final_training_positions(dates, X.notna().all(axis=1), y.notna(), horizon, VALIDATION_END)
    return pd.Series(preds, dtype=float).reindex(mature).dropna()


def _quantiles(location, shocks, scale, taus):
    shocks = np.asarray(shocks, dtype=float)
    shocks = shocks[np.isfinite(shocks)]
    if len(shocks) < MIN_SHOCKS:
        return None
    width = np.mean(np.abs(shocks))
    if not np.isfinite(width) or width <= 0:
        return None
    return np.asarray(location)[:, None] + np.asarray(scale)[:, None] * np.quantile(shocks, taus)[None, :] / width


def _ewma_state(table, lam):
    returns = safe_log_price(table['target_wti']).diff()
    sigma, gaps, offsets = [], [], []
    anchor = safe_log_price(observable_price(table))
    price = safe_log_price(table['target_wti'])
    for p, date in enumerate(table['date']):
        if p == 0 or (p + 1) % 500 == 0 or p + 1 == len(table):
            print(f'residual EWMA lam={lam} origin={p + 1}/{len(table)}', flush=True)
        mask = available_return_mask(table, date)
        seen = np.flatnonzero(mask)
        if not len(seen):
            sigma.append(np.nan)
            gaps.append(0)
            offsets.append(np.nan)
            continue
        last = int(seen[-1])
        # Keep the quote grid. Invalid non-positive price returns have already
        # been causally repaired by safe_log_price, not deleted and renumbered.
        r = returns.iloc[:last + 1].reset_index(drop=True)
        sigma.append(float(ewma_sigma(r, lam).iloc[-1]))
        gaps.append(p - last)
        offsets.append(float(price.iloc[last] - anchor.iloc[p]))
    return pd.Series(sigma), np.asarray(gaps), pd.Series(offsets)


def _ewma_q(state, y, train, origins, h, shape):
    sigma, gaps, offset = state
    scales = sigma * np.sqrt(h + gaps)
    z = stats.norm.ppf(TAUS) if shape == 'normal' else np.quantile(((y - offset) / scales).iloc[train].dropna(), TAUS)
    return offset.iloc[origins].to_numpy()[:, None] + scales.iloc[origins].to_numpy()[:, None] * z[None, :]


def _ewma_score(state, y, X, dates, h, shape):
    losses = []
    mature = final_training_positions(dates, X.notna().all(axis=1), y.notna(), h, VALIDATION_END)
    for _, train, origins in walk_forward(dates, X.notna().all(axis=1), y.notna(), *VALIDATION, h, REFIT_VALIDATION):
        origins = origins[np.isin(origins, mature)]
        if not len(origins) or len(train) < MIN_SHOCKS:
            continue
        q = _ewma_q(state, y, train, origins, h, shape)
        error = y.iloc[origins].to_numpy()[:, None] - q
        losses.extend(np.mean(np.maximum(np.asarray(TAUS) * error, (np.asarray(TAUS) - 1) * error), axis=1))
    return float(np.mean(losses)) if losses else np.inf


def _validation_job(context, task):
    X, Y, base, dates, states, end = context
    method, params, h = task
    if method == 'ewma':
        return _ewma_score(states[params['lam']], Y[h], X, dates, h, params['shape'])
    target = Y[h] - base[f'base_{h}']
    make = elastic_net if method == 'scale' else MODELS[method][0]
    if method == 'scale':
        target = target.abs()
    predictions = _predict(make, params, X, target, dates, VALIDATION, h, REFIT_VALIDATION)
    if method == 'scale':
        predictions = predictions.abs()
    return validation_rmse(predictions, target, h, end)


@bounded_cpu
def run():
    report_compute()
    table = load_table()
    dates, X = table['date'], build_features(table)
    base = build_base(table)
    Y = targets(table, horizons=tuple(range(1, 61)))
    anchor = observable_price(table)
    end = int(np.flatnonzero(dates <= VALIDATION_END)[-1])
    states = {}
    for lam in (0.94, 0.97, 0.99):
        print(f'residual EWMA state lam={lam}', flush=True)
        states[lam] = _ewma_state(table, lam)
    selected = {}
    grids = {name: grid for name, (_, grid) in MODELS.items()}
    grids.update(scale=SCALE_GRID, ewma=EWMA_GRID)
    tasks = [(name, params, h) for h in HORIZONS for name, grid in grids.items() for params in grid]
    cache = ValidationCache('residual_scores', {'table': table, 'X': X, 'Y': Y, 'base': base})
    scores = iter(run_jobs(_validation_job, tasks, (X, Y, base, dates, states, end),
                           'residual validation', cache))
    for h in HORIZONS:
        for name, grid in grids.items():
            values = [next(scores) for _ in grid]
            best = int(np.argmin(values))
            metric = 'validation_pinball' if name == 'ewma' else 'validation_rmse'
            selected[f'{name}/h{h}'] = {'params': grid[best], metric: float(values[best])}
    points, quantiles = [], []
    # Fit each horizon exactly once; both modes predict with the same estimator.
    all_h = sorted(set(path_horizons(dates, 'fixed_origin')) | set(HORIZONS))
    for index, h in enumerate(all_h, 1):
        print(f'residual final horizon={index}/{len(all_h)} h={h}', flush=True)
        y, base_h = Y[h], base[f'base_{h}']
        residual = y - base_h
        ok = X.notna().all(axis=1) & base_h.notna()
        train = final_training_positions(dates, ok, residual.notna(), h)
        if len(train) < MIN_SHOCKS:
            raise ValueError(f'Insufficient mature residual history for horizon {h}')
        key = selection_horizon(h)
        models = {}
        for name, (make, _) in MODELS.items():
            print(f'residual final method={name} h={h}', flush=True)
            models[name] = make(**selected[f'{name}/h{key}']['params']).fit(X.iloc[train], residual.iloc[train])
        scale_model = elastic_net(**selected[f'scale/h{key}']['params']).fit(X.iloc[train], residual.abs().iloc[train])
        shocks = residual.iloc[train].to_numpy()
        width = np.mean(np.abs(shocks))
        ewma = selected[f'ewma/h{key}']['params']
        for mode in MODES:
            if h not in path_horizons(dates, mode):
                continue
            pos = final_origins(dates, mode)
            center = base_h.iloc[pos].to_numpy()
            point = {'random_walk': np.zeros(len(pos)), 'seasonality': base[f'seas_{h}'].iloc[pos].to_numpy(),
                     'arima': base['ts'].iloc[pos].to_numpy(), 'base': center}
            point.update({name: center + model.predict(X.iloc[pos]) for name, model in models.items()})
            for name, prediction in point.items():
                block = frame(dates, pos, h, mode, name, anchor.iloc[pos].to_numpy() * np.exp(prediction),
                              anchor.iloc[pos].to_numpy() * np.exp(y.iloc[pos].to_numpy()))
                block['model'], block['origin_price'] = name, anchor.iloc[pos].to_numpy()
                block['forecast_log_return'] = prediction
                block['actual_log_return'] = y.iloc[pos].to_numpy()
                block.loc[block['actual'].isna(), 'actual_log_return'] = np.nan
                block['calendar_assumption'] = 'observed_quote_dates_then_weekday_scenario'
                points.append(block)
            qmethods = {f'{name}_hs': _quantiles(prediction, shocks, np.full(len(pos), width), TAUS)
                        for name, prediction in point.items() if name in ('base', *MODELS)}
            qmethods['lightgbm_scale'] = _quantiles(point['lightgbm'], shocks,
                                                  np.abs(scale_model.predict(X.iloc[pos])), TAUS)
            qmethods['ewma'] = _ewma_q(states[ewma['lam']], y, train, pos, h, ewma['shape'])
            for method, q in qmethods.items():
                if q is None:
                    raise ValueError(f'Degenerate training shocks for h={h}')
                block = frame(dates, pos, h, mode, method, np.median(q, axis=1), y.iloc[pos])
                block['actual_log_return'] = block['actual']
                block['origin_price'] = anchor.iloc[pos].to_numpy()
                for j, tau in enumerate(TAUS):
                    block[f'q_{tau:.3f}'] = q[:, j]
                quantiles.append(block)
    forecasts, q = pd.concat(points, ignore_index=True), pd.concat(quantiles, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(OUT / 'residual_forecasts.csv', index=False, date_format='%Y-%m-%d', lineterminator='\n')
    q.to_csv(OUT / 'residual_quantiles.csv', index=False, date_format='%Y-%m-%d', lineterminator='\n')
    save_selection(OUT / 'residual_params.json', selected)
    print(f'wrote {len(forecasts)} point and {len(q)} quantile rows', flush=True)
    return forecasts, q


if __name__ == '__main__':
    run()

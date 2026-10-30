"""Study 2: probabilities of large h-day moves (ml/risk/PLAN.md).

  uv run python -m m3_ml.risk.events validate
  uv run python -m m3_ml.risk.events test

Events: down = y_{t,h} < -c_h, up = y_{t,h} > c_h, with c_h fixed in advance.
Probabilities come from (a) the distribution forecasts (CDF by linear
interpolation of their quantiles), (b) direct classifiers trained here, and
(c) climatology, expanding during validation and frozen before October.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from m3_ml.common.compute import bounded_cpu, lightgbm_params, report_compute
from m3_ml.common.data import HORIZONS
from m3_ml.common.protocol import FIT_END, MODES, PROTOCOL_ID, VALIDATION_END, VALIDATION_START
from m3_ml.common.walkforward import final_training_positions, forecast_origins as final_origins
from m3_ml.common.preprocessing import make_preprocessor
from m3_ml.distribution import run as dist
from m3_ml.common.execution import ValidationCache, input_frames, run_jobs
from m3_ml.distribution.quantiles import QUANTILE_COLUMNS, TAUS
from m3_ml.distribution.tails import ALL_COLUMNS, ALL_TAUS

from .volatility import Context
from .protocol import frame, load_selection, path_horizons, save_selection, selection_horizon

OUT = Path(__file__).resolve().parent / 'outputs' / 'frozen_october_2026'
THRESHOLDS = {1: 0.03, 5: 0.07, 20: 0.15, 60: 0.25}
SIDES = ('down', 'up')
EPS = 1e-4
SEED = 0


def threshold(h):
    return float(np.interp(h, list(THRESHOLDS), list(THRESHOLDS.values())))


def labels(inp, h, side):
    y = inp.y[h]
    event = (y < -threshold(h)) if side == 'down' else (y > threshold(h))
    return event.astype(float).where(y.notna())


def derived_probability(Q, taus, h, side):
    """P(event) from quantile rows by linear interpolation of the CDF; beyond the
    outermost level the tail gets half of that level's probability."""
    lo, hi = taus[0] / 2, 1 - taus[0] / 2
    x = -threshold(h) if side == 'down' else threshold(h)
    Q = np.sort(np.asarray(Q, dtype=float), axis=1)   # main and tail grids are each sorted; sort jointly
    cdf = np.array([np.interp(x, q, taus, left=lo, right=hi) for q in Q])
    return cdf if side == 'down' else 1 - cdf



def fit_predict(make_model, X, y, train, origins):
    yt = y.iloc[train]
    if yt.nunique() < 2:                        # no event (or only events) in the window
        return np.full(len(origins), np.clip(yt.mean(), EPS, 1 - EPS))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = make_model().fit(X.iloc[train], yt.astype(int))
    return np.clip(model.predict_proba(X.iloc[origins])[:, 1], EPS, 1 - EPS)


def predict_climatology(ctx, y, h, train, origins):
    return np.full(len(origins), np.clip(y.iloc[train].mean(), EPS, 1 - EPS))


def predict_logistic(ctx, y, h, train, origins, C):
    make = lambda: make_pipeline(make_preprocessor('scaled'), LogisticRegression(C=C, max_iter=2000))
    return fit_predict(make, ctx.matrix('cc_gk_implied'), y, train, origins)


def predict_lightgbm(ctx, y, h, train, origins, num_leaves, n_estimators):
    make = lambda: LGBMClassifier(num_leaves=num_leaves, n_estimators=n_estimators, learning_rate=0.02,
                                  min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  random_state=SEED, verbose=-1, **lightgbm_params())
    return fit_predict(make, ctx.matrix('har_base'), y, train, origins)


MODELS = {
    'climatology': (predict_climatology, [{}]),
    'logistic': (predict_logistic, [{'C': c} for c in (0.01, 0.1, 1, 10)]),
    'lightgbm_clf': (predict_lightgbm, [{'num_leaves': l, 'n_estimators': n} for l in (4, 8) for n in (100, 300)]),
}


def walk_forward(ctx, method, params, h, side, origins, refit_every):
    if ctx.inp.dates.iloc[list(origins)].gt(pd.Timestamp(FIT_END)).any():
        raise ValueError('walk_forward is historical validation only; use frozen test() for October')
    predict = MODELS[method][0]
    y = labels(ctx.inp, h, side)
    ok = y.notna().to_numpy()
    if method != 'climatology':
        X = ctx.matrix('cc_gk_implied' if method == 'logistic' else 'har_base')
        ok = ok & X.notna().all(axis=1).to_numpy()
    out = []
    for i in range(0, len(origins), refit_every):
        print(f'events method={method} h={h} side={side} '
              f'refit={i // refit_every + 1}/{(len(origins) + refit_every - 1) // refit_every}', flush=True)
        block = origins[i:i + refit_every]
        refit = int(block[0])
        train = final_training_positions(ctx.inp.dates, ok, y.notna(), h,
                                         cutoff=min(ctx.inp.dates.iloc[refit], pd.Timestamp(FIT_END)))
        out.append(predict(ctx, y, h, train, block, **params))
    return np.concatenate(out)


def brier(p, y):
    return (np.asarray(p) - np.asarray(y)) ** 2


def fold_score(ctx, method, params, h, side, year):
    candidate = dist.fold_origins(ctx.inp, h, year)
    origins = candidate[ctx.inp.dates.iloc[candidate].between(VALIDATION_START, VALIDATION_END).to_numpy()]
    if not len(origins):
        return np.nan
    p = walk_forward(ctx, method, params, h, side, origins, dist.REFIT_VALIDATION)
    return float(np.nanmean(brier(p, labels(ctx.inp, h, side).iloc[origins])))


def _validation_job(ctx, task):
    method, candidate, params, h, side, year = task
    return fold_score(ctx, method, params, h, side, year)


@bounded_cpu
def validate():
    report_compute()
    ctx = Context()
    tasks = [(m, i, p, h, s, y) for m, (_, g) in MODELS.items() for i, p in enumerate(g)
             for h in HORIZONS for s in SIDES for y in dist.FOLD_YEARS]
    t0 = time.time()
    cache = ValidationCache('events_scores', {**input_frames(ctx.inp), 'vol': ctx.vol})
    scores = run_jobs(_validation_job, tasks, ctx, 'events validation', cache)
    frame = pd.DataFrame([{'method': m, 'grid': i, 'params': json.dumps(p), 'h': h, 'side': s, 'year': y, 'brier': b}
                          for (m, i, p, h, s, y), b in zip(tasks, scores)])
    # Distribution-derived probabilities on the same folds (validation forecasts, main grid).
    v = pd.read_csv(dist.OUT / 'validation_selected_quantiles.csv', parse_dates=['origin_date'])
    pos = pd.Series(np.arange(len(ctx.inp.dates)), index=pd.DatetimeIndex(ctx.inp.dates))
    rows = []
    for (method, h), g in v.groupby(['method', 'horizon']):
        origins = pos.reindex(g['origin_date']).to_numpy()
        for side in SIDES:
            y = labels(ctx.inp, h, side).iloc[origins].to_numpy()
            b = brier(derived_probability(g[QUANTILE_COLUMNS].to_numpy(), TAUS, h, side), y)
            for year, idx in pd.Series(g['origin_date'].dt.year.to_numpy()).groupby(g['origin_date'].dt.year.to_numpy()).groups.items():
                rows.append({'method': f'dist_{method}', 'grid': 0, 'params': '{}', 'h': h, 'side': side,
                             'year': year, 'brier': float(np.nanmean(b[list(idx)]))})
    frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / 'events_validation_scores.csv', index=False, lineterminator='\n')
    summary = frame.groupby(['method', 'grid', 'params', 'h', 'side'])['brier'].mean().rename('score').reset_index()
    trained = summary[summary.method.isin(MODELS)]
    best = trained.loc[trained.groupby(['method', 'h', 'side'])['score'].idxmin()]
    selected = {f'{r.method}/h{r.h}/{r.side}': {'params': json.loads(r.params), 'score': round(r.score, 6)}
                for r in best.itertuples()}
    save_selection(OUT / 'events_selected.json', selected)
    clim = summary[summary.method == 'climatology'].set_index(['h', 'side'])['score']
    summary['skill'] = 1 - summary['score'] / clim.reindex(pd.MultiIndex.from_frame(summary[['h', 'side']])).to_numpy()
    best_all = summary.loc[summary.groupby(['method', 'h', 'side'])['score'].idxmin()]
    print(best_all.pivot_table(index='method', columns=['side', 'h'], values='skill').round(3).to_string())
    print(f'{len(tasks)} fold runs in {time.time() - t0:.0f}s')


@bounded_cpu
def test():
    report_compute()
    ctx = Context()
    selected = load_selection(OUT / 'events_selected.json')
    rows = []
    for h in path_horizons(ctx.inp.dates, 'fixed_origin'):
        positions = final_origins(ctx.inp.dates, 'daily_update' if h in HORIZONS else 'fixed_origin')
        for method, (predict, _) in MODELS.items():
            for side in SIDES:
                print(f'events final method={method} h={h} side={side}', flush=True)
                y = labels(ctx.inp, h, side)
                ok = y.notna()
                if method != 'climatology':
                    X = ctx.matrix('cc_gk_implied' if method == 'logistic' else 'har_base')
                    ok &= X.notna().all(axis=1)
                train = final_training_positions(ctx.inp.dates, ok, y.notna(), h)
                params = selected[f'{method}/h{selection_horizon(h)}/{side}']['params']
                probability = predict(ctx, y, h, train, positions, **params)
                for mode in MODES:
                    if mode == 'daily_update' and h not in HORIZONS:
                        continue
                    take = np.arange(len(positions)) if mode == 'daily_update' else np.array([0])
                    block = frame(ctx.inp.dates, positions[take], h, mode, method, probability[take], y.iloc[positions[take]])
                    block['side'], block['threshold'] = side, threshold(h)
                    block['prob'], block['event'] = block['forecast'], block['actual']
                    rows.append(block)
    t = pd.read_csv(dist.OUT / 'test_quantiles.csv', parse_dates=['origin_date', 'target_date'])
    if not {'mode', 'protocol_id', 'target_date', 'target_row'}.issubset(t.columns):
        raise ValueError('Regenerate distribution final forecasts before deriving event probabilities')
    if not t['protocol_id'].eq(PROTOCOL_ID).all():
        raise ValueError('Distribution forecasts use a historical protocol')
    for (mode, method, h), g in t.groupby(['mode', 'method', 'horizon']):
        for side in SIDES:
            actual = g['actual_log_return']
            event = ((actual < -threshold(h)) if side == 'down' else (actual > threshold(h))).astype(float).where(actual.notna())
            block = g[['protocol_id', 'mode', 'origin_date', 'target_date', 'target_row', 'horizon']].copy()
            block['method'], block['side'], block['threshold'] = f'dist_{method}', side, threshold(h)
            block['forecast'] = derived_probability(g[ALL_COLUMNS].to_numpy(), ALL_TAUS, h, side)
            block['actual'] = event.to_numpy()
            block['prob'], block['event'] = block['forecast'], block['actual']
            rows.append(block)
    out = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / 'events_test_probabilities.csv', index=False, date_format='%Y-%m-%d', lineterminator='\n')
    print('wrote', out.shape)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['validate', 'test'])
    validate() if parser.parse_args().stage == 'validate' else test()

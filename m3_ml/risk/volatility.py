"""Study 1: forecasts of the h-day realized variance (ml/risk/PLAN.md).

  uv run python -m m3_ml.risk.volatility validate
  uv run python -m m3_ml.risk.volatility test

Historical validation uses publication-aware expanding fits. Final coefficients
freeze by Sep30; both modes forecast October targets with separate scoring.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch.univariate import GARCH, StudentsT, ZeroMean
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline

from m3_ml.common.compute import bounded_cpu, lightgbm_params, report_compute
from m3_ml.common.execution import ValidationCache, input_frames, run_jobs
from m3_ml.common.data import HORIZONS, label_available_dates, targets as return_targets
from m3_ml.common.protocol import FIT_END, MODES, VALIDATION_END, VALIDATION_START
from m3_ml.common.walkforward import final_training_positions, forecast_origins as final_origins
from m3_ml.common.preprocessing import make_preprocessor
from m3_ml.distribution import run as dist
from m3_ml.distribution.benchmarks import ewma_sigma
from m3_ml.distribution.data import VOL_COLUMNS, load_inputs
from m3_ml.distribution.scale import purged_folds

from .features import FEATURE_SETS, FLOOR, realized_variance, volatility_features
from .protocol import (available_return_mask, frame, load_selection,
                       path_horizons, save_selection, selection_horizon)

OUT = Path(__file__).resolve().parent / 'outputs' / 'frozen_october_2026'
SEED = 0


class Context:
    def __init__(self, table=None, futures=None):
        self.inp = load_inputs(table)
        self.inp.y = return_targets(self.inp.table, horizons=tuple(range(1, 61)))
        self.vol = volatility_features(self.inp, futures)
        self.base = self.inp.features['base_vol'].drop(columns=VOL_COLUMNS)
        self.rv = pd.DataFrame({h: realized_variance(self.inp, h) for h in range(1, 61)})
        self.log_rv = np.log(self.rv + FLOOR)
        self._ewma = {}
        self.available = pd.to_datetime(label_available_dates(self.inp.table))

    def state(self, origin):
        mask = available_return_mask(self.inp.table, self.inp.dates.iloc[int(origin)])
        return self.inp.returns.where(mask)

    def last_observed(self, origin):
        mask = np.logical_and.accumulate(self.available.le(self.inp.dates.iloc[int(origin)]).to_numpy())
        positions = np.flatnonzero(mask)
        if not len(positions):
            raise ValueError('No policy-available WTI history at this origin')
        return int(positions[-1])

    def matrix(self, features):
        if features == 'har_base':
            return pd.concat([self.vol[FEATURE_SETS['cc_gk_implied']], self.base], axis=1)
        return self.vol[FEATURE_SETS[features]]

    def ewma(self, lam):
        if lam not in self._ewma:
            values = []
            for origin in range(len(self.inp.dates)):
                if origin == 0 or (origin + 1) % 500 == 0 or origin + 1 == len(self.inp.dates):
                    print(f'volatility EWMA lam={lam} origin={origin + 1}/{len(self.inp.dates)}', flush=True)
                last = self.last_observed(origin) if available_return_mask(self.inp.table, self.inp.dates.iloc[origin]).any() else -1
                r = self.inp.returns.iloc[:last + 1]
                values.append(float(ewma_sigma(r, lam).iloc[-1]) if len(r) else np.nan)
            self._ewma[lam] = pd.Series(values, index=self.inp.dates.index)
        return self._ewma[lam]


def qlike(rv, s2):
    ratio = np.asarray(rv, dtype=float) / np.asarray(s2, dtype=float)
    return ratio - np.log(ratio) - 1



def predict_hist(ctx, h, refit, train, origins, k):
    return np.array([h * ctx.state(p).dropna().iloc[-k:].pow(2).mean() for p in origins])


def predict_ewma(ctx, h, refit, train, origins, lam):
    return h * ctx.ewma(lam).iloc[origins].to_numpy() ** 2


def predict_garch(ctx, h, refit, train, origins, vol):
    # Fit once on policy-available quote history at the refit date. Later releases
    # only filter the same coefficient vector; gaps retain their quoted-row age.
    last_fit = ctx.last_observed(refit)
    rfit = (100 * ctx.state(refit)).iloc[1:last_fit + 1].fillna(0).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fitted = ZeroMean(rfit, volatility=GARCH(p=1, o=int(vol == 'gjr'), q=1),
                          distribution=StudentsT()).fit(disp='off', show_warning=False)
    params = np.asarray(fitted.params)
    omega, alpha = params[0], params[1]
    gamma = params[2] if vol == 'gjr' else 0.0
    beta = params[3] if vol == 'gjr' else params[2]
    persistence = alpha + beta + gamma / 2
    out = []
    for origin in origins:
        state = (100 * ctx.state(origin)).to_numpy()
        variance = float(np.var(rfit[np.isfinite(rfit)]))
        for shock in state[1:ctx.last_observed(origin) + 1]:
            if np.isfinite(shock):
                variance = omega + (alpha + gamma * (shock < 0)) * shock ** 2 + beta * variance
            else:
                variance = omega + persistence * variance
        gap = int(origin) - ctx.last_observed(origin)
        total = 0.0
        for step in range(1, gap + h + 1):
            if step > gap:
                total += variance / 1e4
            variance = omega + persistence * variance
        out.append(total)
    return np.asarray(out)


def predict_ovx_raw(ctx, h, refit, train, origins):
    return h * (ctx.inp.table['oil_implied_volatility'].iloc[origins].to_numpy() / 100) ** 2 / 252


def log_model(make_model, X, ctx, h, train, origins):
    """Fit log RV, then rescale exp(prediction) by the Duan smearing factor of
    purged, embargoed cross-fitted residuals (mean of exp(residual))."""
    y = ctx.log_rv[h]
    train = np.sort(train)
    resid = []
    for fit, block in purged_folds(train, h):
        m = make_model().fit(X.iloc[fit], y.iloc[fit])
        resid.append(y.iloc[block].to_numpy() - m.predict(X.iloc[block]))
    smear = float(np.mean(np.exp(np.concatenate(resid))))
    model = make_model().fit(X.iloc[train], y.iloc[train])
    return np.exp(model.predict(X.iloc[origins])) * smear


def predict_ovx_cal(ctx, h, refit, train, origins):
    return log_model(LinearRegression, ctx.vol[['ovx2']], ctx, h, train, origins)


def predict_har(ctx, h, refit, train, origins, alpha, features):
    make = lambda: make_pipeline(make_preprocessor('scaled'), Ridge(alpha=alpha))
    return log_model(make, ctx.matrix(features), ctx, h, train, origins)


def predict_lightgbm(ctx, h, refit, train, origins, num_leaves, n_estimators, features):
    make = lambda: LGBMRegressor(num_leaves=num_leaves, n_estimators=n_estimators, learning_rate=0.02,
                                 min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                 random_state=SEED, verbose=-1, **lightgbm_params())
    return log_model(make, ctx.matrix(features), ctx, h, train, origins)


METHODS = {
    'hist': (predict_hist, [{'k': k} for k in (20, 60, 250)]),
    'ewma': (predict_ewma, [{'lam': l} for l in (0.94, 0.97, 0.99)]),
    'garch': (predict_garch, [{'vol': v} for v in ('garch', 'gjr')]),
    'ovx_raw': (predict_ovx_raw, [{}]),
    'ovx_cal': (predict_ovx_cal, [{}]),
    'har': (predict_har, [{'alpha': a, 'features': f} for a in (0.1, 1, 10, 100)
                          for f in ('cc', 'cc_gk', 'cc_gk_implied')]),
    'lightgbm': (predict_lightgbm, [{'num_leaves': l, 'n_estimators': n, 'features': f}
                                    for l in (4, 8) for n in (200, 500) for f in ('cc_gk_implied', 'har_base')]),
}
NEEDS_FEATURES = {'ovx_cal', 'har', 'lightgbm'}


def usable(ctx, method, params, h):
    ok = ctx.rv[h].notna()
    if method in NEEDS_FEATURES:
        X = ctx.vol[['ovx2']] if method == 'ovx_cal' else ctx.matrix(params['features'])
        ok &= X.notna().all(axis=1)
    return ok.to_numpy()


def walk_forward(ctx, method, params, h, origins, refit_every):
    if ctx.inp.dates.iloc[list(origins)].gt(pd.Timestamp(FIT_END)).any():
        raise ValueError('walk_forward is historical validation only; use frozen test() for October')
    predict = METHODS[method][0]
    ok = usable(ctx, method, params, h)
    out = []
    for i in range(0, len(origins), refit_every):
        block = origins[i:i + refit_every]
        refit = int(block[0])
        train = final_training_positions(ctx.inp.dates, ok, ctx.rv[h].notna(), h,
                                         cutoff=min(ctx.inp.dates.iloc[refit], pd.Timestamp(FIT_END)))
        out.append(predict(ctx, h, refit, train, block, **params))
    return np.concatenate(out)


def fold_score(ctx, method, params, h, year):
    candidates = dist.fold_origins(ctx.inp, h, year)
    origins = candidates[ctx.inp.dates.iloc[candidates].between(VALIDATION_START, VALIDATION_END).to_numpy()]
    if not len(origins):
        return np.nan
    s2 = walk_forward(ctx, method, params, h, origins, dist.REFIT_VALIDATION)
    return float(np.nanmean(qlike(ctx.rv[h].iloc[origins], s2)))


def _validation_job(ctx, task):
    method, candidate, params, h, year = task
    return fold_score(ctx, method, params, h, year)


@bounded_cpu
def validate():
    report_compute()
    ctx = Context()
    tasks = [(m, i, p, h, y) for m, (_, g) in METHODS.items() for i, p in enumerate(g)
             for h in HORIZONS for y in dist.FOLD_YEARS]
    t0 = time.time()
    cache = ValidationCache('volatility_scores', {**input_frames(ctx.inp), 'vol': ctx.vol, 'rv': ctx.rv})
    scores = run_jobs(_validation_job, tasks, ctx, 'volatility validation', cache)
    frame = pd.DataFrame([{'method': m, 'grid': i, 'params': json.dumps(p), 'h': h, 'year': y, 'qlike': s}
                          for (m, i, p, h, y), s in zip(tasks, scores)])
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / 'volatility_validation_scores.csv', index=False, lineterminator='\n')
    summary = frame.groupby(['method', 'grid', 'params', 'h'])['qlike'].mean().rename('score').reset_index()
    best = summary.loc[summary.groupby(['method', 'h'])['score'].idxmin()]
    selected = {f'{r.method}/h{r.h}': {'params': json.loads(r.params), 'score': round(r.score, 5)}
                for r in best.itertuples()}
    save_selection(OUT / 'volatility_selected.json', selected)
    print(best.pivot(index='method', columns='h', values='score').round(4).to_string())
    print(f'{len(tasks)} fold runs in {time.time() - t0:.0f}s')


@bounded_cpu
def test():
    report_compute()
    ctx = Context()
    selected = load_selection(OUT / 'volatility_selected.json')
    refit = int(final_origins(ctx.inp.dates, 'fixed_origin')[0])
    rows = []
    for h in path_horizons(ctx.inp.dates, 'fixed_origin'):
        positions = final_origins(ctx.inp.dates, 'daily_update' if h in HORIZONS else 'fixed_origin')
        for method, (predict, _) in METHODS.items():
            print(f'volatility final method={method} h={h}', flush=True)
            params = selected[f'{method}/h{selection_horizon(h)}']['params']
            train = final_training_positions(ctx.inp.dates, usable(ctx, method, params, h),
                                             ctx.rv[h].notna(), h)
            forecast = predict(ctx, h, refit, train, positions, **params)
            for mode in MODES:
                if mode == 'daily_update' and h not in HORIZONS:
                    continue
                take = np.arange(len(positions)) if mode == 'daily_update' else np.array([0])
                block = frame(ctx.inp.dates, positions[take], h, mode, method, forecast[take],
                              ctx.rv[h].iloc[positions[take]])
                block['sigma2'], block['rv'] = block['forecast'], block['actual']
                rows.append(block)
    out = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / 'volatility_test_forecasts.csv', index=False, date_format='%Y-%m-%d', lineterminator='\n')
    print('wrote', out.shape)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['validate', 'test'])
    validate() if parser.parse_args().stage == 'validate' else test()

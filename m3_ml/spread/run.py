"""Study 4: forecasts of oil price spread changes (ml/spread/PLAN.md).

  uv run python -m m3_ml.spread.run validate [--spread brent_wti|crack_321]
  uv run python -m m3_ml.spread.run test [--spread ...]

Positions are rows of the spread series (days on which every leg has a quote).
The target is the h-row spread change; features use the spread up to row t
and weekly EIA features published before that date.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline

from m3_ml.common.compute import bounded_cpu, lightgbm_params, report_compute
from m3_ml.common.data import HORIZONS, label_available_dates, load_table
from m3_ml.common.protocol import FIT_END, MODES, PROTOCOL_ID, VALIDATION_END, VALIDATION_START
from m3_ml.common.walkforward import final_training_positions, forecast_origins as final_origins
from m3_ml.risk.protocol import frame, load_selection, path_horizons, save_selection, selection_horizon
from m3_ml.common.features import build_features
from m3_ml.common.preprocessing import make_preprocessor
from m3_ml.distribution.run import FOLD_YEARS, REFIT_VALIDATION
from m3_ml.tabular.models import RelativeElasticNet

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'outputs' / 'frozen_october_2026'
START = '2012-08-23'
SEED = 0
EIA_COLUMNS = [f'{v}_{k}' for v in ('crude_exports', 'crude_imports', 'crude_net_imports', 'crude_production')
               for k in ('wk_chg', 'yoy_chg')]
GALLONS = 42


def fred(name):
    f = pd.read_csv(ROOT / 'data/raw/fred' / name, na_values='.')
    return f.set_index(pd.to_datetime(f.iloc[:, 0])).iloc[:, 1].astype(float).dropna()


def eia_spot(name):
    """NY Harbor product spot prices fetched from the EIA API (ml/spread/fetch_products.py)."""
    f = pd.read_csv(ROOT / 'data/raw/eia_api/spot_prices' / name, parse_dates=['observation_date'])
    return f.set_index('observation_date')['value'].astype(float).dropna()


def spread_series(kind):
    wti = fred('wti_spot_daily.csv')
    wti = wti[wti > 0]                                   # 2020-04-20 negative price excluded
    if kind == 'brent_wti':
        legs = pd.concat([fred('brent_spot_daily.csv').rename('brent'), wti.rename('wti')], axis=1).dropna()
        s = legs['brent'] - legs['wti']
    else:
        gas = eia_spot('ny_conventional_gasoline_spot_daily.csv')
        ho = eia_spot('ny_heating_oil_spot_daily.csv')
        legs = pd.concat([gas.rename('gas'), ho.rename('ho'), wti.rename('wti')], axis=1, sort=True).dropna()
        s = (2 * GALLONS * legs['gas'] + GALLONS * legs['ho']) / 3 - legs['wti']
    return s[s.index >= START].rename('spread')


class Data:
    def __init__(self, kind, end=None):
        s = spread_series(kind)
        if end is not None:
            s = s[s.index <= pd.Timestamp(end)]
        self.raw_s = s.reset_index(drop=True)
        self.dates = pd.Series(s.index)
        available = pd.to_datetime(label_available_dates(self.dates))
        self.available_positions = []
        values = []
        for date in self.dates:
            if len(values) == 0 or (len(values) + 1) % 500 == 0 or len(values) + 1 == len(self.dates):
                print(f'spread={kind} inputs origin={len(values) + 1}/{len(self.dates)}', flush=True)
            known = np.flatnonzero(np.logical_and.accumulate(available.le(date).to_numpy()))
            p = int(known[-1]) if len(known) else -1
            self.available_positions.append(p)
            published = np.flatnonzero(available.le(date).to_numpy())
            newest = int(published[-1]) if len(published) else -1
            values.append(self.raw_s.iloc[newest] if newest >= 0 else np.nan)
        self.s = pd.Series(values)
        d = self.s.diff()
        feats = {f'dev_{k}': self.s - self.s.rolling(k).mean() for k in (20, 60, 120, 250)}
        feats.update({f'chg_{k}': self.s - self.s.shift(k) for k in (1, 5, 20)})
        feats['vol_20'] = d.rolling(20).std()
        spread_only = pd.DataFrame(feats)
        table = load_table()
        if end is not None:
            table = table[table['date'] <= pd.Timestamp(end)]
        eia = build_features(table)[EIA_COLUMNS].assign(date=table['date'].to_numpy())
        # wti_daily row d holds data published before 00:00 on d: as of the spread date.
        eia = pd.merge_asof(self.dates.to_frame('date'), eia.sort_values('date'), on='date').drop(columns='date')
        self.features = {'spread': spread_only, 'spread_eia': pd.concat([spread_only, eia], axis=1)}
        self.y = pd.DataFrame({h: self.raw_s.shift(-h) - self.s for h in range(1, 61)})



def predict_rw(data, h, refit, train, origins):
    return np.zeros(len(origins))


def predict_drift(data, h, refit, train, origins):
    return np.full(len(origins), data.y[h].iloc[train].mean())


def predict_ou(data, h, refit, train, origins, window):
    """AR(1) on the spread level using rows up to the refit; mean-reverting forecast."""
    last = data.available_positions[int(refit)]
    level = data.raw_s.iloc[:last + 1]
    if window != 'expanding':
        level = level.iloc[-int(window):]
    x, z = level.iloc[:-1].to_numpy(), level.iloc[1:].to_numpy()
    phi, c = np.polyfit(x, z, 1)
    if not 0 < phi < 1:
        return np.zeros(len(origins))
    mu = c / (1 - phi)
    s = data.s.iloc[origins].to_numpy()
    gap = np.asarray(origins, dtype=int) - np.asarray(data.available_positions)[origins]
    state = data.raw_s.iloc[np.asarray(data.available_positions)[origins]].to_numpy()
    return (mu + phi ** (h + gap) * (state - mu)) - s


def predict_ma_dev(data, h, refit, train, origins, k):
    x = data.features['spread'][f'dev_{k}']
    b, a = np.polyfit(x.iloc[train].to_numpy(), data.y[h].iloc[train].to_numpy(), 1)
    return a + b * x.iloc[origins].to_numpy()


def predict_enet(data, h, refit, train, origins, alpha_ratio, l1_ratio, features):
    X = data.features[features]
    model = make_pipeline(make_preprocessor('scaled'), RelativeElasticNet(alpha_ratio=alpha_ratio, l1_ratio=l1_ratio))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return model.fit(X.iloc[train], data.y[h].iloc[train]).predict(X.iloc[origins])


def predict_lgbm(data, h, refit, train, origins, num_leaves, n_estimators, features):
    X = data.features[features]
    model = LGBMRegressor(num_leaves=num_leaves, n_estimators=n_estimators, learning_rate=0.02, min_child_samples=50,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                          verbose=-1, **lightgbm_params())
    return model.fit(X.iloc[train], data.y[h].iloc[train]).predict(X.iloc[origins])


METHODS = {
    'rw': (predict_rw, [{}]),
    'drift': (predict_drift, [{}]),
    'ou': (predict_ou, [{'window': w} for w in ('expanding', 1500, 750)]),
    'ma_dev': (predict_ma_dev, [{'k': k} for k in (20, 60, 120, 250)]),
    'enet': (predict_enet, [{'alpha_ratio': a, 'l1_ratio': l, 'features': f} for a in (1.0, 0.3, 0.1, 0.03, 0.01)
                            for l in (0.1, 0.5, 0.9) for f in ('spread', 'spread_eia')]),
    'lgbm': (predict_lgbm, [{'num_leaves': n, 'n_estimators': t, 'features': f} for n in (4, 8) for t in (100, 300)
                            for f in ('spread', 'spread_eia')]),
}


def feature_set(method, params):
    if method in ('enet', 'lgbm'):
        return params['features']
    return 'spread' if method == 'ma_dev' else None


def walk_forward(data, method, params, h, origins, refit_every):
    if data.dates.iloc[list(origins)].gt(pd.Timestamp(FIT_END)).any():
        raise ValueError('walk_forward is historical validation only; use frozen test() for October')
    predict = METHODS[method][0]
    ok = data.y[h].notna().to_numpy()
    fs = feature_set(method, params)
    if fs:
        ok = ok & data.features[fs].notna().all(axis=1).to_numpy()
    out = []
    for i in range(0, len(origins), refit_every):
        block = origins[i:i + refit_every]
        refit = int(block[0])
        train = final_training_positions(data.dates, ok, data.y[h].notna(), h,
                                         cutoff=min(data.dates.iloc[refit], pd.Timestamp(FIT_END)))
        out.append(predict(data, h, refit, train, block, **params))
    return np.concatenate(out)


def complete(data):
    return data.features['spread_eia'].notna().all(axis=1).to_numpy()


def fold_origins(data, h, year):
    positions = final_training_positions(data.dates, complete(data), data.y[h].notna(), h, VALIDATION_END)
    dates = data.dates.iloc[positions]
    return positions[(dates.dt.year.eq(year) & dates.between(VALIDATION_START, VALIDATION_END)).to_numpy()]




def fold_rmse(data, method, params, h, year):
    origins = fold_origins(data, h, year)
    if not len(origins):
        return np.nan
    err = walk_forward(data, method, params, h, origins, REFIT_VALIDATION) - data.y[h].iloc[origins].to_numpy()
    return float(np.sqrt(np.nanmean(err ** 2)))


@bounded_cpu
def validate(kind):
    report_compute()
    data = Data(kind)
    tasks = [(m, i, p, h, y) for m, (_, g) in METHODS.items() for i, p in enumerate(g)
             for h in HORIZONS for y in FOLD_YEARS]
    t0 = time.time()
    scores = []
    for index, (m, candidate, p, h, year) in enumerate(tasks, 1):
        print(f'spread={kind} validate job={index}/{len(tasks)} method={m} h={h} '
              f'fold={year} candidate={candidate + 1}/{len(METHODS[m][1])}', flush=True)
        scores.append(fold_rmse(data, m, p, h, year))
    frame = pd.DataFrame([{'method': m, 'grid': i, 'params': json.dumps(p), 'h': h, 'year': y, 'rmse': s}
                          for (m, i, p, h, y), s in zip(tasks, scores)])
    rw = frame[frame.method == 'rw'].set_index(['h', 'year'])['rmse']
    frame['ratio'] = frame['rmse'] / rw.reindex(pd.MultiIndex.from_frame(frame[['h', 'year']])).to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / f'{kind}_validation_scores.csv', index=False, lineterminator='\n')
    summary = (frame.groupby(['method', 'grid', 'params', 'h'])['ratio']
               .apply(lambda r: float(np.exp(np.log(r).mean()))).rename('score').reset_index())
    best = summary.loc[summary.groupby(['method', 'h'])['score'].idxmin()]
    selected = {f'{r.method}/h{r.h}': {'params': json.loads(r.params), 'score': round(r.score, 5)}
                for r in best.itertuples()}
    save_selection(OUT / f'{kind}_selected.json', selected)
    print(best.pivot(index='method', columns='h', values='score').round(4).to_string())
    print(f'{len(tasks)} fold runs in {time.time() - t0:.0f}s')


@bounded_cpu
def test(kind):
    report_compute()
    data = Data(kind)
    selected = load_selection(OUT / f'{kind}_selected.json')
    refit = int(final_origins(data.dates, 'fixed_origin')[0])
    rows = []
    for h in path_horizons(data.dates, 'fixed_origin'):
        positions = final_origins(data.dates, 'daily_update' if h in HORIZONS else 'fixed_origin')
        for method, (predict, _) in METHODS.items():
            print(f'spread={kind} final method={method} h={h}', flush=True)
            params = selected[f'{method}/h{selection_horizon(h)}']['params']
            ok = data.y[h].notna()
            fs = feature_set(method, params)
            if fs:
                ok &= data.features[fs].notna().all(axis=1)
            train = final_training_positions(data.dates, ok, data.y[h].notna(), h)
            changes = predict(data, h, refit, train, positions, **params)
            for mode in MODES:
                if mode == 'daily_update' and h not in HORIZONS:
                    continue
                take = np.arange(len(positions)) if mode == 'daily_update' else np.array([0])
                pos, change = positions[take], changes[take]
                anchor = data.s.iloc[pos].to_numpy()
                actual_change = data.y[h].iloc[pos].to_numpy()
                block = frame(data.dates, pos, h, mode, method, anchor + change, anchor + actual_change)
                block['spread'], block['forecast_change'] = anchor, change
                block['actual_change'] = block['actual'] - anchor
                block['release_policy'] = 'all_legs_delayed_to_wti_release_assumption'
                rows.append(block)
    out = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / f'{kind}_test_forecasts.csv', index=False, date_format='%Y-%m-%d', lineterminator='\n')
    print('wrote', out.shape)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['validate', 'test'])
    parser.add_argument('--spread', choices=['brent_wti', 'crack_321'], default='brent_wti')
    args = parser.parse_args()
    validate(args.spread) if args.stage == 'validate' else test(args.spread)

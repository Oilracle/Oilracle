"""Historical validation and one-fit October distribution forecasts.

  uv run python -m m3_ml.distribution.run validate [--methods hs ewma ...]
  uv run python -m m3_ml.distribution.run test [--methods hs ewma ...]

All grids are selected on 2023-09-01..2026-09-30 with publication-aware target
purging. Each final method/horizon invokes its fitting function exactly ONCE
at Sep30 for the union of forecast origins; its calibrated parameters and
preprocessors are shared by fixed_origin and daily_update. The former uses
only the Sep30 row; the latter can filter newly published history.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from m3_ml.common.compute import bounded_cpu, report_compute
from m3_ml.common.data import targets
from m3_ml.common.execution import ValidationCache, input_frames, run_jobs
from m3_ml.common.news import NEWS_POLICY_ID, news_metadata, require_complete_features
from m3_ml.common.protocol import (FIT_END, HORIZONS, MODES, PROTOCOL_ID, TEST_START, TEST_END,
                                   TRAIN_END, VALIDATION_START, VALIDATION_END)
from m3_ml.common.walkforward import (final_training_positions, fixed_path_horizons,
                                      forecast_origins)
from .benchmarks import BENCHMARKS
from .data import load_inputs, volatility_target
from .evt import EVT_MODELS
from .mixture import MIXTURE_MODELS, usable_mask
from .quantile_models import QUANTILE_MODELS
from .quantiles import TAUS, crps, rearrange
from .scale import SCALE_MODELS
from .tails import ALL_COLUMNS, ALL_TAUS, MAIN, TAIL_LOWER, mean_fz0

OUT = Path(__file__).resolve().parent / 'outputs/frozen_october_2026'
METHODS = {**BENCHMARKS, **EVT_MODELS, **SCALE_MODELS, **QUANTILE_MODELS, **MIXTURE_MODELS}
ML_METHODS = set(SCALE_MODELS) | set(QUANTILE_MODELS)
TAIL_SELECTED = {'garch_evt', 'a_evt'}
FOLD_YEARS = range(2023, 2027)
REFIT_VALIDATION = 63
REFERENCE = ('hs', {'window': 'expanding'})
ENSEMBLE = ['a_ridge', 'a_lightgbm', 'b_linear', 'b_lightgbm', 'b_forest']


def manifest():
    return {'protocol_id': PROTOCOL_ID, 'train_end': TRAIN_END, 'validation_start': VALIDATION_START,
            'validation_end': VALIDATION_END, 'fit_end': FIT_END, 'test_start': TEST_START,
            'test_end': TEST_END, 'modes': list(MODES), 'horizons': list(HORIZONS),
            'publication_policy': 'dated_archive_else_assumed_weekly_release',
            'vintage_status': 'retrospective_current_vintage',
            'fit_policy': 'one_fit_per_method_horizon_before_test',
            'news': news_metadata()}


def save_manifest():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'protocol.json').write_text(json.dumps(manifest(), indent=2) + '\n', encoding='utf-8')


def selected_settings():
    path = OUT / 'selected.json'
    if not path.exists():
        return {}
    saved = json.loads(path.read_text(encoding='utf-8'))
    if saved.get('protocol_id') != PROTOCOL_ID or saved.get('validation_end') != VALIDATION_END:
        raise ValueError('selected settings do not match the frozen October protocol')
    if saved.get('news') != news_metadata():
        raise ValueError('news policy/source snapshot changed; archive old distribution outputs '
                         'and rerun complete validation before final fitting')
    return saved['selected']


def resolve(method, params, h, selected=None):
    if method != 'a_evt':
        return params
    selected = selected_settings() if selected is None else selected
    base = min(('a_ridge', 'a_lightgbm'), key=lambda m: selected[f'{m}/h{h}']['score'])
    return {**params, 'base_method': base, 'base_params': selected[f'{base}/h{h}']['params']}


def horizons(method, inp=None):
    all_h = fixed_path_horizons(inp.dates) if inp is not None else HORIZONS
    return tuple(h for h in all_h if method != 'c_mixture' or h != 1)


def ensure_horizon(inp, h):
    if h not in inp.y:
        inp.y[h] = targets(inp.table, [h])[h]
        inp.vol_target[h] = volatility_target(inp.returns, h)


def usable(inp, method, params, h):
    ensure_horizon(inp, h)
    ok = inp.y[h].notna()
    if method in ML_METHODS:
        features = params['base_params']['features'] if method == 'a_evt' else params['features']
        ok &= inp.vol_target[h].notna() & inp.features[features].notna().all(axis=1)
    if method == 'c_mixture':
        ok &= usable_mask(inp, h)
    return ok.to_numpy()


def arrange_all(Q):
    Q = np.array(Q, dtype=float)
    lower = np.isin(ALL_TAUS, TAIL_LOWER)
    upper = ~MAIN & ~lower
    for mask in (MAIN, lower, upper):
        Q[:, mask] = np.sort(Q[:, mask], axis=1)
    return Q


def fitted_block(inp, method, params, h, refit, origins, taus=TAUS, selected=None):
    """One function invocation = one estimator/calibration fit, reused for all rows.

    Internal route-A/C cross-fits are exclusively on the purged training sample.
    No fitting function is invoked separately for an October origin or mode.
    """
    params = resolve(method, params, h, selected)
    ok = usable(inp, method, params, h)
    train = final_training_positions(inp.dates, np.ones(len(ok), dtype=bool), ok, h,
                                     inp.dates.iloc[int(refit)])
    Q = METHODS[method][0](inp, h, int(refit), train, np.asarray(origins), **params, taus=taus)
    return rearrange(Q) if len(taus) == len(TAUS) else arrange_all(Q)


def walk_forward(inp, method, params, h, origins, refit_every, taus=TAUS, extra=None, selected=None):
    """Historical validation only. October refitting is explicitly rejected."""
    origins = np.asarray(origins, dtype=int)
    if len(origins) and inp.dates.iloc[origins].max() > pd.Timestamp(VALIDATION_END):
        raise ValueError('October fitted parameters must stay frozen')
    blocks = []
    count = (len(origins) + refit_every - 1) // refit_every
    for i in range(0, len(origins), refit_every):
        print(f'distribution method={method} h={h} refit={i // refit_every + 1}/{count} '
              f'origin={inp.dates.iloc[origins[i]]:%Y-%m-%d}', flush=True)
        blocks.append(fitted_block(inp, method, {**params, **(extra or {})}, h, int(origins[i]),
                                   origins[i:i + refit_every], taus, selected))
    return np.vstack(blocks) if blocks else np.empty((0, len(taus)))


def fold_origins(inp, h, year):
    ensure_horizon(inp, h)
    mature = final_training_positions(inp.dates, inp.complete, inp.y[h].notna(), h, VALIDATION_END)
    dates = inp.dates.iloc[mature]
    return mature[(dates.between(VALIDATION_START, VALIDATION_END) & dates.dt.year.eq(year)).to_numpy()]


def test_origins(inp):
    return forecast_origins(inp.dates, 'daily_update')


def fold_score(inp, method, params, h, year, metric, selected):
    origins = fold_origins(inp, h, year)
    if not len(origins):
        return np.nan
    taus = TAUS if metric == 'crps' else ALL_TAUS
    Q = walk_forward(inp, method, params, h, origins, REFIT_VALIDATION, taus=taus, selected=selected)
    y = inp.y[h].iloc[origins]
    return float(np.nanmean(crps(Q, y) if metric == 'crps' else mean_fz0(Q, y)))


def _validation_job(context, task):
    inp, selected = context
    m, candidate, params, h, year, metric = task
    print(f'distribution start method={m} h={h} fold={year} candidate={candidate + 1}', flush=True)
    return fold_score(inp, m, params, h, year, metric, selected)


def _selected_forecast_job(context, task):
    inp, selected = context
    m, h = task
    origins = np.concatenate([fold_origins(inp, h, year) for year in FOLD_YEARS])
    return walk_forward(inp, m, selected[f'{m}/h{h}']['params'], h, origins,
                        REFIT_VALIDATION, taus=ALL_TAUS, selected=selected)


def _final_job(context, task):
    inp, selected, fixed, daily = context
    m, h = task
    return fitted_block(inp, m, selected[f'{m}/h{h}']['params'], h, int(fixed[0]),
                         daily if h in HORIZONS else fixed, ALL_TAUS, selected)


@bounded_cpu
def validate(methods):
    """Select every grid using only historical validation and publish its forecasts."""
    report_compute()
    inp = load_inputs()
    selected = selected_settings()
    if 'a_evt' in methods and any(m in methods for m in ('a_ridge', 'a_lightgbm')):
        validate([m for m in methods if m != 'a_evt'])
        validate(['a_evt'])
        return
    metric = {m: 'fz0' if m in TAIL_SELECTED else 'crps' for m in methods}
    tasks = [(m, i, p, h, year, metric[m]) for m in methods for i, p in enumerate(METHODS[m][1])
             for h in horizons(m, inp) for year in FOLD_YEARS]
    refs = [(REFERENCE[0], -1, REFERENCE[1], h, year, k) for k in sorted(set(metric.values()))
            for h in fixed_path_horizons(inp.dates) for year in FOLD_YEARS]
    jobs = tasks + refs
    cache = ValidationCache('distribution_scores', input_frames(inp))
    scores = run_jobs(_validation_job, jobs, (inp, selected), 'distribution validation', cache,
                      cache_key=lambda t: (t, resolve(t[0], t[2], t[3], selected)))
    frame = pd.DataFrame([{'method': m, 'grid': i, 'params': json.dumps(p), 'h': h,
                           'year': year, 'metric': k, 'value': s,
                           'n': len(fold_origins(inp, h, year))}
                          for (m, i, p, h, year, k), s in zip(tasks + refs, scores)])
    ref = frame[frame.grid.eq(-1)].set_index(['metric', 'h', 'year'])['value']
    frame = frame[frame.grid.ge(0)].copy()
    base = ref.reindex(pd.MultiIndex.from_frame(frame[['metric', 'h', 'year']])).to_numpy()
    frame['relative'] = np.where(frame.metric.eq('crps'), frame.value / base, frame.value - base)
    save_manifest()
    path = OUT / 'validation_scores.csv'
    if path.exists():
        old = pd.read_csv(path, float_precision='round_trip')
        if 'news_policy_id' not in old or not old.news_policy_id.eq(NEWS_POLICY_ID).all():
            raise ValueError('cannot merge validation scores from a different news policy')
        frame = pd.concat([old[~old.method.isin(methods)], frame], ignore_index=True)
    frame['protocol_id'] = PROTOCOL_ID
    frame['news_policy_id'] = NEWS_POLICY_ID
    frame.to_csv(path, index=False, lineterminator='\n')
    run = frame[frame.method.isin(methods)].copy()
    is_crps = run.metric.eq('crps')
    run['term'] = np.where(is_crps, np.log(run.relative.where(is_crps, 1.0)), run.relative)
    summary = run.groupby(['method', 'grid', 'params', 'h', 'metric']).term.mean().rename('score').reset_index()
    summary.loc[summary.metric.eq('crps'), 'score'] = np.exp(summary.loc[summary.metric.eq('crps'), 'score'])
    if summary.score.isna().any():
        raise ValueError('a validation grid has no finite mature-target score')
    best = summary.loc[summary.groupby(['method', 'h']).score.idxmin()]
    new = {f'{r.method}/h{r.h}': {'params': json.loads(r.params), 'score': float(r.score),
                                'metric': r.metric} for r in best.itertuples()}
    selected.update(new)
    (OUT / 'selected.json').write_text(json.dumps({**manifest(), 'selected': selected}, indent=2,
                                                 sort_keys=True) + '\n', encoding='utf-8')
    # These genuinely out-of-origin forecasts are also the calibration source
    # for the event-probability study; no in-sample/final-test calibration.
    selected_frames = []
    selected_jobs = [(m, h) for m in methods for h in horizons(m, inp)]
    cache = ValidationCache('distribution_selected_forecasts', input_frames(inp))
    predictions = run_jobs(_selected_forecast_job, selected_jobs, (inp, selected),
                           'distribution selected forecasts', cache,
                           cache_key=lambda t: (t, resolve(t[0], selected[f'{t[0]}/h{t[1]}']['params'], t[1], selected)))
    for (m, h), Q in zip(selected_jobs, predictions):
        origins = np.concatenate([fold_origins(inp, h, year) for year in FOLD_YEARS])
        selected_frames.append(frame_forecasts(inp, m, h, origins, Q, 'validation'))
    selected_path = OUT / 'validation_selected_quantiles.csv'
    if selected_path.exists():
        old = pd.read_csv(selected_path, parse_dates=['origin_date', 'target_date'])
        if 'news_policy_id' not in old or not old.news_policy_id.eq(NEWS_POLICY_ID).all():
            raise ValueError('cannot merge validation forecasts from a different news policy')
        selected_frames.insert(0, old[~old.method.isin(methods)])
    pd.concat(selected_frames, ignore_index=True).to_csv(selected_path, index=False,
                                                        date_format='%Y-%m-%d', lineterminator='\n')
    print(summary.sort_values(['method', 'h', 'score']).to_string(index=False))


def frame_forecasts(inp, method, h, origins, Q, mode):
    ensure_horizon(inp, h)
    origins = np.asarray(origins, dtype=int)
    out = pd.DataFrame(Q, columns=ALL_COLUMNS)
    out.insert(0, 'mode', mode)
    out['origin_date'] = inp.dates.iloc[origins].to_numpy()
    out['target_row'] = origins + h
    out['target_date'] = inp.dates.reindex(origins + h).to_numpy()
    out['horizon'], out['method'] = h, method
    out['origin_price'] = inp.anchor.iloc[origins].to_numpy()
    out['actual_log_return'] = inp.y[h].iloc[origins].to_numpy()
    if mode != 'validation':
        out.loc[~out.target_date.between(TEST_START, TEST_END), 'actual_log_return'] = np.nan
    out['forecast'] = out.origin_price * np.exp(out['q_0.500'])
    out['actual'] = out.origin_price * np.exp(out.actual_log_return)
    out['status'] = np.where(out.actual.notna(), 'realized', 'unreached')
    for name, value in {'protocol_id': PROTOCOL_ID, 'fit_end': FIT_END, 'train_end': TRAIN_END,
                        'validation_start': VALIDATION_START, 'validation_end': VALIDATION_END,
                        'publication_policy': 'dated_archive_else_assumed_weekly_release'}.items():
        out[name] = value
    out['news_policy_id'] = NEWS_POLICY_ID
    return out


@bounded_cpu
def test(methods=None):
    report_compute()
    inp = load_inputs()
    selected = selected_settings()
    methods = list(METHODS) if methods is None else methods
    fixed, daily = forecast_origins(inp.dates, 'fixed_origin'), forecast_origins(inp.dates, 'daily_update')
    require_complete_features(inp.features['base_vol_news'], inp.dates, daily)
    jobs = [(m, h) for m in methods for h in horizons(m, inp)]
    missing = [f'{m}/h{h}' for m, h in jobs if f'{m}/h{h}' not in selected]
    if missing:
        raise ValueError(f'run validation first; missing settings: {missing}')
    cache = ValidationCache('distribution_final', input_frames(inp))
    results = run_jobs(_final_job, jobs, (inp, selected, fixed, daily), 'distribution final', cache,
                       cache_key=lambda t: (t, resolve(t[0], selected[f'{t[0]}/h{t[1]}']['params'], t[1], selected)))
    frames = []
    for (m, h), Q in zip(jobs, results):
        frames.append(frame_forecasts(inp, m, h, fixed, Q[:1], 'fixed_origin'))
        if h in HORIZONS:
            frames.append(frame_forecasts(inp, m, h, daily, Q, 'daily_update'))
    if all(m in methods for m in ENSEMBLE):
        for h in fixed_path_horizons(inp.dates):
            Q = arrange_all(np.mean([r for (m, hh), r in zip(jobs, results) if hh == h and m in ENSEMBLE], axis=0))
            frames.append(frame_forecasts(inp, 'ensemble', h, fixed, Q[:1], 'fixed_origin'))
            if h in HORIZONS:
                frames.append(frame_forecasts(inp, 'ensemble', h, daily, Q, 'daily_update'))
    save_manifest()
    path = OUT / 'test_quantiles.csv'
    if path.exists() and set(methods) != set(METHODS):
        existing = pd.read_csv(path, parse_dates=['origin_date', 'target_date'])
        if not existing['protocol_id'].eq(PROTOCOL_ID).all():
            raise ValueError('cannot append to a different forecast protocol')
        if 'news_policy_id' not in existing or not existing.news_policy_id.eq(NEWS_POLICY_ID).all():
            raise ValueError('cannot append forecasts from a different news policy')
        if existing['method'].isin(methods).any():
            raise RuntimeError('partial append already has this method; rerun the complete test to refresh ensembles consistently')
        replaced = methods + (['ensemble'] if all(m in methods for m in ENSEMBLE) else [])
        frames.insert(0, existing[~existing.method.isin(replaced)])
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(path, index=False, date_format='%Y-%m-%d', lineterminator='\n')
    print('wrote', path, out.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['validate', 'test'])
    parser.add_argument('--methods', nargs='+', choices=list(METHODS))
    args = parser.parse_args()
    validate(args.methods or list(METHODS)) if args.stage == 'validate' else test(args.methods)

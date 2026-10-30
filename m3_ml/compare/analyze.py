"""Strict, mode-separated October comparison (never reads legacy outputs).

Run: uv run python -m m3_ml.compare.analyze
All October statistics are descriptive; the short sample cannot support DM/HAC
inference. Unmatured horizons retain an explicit n=0 rather than disappearing.
"""
import json

import numpy as np
import pandas as pd

from m3_ml.distribution.quantiles import QUANTILE_COLUMNS, covered, crps
from m3_ml.distribution.tails import ALL_COLUMNS, mean_fz0

from .common import (
    HORIZONS, KEYS, MODES, OUT, PROVENANCE, ROOT, TEST_END, TEST_START,
    forecast_horizons, forecast_origins, load_prices, read_forecasts, validate_forecasts,
)

METHODS = [
    ('rw', 'Random walk', 'Benchmark', 'm3_ml/tabular'),
    ('tab_hist_mean', 'Historical mean', 'Benchmark', 'm3_ml/tabular'),
    ('tab_elastic_net', 'Elastic Net', 'ML direct', 'm3_ml/tabular'),
    ('tab_random_forest', 'Random Forest', 'ML direct', 'm3_ml/tabular'),
    ('tab_lightgbm', 'LightGBM', 'ML direct', 'm3_ml/tabular'),
    ('p1_arima', 'Python ARIMA(0,1,1), fixed specification', 'Python classical study', 'm3_ml/compare/part1.py'),
    ('p1_arimax', 'Python ARIMAX(0,1,1), fixed specification', 'Python classical study', 'm3_ml/compare/part1.py'),
    ('p1_vecm', 'Python VECM, independently selected', 'Python classical study', 'm3_ml/compare/part1.py'),
    ('lstm', 'LSTM', 'Sequence', 'm3_ml/lstm'),
]
DIST_METHODS = [
    ('hs', 'Historical simulation', 'Benchmark', 'm3_ml/distribution'),
    ('ewma', 'EWMA', 'Volatility', 'm3_ml/distribution'),
    ('garch', 'GARCH', 'Volatility', 'm3_ml/distribution'),
    ('garch_evt', 'GARCH-EVT', 'Volatility', 'm3_ml/distribution'),
    ('a_ridge', 'Ridge volatility', 'ML distribution', 'm3_ml/distribution'),
    ('a_lightgbm', 'LightGBM volatility', 'ML distribution', 'm3_ml/distribution'),
    ('a_evt', 'EVT tails', 'ML distribution', 'm3_ml/distribution'),
    ('b_linear', 'Linear quantile regression', 'ML distribution', 'm3_ml/distribution'),
    ('b_lightgbm', 'LightGBM quantile regression', 'ML distribution', 'm3_ml/distribution'),
    ('b_forest', 'Quantile forest', 'ML distribution', 'm3_ml/distribution'),
    ('ensemble', 'Quantile ensemble', 'ML distribution', 'm3_ml/distribution'),
    ('c_mixture', 'Variance mixture (retrospective)', 'ML distribution', 'm3_ml/distribution'),
    ('p1_arima', 'Python ARIMA Gaussian, fixed specification', 'Python classical study', 'm3_ml/compare/part1.py'),
    ('p1_arima_garch', 'Python two-step ARIMA-GARCH approximation', 'Python classical study', 'm3_ml/compare/part1.py'),
]
CORE_DIST = [row[0] for row in DIST_METHODS if not row[0].startswith('p1_')]
TAB_NAMES = {'random_walk': 'rw', 'historical_mean': 'tab_hist_mean',
             'elastic_net': 'tab_elastic_net', 'random_forest': 'tab_random_forest', 'lightgbm': 'tab_lightgbm'}


def require_methods(frame, expected, source):
    present = set(frame['method'])
    if present != set(expected):
        raise ValueError(f'{source}: method mismatch; missing={sorted(set(expected) - present)}, unexpected={sorted(present - set(expected))}')


def grid(table, mode, horizon):
    origins = forecast_origins(table, mode)
    return {(table['date'].iloc[t], int(t + horizon)) for t in origins
            if horizon in forecast_horizons(table, int(t), mode)}


def validate_coverage(frame, methods, distribution=False):
    """Require a complete forecast grid; never rescue absent methods by intersection."""
    table, _ = load_prices()
    for mode in MODES:
        horizons = forecast_horizons(table, int(forecast_origins(table, mode)[0]), mode)
        for method in methods:
            for horizon in horizons:
                if distribution and method == 'c_mixture' and horizon == 1:
                    continue
                expected = grid(table, mode, horizon)
                group = frame[(frame['mode'] == mode) & (frame['method'] == method) & (frame['horizon'] == horizon)]
                actual = set(zip(group['origin_date'], group['target_row']))
                if actual != expected:
                    raise ValueError(f'{mode}/{method}/h{horizon}: incomplete or unexpected forecast grid: expected {len(expected)}, got {len(actual)}; no common-origin fallback')
    if frame.duplicated(KEYS + ['method']).any():
        raise ValueError('Duplicate forecast rows across comparison sources')
    # Anchors and outcomes must agree for the identical information clock.
    for _, group in frame.groupby(KEYS):
        anchors = group['origin_price'].to_numpy()
        if not np.isfinite(anchors).all() or not np.allclose(anchors, anchors[0], atol=1e-6):
            raise ValueError('Observable anchors differ between methods on the common clock')
        actual = group['actual'].dropna().to_numpy()
        if len(actual) and not np.allclose(actual, actual[0], atol=1e-6):
            raise ValueError('Realized targets differ between methods')


def distribution_forecasts():
    core_path = ROOT / 'm3_ml/distribution/outputs/frozen_october_2026/test_quantiles.csv'
    core = read_forecasts(core_path, ['actual_log_return', *ALL_COLUMNS])
    require_methods(core, CORE_DIST, core_path)
    p1 = read_forecasts(OUT / 'part1_quantiles.csv', ['actual_log_return', *ALL_COLUMNS])
    require_methods(p1, ['p1_arima', 'p1_arima_garch'], 'classical quantiles')
    frame = pd.concat([core, p1], ignore_index=True)
    if 'actual' not in frame or frame['actual'].isna().any():
        implied = frame['origin_price'] * np.exp(frame['actual_log_return'])
        frame['actual'] = frame.get('actual', pd.Series(np.nan, index=frame.index)).fillna(implied)
    implied = frame['origin_price'] * np.exp(frame['actual_log_return'])
    scored = frame['target_date'].between(TEST_START, TEST_END)
    if frame.loc[scored, 'actual_log_return'].isna().any() or not np.allclose(frame.loc[scored, 'actual'], implied[scored], atol=1e-6):
        raise ValueError('Distribution actual log returns do not match the observable-anchor price targets')
    if not np.isfinite(frame[ALL_COLUMNS].to_numpy()).all():
        raise ValueError('Missing or nonfinite distribution predictions; regenerate rather than intersect')
    # Existing scoring rearranges the main grid and tail grids independently.
    if (np.diff(frame[QUANTILE_COLUMNS].to_numpy(), axis=1) < -1e-10).any():
        raise ValueError('Crossed main-grid distribution quantiles')
    validate_coverage(frame, [row[0] for row in DIST_METHODS], distribution=True)
    return frame


def external_price_forecasts():
    """Import actual R results, preserving names and explicit source conventions."""
    table, _ = load_prices()
    dates_to_rows = pd.Series(np.arange(len(table)), index=table['date'])
    m1_directory = ROOT / 'm1_time_series/outputs/frozen_october_2026'
    m1_path = m1_directory / 'forecasts_2026-10-01_2026-10-23_asof_2026-10-23.csv'
    manifest_path = m1_directory / 'manifest_2026-10-01_2026-10-23_asof_2026-10-23.csv'
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != 1:
        raise ValueError(f'{manifest_path}: expected one current selection manifest')
    for name, value in PROVENANCE.items():
        if name not in manifest or not manifest[name].astype(str).eq(value).all():
            raise ValueError(f'{manifest_path}: incompatible or absent {name}')
    if 'selection_signature' not in manifest or manifest['selection_signature'].isna().any():
        raise ValueError(f'{manifest_path}: missing fresh R validation selection signature; regenerate M1')
    signature = str(manifest['selection_signature'].iloc[0])
    if not signature.strip():
        raise ValueError(f'{manifest_path}: empty R selection signature')
    selected = pd.read_csv(m1_directory / 'selected_config.csv')
    required_selected = {'family', 'method', 'protocol_id', 'selection_signature', 'input_md5'}
    if required_selected - set(selected) or len(selected) != 4 or set(selected['family']) != {'ARIMA', 'ARIMAX', 'GARCH', 'VECM'}:
        raise ValueError('The actual R selected-method registry must contain its four selected families')
    if not selected['protocol_id'].eq(PROVENANCE['protocol_id']).all() or not selected['selection_signature'].astype(str).eq(signature).all():
        raise ValueError('R selection manifest and selected-method registry have incompatible provenance')
    if 'selection_input_md5' not in manifest or manifest['selection_input_md5'].isna().any():
        raise ValueError('R final manifest lacks selection input evidence')
    if not selected['input_md5'].astype(str).eq(str(manifest['selection_input_md5'].iloc[0])).all():
        raise ValueError('R final manifest and selected-method registry refer to different selection inputs')
    checkpoint = m1_directory / 'frozen_fit_selected_2026-09-30.rds'
    if not checkpoint.is_file() or 'bundle' not in manifest or str(manifest['bundle'].iloc[0]).replace('\\', '/').split('/')[-1] != checkpoint.name:
        raise ValueError('R final manifest does not identify the selected-specification checkpoint')
    # The R runner commits its dated final manifest after replacing forecasts.
    # Its selection signature is absent from historical fixed-spec smoke runs.
    m1 = pd.read_csv(m1_path)
    require_methods(m1, set(selected['method']) | {'RW'}, m1_path)
    if 'selection_signature' in m1 and not m1['selection_signature'].astype(str).eq(signature).all():
        raise ValueError('R forecast rows have a different selection signature from the final manifest')
    m2_path = ROOT / 'm2_seasonality/outputs/frozen_october_2026/final_predictions.csv'
    m2 = pd.read_csv(m2_path)
    if 'target_type' not in m2:
        raise ValueError(f'{m2_path}: missing target_type')
    m2 = m2[m2['target_type'].eq('price')].copy()
    require_methods(m2, ['random_walk', 'calendar_mean_original', 'calendar_mean_3iqr',
                         'stl_monthly_daily_bridge', 'stl_weekly_daily_bridge'], m2_path)
    frames, catalog = [], []
    for source, path, frame, row_base, family in (
            ('m1', m1_path, m1, 0, 'Actual R selected time series'),
            ('m2', m2_path, m2, 1, 'Actual R seasonality')):
        required = {'mode', 'origin_date', 'target_date', 'horizon', 'method', 'forecast', 'actual',
                    'origin_price', 'protocol_id', 'fit_end'}
        if required - set(frame):
            raise ValueError(f'{path}: missing required columns {sorted(required - set(frame))}')
        if frame.empty or set(frame['mode']) != set(MODES) or frame['method'].isna().any():
            raise ValueError(f'{path}: both modes and nonempty named methods required')
        frame = frame.copy()
        frame['origin_date'] = pd.to_datetime(frame['origin_date'], errors='raise')
        origin_rows = frame['origin_date'].map(dates_to_rows)
        if origin_rows.isna().any():
            raise ValueError(f'{path}: unrecognized source origin dates')
        if 'origin_row' in frame and not frame['origin_row'].sub(row_base).eq(origin_rows).all():
            raise ValueError(f'{path}: source origin row convention is not the declared {row_base}-based grid')
        if 'origin_row' in frame:
            frame['source_origin_row'] = frame['origin_row']
        frame['origin_row'] = origin_rows
        frame['source_row_base'] = row_base
        if 'target_row' in frame:
            frame['source_target_row'] = frame['target_row']
            frame['target_row'] = frame['target_row'] - row_base
        else:
            frame['target_row'] = origin_rows + frame['horizon']
        source_methods = sorted(frame['method'].unique())
        keep = pd.Series(False, index=frame.index)
        for mode in MODES:
            origins = forecast_origins(table, mode)
            horizons = forecast_horizons(table, int(origins[0]), mode)
            keep |= (frame['mode'].eq(mode) & frame['origin_date'].isin(table['date'].iloc[origins])
                     & frame['horizon'].isin(horizons))
        frame = frame.loc[keep].copy()
        require_methods(frame, source_methods, path)
        frame = validate_forecasts(frame, path, ['forecast', 'actual'],
                                   provenance={name: PROVENANCE[name] for name in ('protocol_id', 'fit_end')})
        frame['source_method'] = frame['method']
        frame['source'] = path.relative_to(ROOT).as_posix()
        frame['method'] = source + '_' + frame['method']
        if source == 'm1':
            frame['source_selection_signature'] = signature
            for name in PROVENANCE:
                if name not in frame:
                    frame[name] = manifest[name].iloc[0]
        for method in source_methods:
            catalog.append((source + '_' + method, source.upper() + ' R: ' + method, family,
                            path.relative_to(ROOT).as_posix()))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), catalog


def point_forecasts(distributions):
    path = ROOT / 'm3_ml/tabular/outputs/frozen_october_2026/direct_forecasts.csv'
    tab = read_forecasts(path, ['forecast', 'actual'])
    if 'variant' in tab:
        tab = tab[tab['variant'].eq('direct')].copy()
    names = tab['method'].map(TAB_NAMES)
    if names.isna().any():
        raise ValueError(f'{path}: unexpected tabular method')
    tab['method'] = names
    require_methods(tab, list(TAB_NAMES.values()), path)
    classical = read_forecasts(OUT / 'part1_forecasts.csv', ['forecast', 'actual'])
    require_methods(classical, ['p1_arima', 'p1_arimax', 'p1_vecm'], 'classical point forecasts')
    sequence = read_forecasts(OUT / 'lstm_forecasts.csv', ['forecast', 'actual'])
    require_methods(sequence, ['lstm', 'lstm_persistence'], 'sequence adapter')
    if not np.allclose(sequence.loc[sequence['method'].eq('lstm_persistence'), 'forecast'],
                       sequence.loc[sequence['method'].eq('lstm_persistence'), 'origin_price'], atol=1e-6):
        raise ValueError('Sequence persistence is not the observable-anchor random walk')
    sequence = sequence[sequence['method'].eq('lstm')]
    medians = distributions[distributions['method'].isin(CORE_DIST)].copy()
    medians['forecast'] = medians['origin_price'] * np.exp(medians['q_0.500'])
    medians['method'] = 'dist_' + medians['method']
    external, external_catalog = external_price_forecasts()
    frame = pd.concat([tab, classical, sequence, medians, external], ignore_index=True)
    expected = [row[0] for row in METHODS + external_catalog] + ['dist_' + method for method in CORE_DIST]
    require_methods(frame, expected, 'point comparison')
    if not np.isfinite(frame['forecast'].to_numpy()).all():
        raise ValueError('Missing or nonfinite point predictions; no common-origin fallback')
    # The mixture's unsupported h1 is explicit, not an empty intersection.
    validate_coverage(frame, [m for m in expected if m != 'dist_c_mixture'])
    validate_coverage(frame[frame['method'].eq('dist_c_mixture')].assign(method='c_mixture'), ['c_mixture'], distribution=True)
    return frame, external_catalog


def ratio(numerator, denominator):
    return float(numerator / denominator) if np.isfinite(denominator) and denominator > 0 else np.nan


def sample_status(n):
    return 'unscored_no_matured_test_targets' if n == 0 else 'small_sample_descriptive_only'


def point_metrics(frame, methods):
    rows = []
    for mode in MODES:
        rw = frame[(frame['mode'] == mode) & (frame['method'] == 'rw')].set_index(['origin_date', 'horizon'])
        for method in methods:
            for horizon in HORIZONS:
                group = frame[(frame['mode'] == mode) & (frame['method'] == method) & (frame['horizon'] == horizon)]
                scored = group.dropna(subset=['actual'])
                n = len(scored)
                row = {'mode': mode, 'method': method, 'horizon': horizon, 'n': n,
                       'n_forecasts': len(group), 'status': sample_status(n),
                       'rmse': np.nan, 'mae': np.nan, 'rel_rmse': np.nan, 'rel_mae': np.nan,
                       'direction': np.nan, 'n_direction': 0, 'dm_p': np.nan}
                if n:
                    reference = rw.reindex(pd.MultiIndex.from_frame(scored[['origin_date', 'horizon']]))
                    error = scored['forecast'].to_numpy() - scored['actual'].to_numpy()
                    baseline = reference['forecast'].to_numpy() - scored['actual'].to_numpy()
                    call = np.sign(scored['forecast'].to_numpy() - scored['origin_price'].to_numpy())
                    move = np.sign(scored['actual'].to_numpy() - scored['origin_price'].to_numpy())
                    active = (call != 0) & (move != 0)
                    mse, bmse = np.mean(error ** 2), np.mean(baseline ** 2)
                    row.update(rmse=float(np.sqrt(mse)), mae=float(np.mean(np.abs(error))),
                               rel_rmse=float(np.sqrt(ratio(mse, bmse))),
                               rel_mae=ratio(np.mean(np.abs(error)), np.mean(np.abs(baseline))),
                               direction=float(np.mean(call[active] == move[active])) if active.any() else np.nan,
                               n_direction=int(active.sum()))
                if method == 'dist_c_mixture' and horizon == 1:
                    row['status'] = 'not_supported_by_model'
                rows.append(row)
    return pd.DataFrame(rows)


def dist_metrics(frame):
    rows = []
    for mode in MODES:
        for horizon in HORIZONS:
            subset = frame[(frame['mode'] == mode) & (frame['horizon'] == horizon)]
            scores = {}
            for method, group in subset.groupby('method'):
                group = group.dropna(subset=['actual_log_return']).sort_values('origin_date')
                if group.empty:
                    continue
                quantiles = group[QUANTILE_COLUMNS].to_numpy()
                y = group['actual_log_return'].to_numpy()
                scores[method] = pd.DataFrame({'crps': crps(quantiles, y),
                                              'fz0': mean_fz0(group[ALL_COLUMNS].to_numpy(), y),
                                              'hit90': covered(quantiles, y, 0.9).astype(float)}, index=group['origin_date'])
            for method, *_ in DIST_METHODS:
                s = scores.get(method)
                n = 0 if s is None else len(s)
                row = {'mode': mode, 'method': method, 'horizon': horizon, 'n': n,
                       'n_forecasts': int(subset['method'].eq(method).sum()), 'status': sample_status(n),
                       'crps': np.nan, 'crps_rel_hs': np.nan, 'crps_rel_ewma': np.nan, 'cov90': np.nan,
                       'fz0_minus_ewma': np.nan, 'n_fz0': 0, 'dm_p_ewma': np.nan, 'cov90_p': np.nan, 'fz0_p_ewma': np.nan}
                if n:
                    hs = scores['hs'].reindex(s.index)
                    ewma = scores['ewma'].reindex(s.index)
                    if hs.isna().all().all() or ewma.isna().all().all():
                        raise ValueError('Distribution reference methods lack scored targets')
                    row.update(crps=float(s['crps'].mean()), crps_rel_hs=ratio(s['crps'].mean(), hs['crps'].mean()),
                               crps_rel_ewma=ratio(s['crps'].mean(), ewma['crps'].mean()), cov90=float(s['hit90'].mean()),
                               fz0_minus_ewma=float((s['fz0'] - ewma['fz0']).mean()),
                               n_fz0=int(np.isfinite(s['fz0'] - ewma['fz0']).sum()))
                if method == 'c_mixture' and horizon == 1:
                    row['status'] = 'not_supported_by_model'
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    distributions = distribution_forecasts()
    points, external_catalog = point_forecasts(distributions)
    point_catalog = METHODS + external_catalog + [('dist_' + m, label + ' (median)', 'Distribution median', source)
                                                  for m, label, _, source in DIST_METHODS if m in CORE_DIST]
    pm = point_metrics(points, [row[0] for row in point_catalog])
    dm = dist_metrics(distributions)
    for frame, catalog in ((pm, point_catalog), (dm, DIST_METHODS)):
        labels = {m: (label, family) for m, label, family, _ in catalog}
        frame.insert(2, 'label', frame['method'].map(lambda m: labels[m][0]))
        frame.insert(3, 'family', frame['method'].map(lambda m: labels[m][1]))
    OUT.mkdir(parents=True, exist_ok=True)
    pm.to_csv(OUT / 'point_metrics.csv', index=False)
    dm.to_csv(OUT / 'dist_metrics.csv', index=False)
    points.to_csv(OUT / 'point_forecasts.csv', index=False, date_format='%Y-%m-%d')
    pd.DataFrame(point_catalog, columns=['method', 'label', 'family', 'source']).to_csv(OUT / 'methods_point.csv', index=False)
    pd.DataFrame(DIST_METHODS, columns=['method', 'label', 'family', 'source']).to_csv(OUT / 'methods_distribution.csv', index=False)
    protocol = {**PROVENANCE, 'test_target_start': TEST_START, 'test_target_end': TEST_END,
                'modes': list(MODES), 'horizons': list(HORIZONS),
                'results_status': 'retrospective frozen-parameter October evaluation; not a new blind holdout',
                'availability': 'matched historical WPSR evidence where available; unmatched weekly release assumption; cross-variable common WTI publication schedule assumption',
                'inference': 'all October scores descriptive; no DM/HAC p-values on this short sample',
                'external_point_sources': [{'method': method, 'label': label, 'family': family, 'source': source}
                                           for method, label, family, source in external_catalog],
                'point_counts': pm[['mode', 'method', 'horizon', 'n', 'n_forecasts', 'status']].to_dict('records'),
                'distribution_counts': dm[['mode', 'method', 'horizon', 'n', 'n_forecasts', 'status']].to_dict('records')}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    for mode in MODES:
        print('\n', mode, '(descriptive only; each row reports n)')
        print(pm[pm['mode'].eq(mode)][['method', 'horizon', 'n', 'rel_rmse', 'status']].to_string(index=False))
        print(dm[dm['mode'].eq(mode)][['method', 'horizon', 'n', 'crps_rel_ewma', 'status']].to_string(index=False))


if __name__ == '__main__':
    main()

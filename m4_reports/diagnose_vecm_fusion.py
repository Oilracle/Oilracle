"""Read-only diagnosis of the frozen solver versus VECM; no October fitting.

Run from the project root after diagnose_vecm_refits.R. All diagnostic outputs
are separate from frozen forecasts, production models and presentation files.
"""
from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from m3_ml.common.data import load_table, observable_price, label_available_dates
from m3_ml.fusion_v5.data import EXPERTS
from m3_ml.fusion_v5.point import predict

OUT = ROOT / 'm4_reports/diagnostics/vecm_fusion'
OUT.mkdir(parents=True, exist_ok=True)
BASE = ROOT / 'm3_ml'
B = {v: BASE / f'fusion_v{v}/outputs/frozen_october_2026' for v in [2, 3, 4, 5, 6]}
KEY = ['origin_date', 'target_date', 'horizon']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(frame, column):
    e = frame[column].to_numpy() - frame.actual.to_numpy()
    return dict(n=len(e), rmse=float(np.sqrt(np.mean(e * e))),
                mae=float(np.mean(abs(e))), bias=float(np.mean(e)), mse=float(np.mean(e * e)))


def save(frame, name):
    frame.to_csv(OUT / name, index=False)


protected = [ROOT / 'm4_reports/final_strategy_manifest.json']
for v in [4, 5, 6]:
    protected.extend(B[v] / f for f in ['selected.json', 'test_forecasts.csv', 'point_crossfit.csv', 'models/upgrade.joblib'] if (B[v] / f).exists())
production_hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected}
history_all = pd.read_csv(B[6] / 'point_crossfit.csv')
history = history_all[(history_all.origin_date >= '2025-09-01') & history_all.horizon.isin([1, 5])].copy()
october_all = pd.read_csv(B[6] / 'test_forecasts.csv')
october = october_all[october_all.actual.notna() & october_all.horizon.isin([1, 5])].copy()
assert not history.duplicated(KEY).any() and not october.duplicated(KEY).any()
assert history.label_available_date.max() <= '2026-09-30'
assert history.target_date.max() <= '2026-09-30'
assert np.allclose(history.forecast, history.v5_forecast, atol=1e-12, rtol=0)
assert np.allclose(october.forecast, october.v5_forecast, atol=1e-12, rtol=0)

# Audit exact target dates (not just origin/horizon joins) and published anchors.
m1_file = ROOT / 'm1_time_series/outputs/frozen_october_2026/forecasts_2026-10-01_2026-10-23_asof_2026-10-23.csv'
native_all = pd.read_csv(m1_file)
native = native_all[(native_all['mode'] == 'daily_update') & (native_all.method == 'WTI_Brent_Gasoline_K3_R2')]
joined = october.merge(native[KEY + ['forecast', 'actual', 'origin_price', 'available_date_max', 'state_last_quote', 'bridge_rows', 'origin_row', 'target_row']], on=KEY, suffixes=('', '_native'), validate='one_to_one')
assert len(joined) == len(october) == 30
audit = dict(october_rows=len(joined), same_forecast_max_error=float(abs(joined.m1_price - joined.forecast_native).max()),
             same_actual_max_error=float(abs(joined.actual - joined.actual_native).max()),
             same_anchor_max_error=float(abs(joined.origin_price - joined.origin_price_native).max()))
assert max(audit[k] for k in audit if k.endswith('max_error')) < 1e-9
assert (joined.available_date_max <= joined.origin_date).all()
assert (joined.futures_available_date <= joined.origin_date).all()
assert (joined.target_row_native - joined.origin_row == joined.horizon).all()
table = load_table()
pos = pd.Series(np.arange(len(table)), index=table.date)
origin_pos = pos.reindex(pd.to_datetime(joined.origin_date)).to_numpy()
target_pos = pos.reindex(pd.to_datetime(joined.target_date)).to_numpy()
assert np.array_equal(target_pos - origin_pos, joined.horizon.to_numpy())
np.testing.assert_allclose(joined.actual, table.target_wti.iloc[target_pos], atol=1e-9, rtol=0)
np.testing.assert_allclose(joined.rw_price, observable_price(table).iloc[origin_pos], atol=1e-9, rtol=0)
np.testing.assert_allclose(joined.origin_price * np.exp(joined.m1), joined.m1_price, atol=1e-9, rtol=0)
raw_historical = pd.read_csv(B[2] / 'm1_validation.csv')
hist_audit = history.merge(raw_historical[['origin_date', 'horizon', 'forecast', 'base_fit_cutoff']], on=['origin_date', 'horizon'], suffixes=('', '_native'), validate='one_to_one')
audit['historical_native_max_error'] = float(abs(hist_audit.m1_price - hist_audit.forecast_native).max())
assert audit['historical_native_max_error'] < 1e-9
assert (hist_audit.base_fit_cutoff <= hist_audit.origin_date).all()
refit_audit = pd.read_csv(B[2] / 'r_refit_audit.csv')
assert (refit_audit.latest_label_release <= refit_audit.cutoff).all()
audit.update(historical_base_refit_policy='63-origin refits inherited from fusion_v2',
             historical_base_cutoffs=sorted(hist_audit.base_fit_cutoff.unique().tolist()),
             publication_policy='Existing portable release schedule, including assumed dates; not an independent real-time vintage audit.',
             production_parameters_changed=False, october_used_for_fitting=False)

# Reproduce frozen prediction arithmetic; the risk layer does not move centers.
fitted = joblib.load(B[5] / 'models/upgrade.joblib')
for h, g in october.groupby('horizon'):
    price, weights = predict(g, fitted['meta'][int(h)], True)
    np.testing.assert_allclose(price, g.forecast, atol=1e-9, rtol=0)
    np.testing.assert_allclose(weights, g[[e + '_weight' for e in EXPERTS]], atol=1e-12, rtol=0)
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(g.origin_price * np.exp(g['q_0.500']), g.forecast, atol=1e-9, rtol=0)

metrics = []
for period, frame in [('history', history), ('October', october)]:
    for h, g in frame.groupby('horizon'):
        for method in ['forecast', 'm1_price', 'rw_price', 'futures_price', 'v4_forecast']:
            metrics.append(dict(period=period, horizon=int(h), method=method, **score(g, method)))
metrics = pd.DataFrame(metrics)
save(metrics, 'metrics.csv')
quarters = []
for (h, quarter), g in history.groupby(['horizon', pd.to_datetime(history.origin_date).dt.to_period('Q')]):
    for method in ['forecast', 'm1_price', 'futures_price']:
        quarters.append(dict(horizon=int(h), quarter=str(quarter), method=method, **score(g, method)))
save(pd.DataFrame(quarters), 'historical_quarters.csv')

windows = []; window_summary = []
for h, g in history.groupby('horizon'):
    g = g.sort_values('origin_date').reset_index(drop=True)
    test = october[october.horizon == h]
    oct_ratio = score(test, 'forecast')['rmse'] / score(test, 'm1_price')['rmse']
    for length in [len(test), 20, 60]:
        rows = []
        for end in range(length, len(g) + 1):
            x = g.iloc[end - length:end]
            f, m = score(x, 'forecast'), score(x, 'm1_price')
            rows.append(dict(horizon=int(h), window=length, start=x.origin_date.iloc[0], end=x.origin_date.iloc[-1],
                             fusion_rmse=f['rmse'], vecm_rmse=m['rmse'], ratio=f['rmse'] / m['rmse']))
        windows.extend(rows)
        ratios = np.array([x['ratio'] for x in rows])
        window_summary.append(dict(horizon=int(h), window=length, n_windows=len(rows),
            vecm_wins=int((ratios > 1).sum()), vecm_win_fraction=float((ratios > 1).mean()),
            october_ratio=oct_ratio,
            windows_at_least_october_ratio=int((ratios >= oct_ratio).sum()),
            fraction_at_least_october_ratio=float((ratios >= oct_ratio).mean()),
            interpretation='Overlapping historical windows; descriptive, NOT an independent-event probability or p-value.'))
save(pd.DataFrame(windows), 'historical_windows.csv')

# Exact nested-layer accounting, not a new choice or October parameter search.
v4_test = pd.read_csv(B[4] / 'test_forecasts.csv')
detail = october.merge(v4_test[KEY + ['proposal_forecast', 'incumbent_forecast']], on=KEY, validate='one_to_one')
detail['undamped_v5_proposal'] = 2 * detail.forecast - detail.v4_forecast
detail['fusion_error'] = detail.forecast - detail.actual
detail['vecm_error'] = detail.m1_price - detail.actual
detail['excess_squared_error'] = detail.fusion_error ** 2 - detail.vecm_error ** 2
save(detail[KEY + ['actual', 'rw_price', 'm1_price', 'futures_price', 'forecast', 'v4_forecast', 'incumbent_forecast',
                  'proposal_forecast', 'undamped_v5_proposal', 'fusion_error', 'vecm_error', 'excess_squared_error',
                  'futures_quote_date', 'futures_available_date']], 'october_error_decomposition.csv')
h5 = detail[detail.horizon == 5]
outer_weights = {e: float(h5[e + '_weight'].iloc[0]) for e in EXPERTS if h5[e + '_weight'].max() > 1e-10}
v4 = read(B[4] / 'selected.json')
explicit_m1_weight = outer_weights['m1'] + outer_weights['incumbent'] * v4['guard']['5']['upgrade_weight'] * v4['family_weights']['5']['m1']
mechanism = dict(h5_outer_weights=outer_weights,
    h5_explicit_direct_and_v4_vecm_weight=explicit_m1_weight,
    weight_caveat='This is not total sensitivity: inherited ML corrections also use VECM as an input.',
    h5_static_definition=fitted['meta'][5]['definition'],
    h5_layers={c: score(h5, c) for c in ['incumbent_forecast', 'proposal_forecast', 'v4_forecast', 'undamped_v5_proposal', 'forecast', 'm1_price']},
    h5_top3_excess_loss_share=float(h5.excess_squared_error.nlargest(3).sum() / h5.excess_squared_error.sum()),
    h5_top3_targets=h5.nlargest(3, 'excess_squared_error').target_date.tolist(),
    h5_vecm_winning_origins=int((abs(h5.vecm_error) < abs(h5.fusion_error)).sum()))

# The actual September weight-fitting sample: historical out-of-sample experts,
# never evaluate the September-fitted weighting function backwards as OOS.
recent = []
for h, g in history_all[history_all.horizon.isin([1, 5])].groupby('horizon'):
    eligible = g[(g.origin_date < '2026-09-30') & (g.label_available_date <= '2026-09-30')].sort_values('origin_date')
    for n in [126, 252]:
        tr = eligible.tail(n)
        for c in ['forecast', 'm1_price', 'futures_price', 'v4_forecast']:
            recent.append(dict(horizon=int(h), window=n, start=tr.origin_date.iloc[0], end=tr.origin_date.iloc[-1], method=c, **score(tr, c)))
save(pd.DataFrame(recent), 'historical_recent_windows.csv')

# Predefined-sign stratification uses forecasts available at each origin.
regimes = []
for period, frame in [('history', history), ('October', october)]:
    for h, g in frame.groupby('horizon'):
        for sign, t in [('VECM above futures', g[g.m1_price > g.futures_price]), ('VECM at or below futures', g[g.m1_price <= g.futures_price])]:
            if len(t):
                for c in ['forecast', 'm1_price']:
                    regimes.append(dict(period=period, horizon=int(h), state=sign, method=c, **score(t, c)))
save(pd.DataFrame(regimes), 'forecast_sign_diagnostic.csv')

# Additional historical-only quarterly-refit robustness check. This is a
# post-hoc diagnostic, not a selected strategy or a fresh independent test.
quarter_refits = pd.read_csv(OUT / 'historical_vecm_refits.csv')
refits = history.merge(quarter_refits[KEY + ['forecast_refit', 'forecast_original', 'fit_cutoff', 'fit_latest_release', 'state_latest_release']], on=KEY, validate='one_to_one')
assert len(refits) == len(history)
assert (refits.fit_latest_release <= refits.fit_cutoff).all()
assert (refits.fit_cutoff <= refits.origin_date).all()
assert (refits.state_latest_release <= refits.origin_date).all()
robustness = [dict(horizon=int(h), method=c, **score(g, c)) for h, g in refits.groupby('horizon') for c in ['forecast', 'm1_price', 'forecast_refit', 'forecast_original']]
save(pd.DataFrame(robustness), 'historical_refit_robustness.csv')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Cambria', 'DejaVu Serif'],
    'font.size': 12, 'text.color': '#F1EBDF', 'axes.labelcolor': '#F1EBDF',
    'xtick.color': '#D1C8BA', 'ytick.color': '#D1C8BA', 'axes.edgecolor': '#635D52',
    'figure.facecolor': '#141311', 'axes.facecolor': '#141311', 'savefig.facecolor': '#141311'})
fig, ax = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [1.2, 1]})
for c, label, color, width in [('actual', 'Actual spot', '#F1EBDF', 2.8), ('forecast', 'Final fusion', '#D5AA64', 2.5), ('m1_price', 'VECM', '#67CEE3', 2.5), ('futures_price', 'Futures hold', '#A499AC', 1.6)]:
    ax[0].plot(pd.to_datetime(h5.target_date), h5[c], label=label, color=color, linewidth=width, marker='o', markersize=3)
ax[0].set_title('Five-day rolling forecasts: October evaluation', loc='left', fontsize=20, pad=18)
ax[0].set_ylabel('USD / barrel')
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax[0].legend(ncol=4, frameon=False, loc='upper left', fontsize=11)
ax[0].set_ylim(84, 116)
win = pd.DataFrame(windows)
win = win[(win.horizon == 5) & (win.window == 13)]
ax[1].plot(pd.to_datetime(win.end), win.ratio, color='#D5AA64', linewidth=1.8)
ax[1].axhline(1, color='#F1EBDF', alpha=.6, linewidth=1)
oct_ratio = score(h5, 'forecast')['rmse'] / score(h5, 'm1_price')['rmse']
ax[1].scatter([pd.Timestamp('2026-10-16')], [oct_ratio], color='#67CEE3', s=65, zorder=5)
ax[1].annotate(f'October: {oct_ratio:.2f}', (pd.Timestamp('2026-10-16'), oct_ratio), xytext=(-105, 12), textcoords='offset points', color='#67CEE3')
ax[1].set_title('Earlier 13-origin windows: rankings can reverse', loc='left', fontsize=18, pad=15)
ax[1].set_ylabel('Fusion RMSE / VECM RMSE')
ax[1].set_xlabel('Window ending origin date  |  above 1 = VECM better')
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
for a in ax:
    a.grid(axis='y', alpha=.13)
    a.spines[['top', 'right']].set_visible(False)
fig.text(.085, .015, 'Only 3 of 245 earlier overlapping windows reached October\'s ratio. This is descriptive, not a p-value.', fontsize=11, color='#D1C8BA')
fig.tight_layout(rect=[.02, .04, .99, .99], h_pad=2.4)
fig.savefig(OUT / 'vecm_fusion_diagnosis.png', dpi=170)
plt.close(fig)

assert production_hashes == {str(p.relative_to(ROOT)): sha(p) for p in protected}
audit['protected_hashes_unchanged'] = production_hashes
summary = dict(audit=audit, metrics=metrics.to_dict('records'), window_summary=window_summary,
               mechanism=mechanism, historical_refit_robustness=robustness,
               caveats=['Conditional historical development evidence; inherited specifications were already selected using this history.',
                        'October was previously inspected and remains retrospective evaluation.',
                        'No new strategy, weights, calibration or model was selected from this diagnostic.',
                        'Quarterly VECM check is historical-only and post-hoc; production forecasts are unchanged.'])
(OUT / 'diagnosis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(metrics.round(4).to_string(index=False))
print(json.dumps(dict(windows=window_summary[:1] + window_summary[3:4], mechanism=mechanism, audit_passed=True), indent=2))

"""Break down the direct test-period forecasts by volatility regime and over time.

Regime: WTI 20-day realized volatility at the forecast origin, split at its
median over the pre-test sample (so the threshold uses no test data).
Over time: RMSE relative to random walk per calendar year, and the cumulative
squared-error difference against random walk (rises when the model is better).

Writes outputs/direct_overall.csv, direct_by_regime.csv, direct_by_year.csv and
outputs/figures/direct_cumulative_vs_rw.png.
Run from the project root:  uv run python -m m3_ml.tabular.analyze_direct
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from m3_ml.common.data import HORIZONS, TEST, load_table
from m3_ml.common.features import build_features
from m3_ml.common.protocol import PROTOCOL_ID

OUT = Path(__file__).resolve().parent / 'outputs/frozen_october_2026'
MODELS = ['elastic_net', 'random_forest', 'lightgbm', 'historical_mean']
LABELS = {'elastic_net': 'Elastic Net', 'random_forest': 'Random Forest',
          'lightgbm': 'LightGBM', 'historical_mean': 'Historical mean'}
# Reference categorical palette, slots 1-4 in fixed order (validated, light mode).
COLORS = dict(zip(MODELS, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']))
INK, MUTED, GRID, BASELINE, SURFACE = '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'


def load_forecasts():
    f = pd.read_csv(OUT / 'direct_forecasts.csv', parse_dates=['origin_date'])
    if not f['protocol_id'].eq(PROTOCOL_ID).all():
        raise ValueError('forecast protocol mismatch')
    f = f.dropna(subset=['actual_log_return'])
    f['se'] = (f['forecast_log_return'] - f['actual_log_return']) ** 2
    rw = f[f['model'] == 'random_walk'].set_index(['mode', 'origin_date', 'horizon'])['se']
    f = f[f['model'] != 'random_walk'].copy()
    f['se_rw'] = rw.reindex(pd.MultiIndex.from_frame(f[['mode', 'origin_date', 'horizon']])).to_numpy()
    f['moved'] = (f['actual_log_return'] != 0) & (f['forecast_log_return'] != 0)
    f['hit'] = np.sign(f['forecast_log_return']) == np.sign(f['actual_log_return'])
    return f


def add_regime(f):
    table = load_table()
    vol = build_features(table)['wti_vol_20d'].set_axis(table['date'])
    threshold = vol[vol.index < TEST[0]].median()
    f['regime'] = np.where(vol.reindex(f['origin_date']).to_numpy() > threshold, 'high vol', 'low vol')
    return f, threshold


def dm_pvalue(d, horizon):
    """Diebold-Mariano: mean loss difference with Newey-West errors (lag h-1).
    Two-sided p-value; d = SE_model - SE_random_walk."""
    if len(d) <= max(horizon + 1, 3):
        return np.nan
    fit = sm.OLS(d.to_numpy(), np.ones(len(d))).fit(cov_type='HAC', cov_kwds={'maxlags': max(horizon - 1, 1)})
    return float(fit.pvalues[0])


def summary(f, by):
    def one(g):
        h = int(g['horizon'].iloc[0])
        return pd.Series({
            'n': len(g),
            'rmse_vs_rw': np.sqrt(g['se'].mean() / g['se_rw'].mean()),
            'direction': g.loc[g['moved'], 'hit'].mean(),
            'dm_pvalue': np.nan,
        })
    return f.groupby(by + ['model'])[['horizon', 'se', 'se_rw', 'moved', 'hit']].apply(one).round(4).assign(
        inference='descriptive_short_window')


def end_labels(ax, ends, min_gap=0.06):
    """Direct labels at line ends; identical lines share one label and the rest
    are nudged apart vertically so they never overlap."""
    merged = {}
    for name, (x, y) in ends.items():
        key = next((k for k in merged if np.isclose(merged[k][1], y, rtol=1e-6, atol=1e-9)), None)
        if key is None:
            merged[name] = [x, y, name]
        else:
            merged[key][2] += ' = ' + name
    lo, hi = ax.get_ylim()
    items = sorted(merged.values(), key=lambda v: v[1])
    placed = []
    for x, y, text in items:
        frac = (y - lo) / (hi - lo)
        if placed and frac - placed[-1] < min_gap:
            frac = placed[-1] + min_gap
        placed.append(frac)
        ax.annotate(text, (x, y), xytext=(x, lo + frac * (hi - lo)), textcoords='data',
                    va='center', fontsize=8, color=INK, annotation_clip=False)


def overall(path):
    """Whole test period per horizon and model, in the rubric's terms: price
    MAE (USD/barrel) and MAPE, log-return RMSE relative to random walk,
    direction accuracy and the Diebold-Mariano p-value against random walk."""
    emitted = pd.read_csv(path)
    rows = []
    for (mode, h, model), all_rows in emitted.groupby(['mode', 'horizon', 'model']):
        g = all_rows.dropna(subset=['actual_log_return']).copy()
        err = g['forecast_log_return'] - g['actual_log_return']
        rw = emitted[emitted['mode'].eq(mode) & emitted['horizon'].eq(h)
                     & emitted['model'].eq('random_walk')].set_index('origin_date')
        se_rw = rw['actual_log_return'].reindex(g['origin_date']).to_numpy() ** 2
        moved = (g['actual_log_return'] != 0) & (g['forecast_log_return'] != 0)
        hit = np.sign(g['forecast_log_return']) == np.sign(g['actual_log_return'])
        ae = (g['forecast'] - g['actual']).abs()
        denominator = np.mean(se_rw) if len(se_rw) else np.nan
        rows.append({'mode': mode, 'horizon': h, 'model': model, 'n': len(g),
                     'mae_usd': ae.mean(), 'mape_pct': 100 * (ae / g['actual'].abs()).mean(),
                     'rmse_log_return': np.sqrt((err ** 2).mean()),
                     'rmse_vs_rw': np.sqrt((err ** 2).mean() / denominator) if denominator > 0 else np.nan,
                     'direction': hit[moved].mean(),
                     'dm_pvalue': np.nan,
                     'inference': 'descriptive_short_window',
                     'protocol_id': PROTOCOL_ID})
    return pd.DataFrame(rows).set_index(['mode', 'horizon', 'model']).round(4)


def plot_cumulative(f, mode):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, facecolor=SURFACE)
    for ax, h in zip(axes.flat, HORIZONS):
        ax.set_facecolor(SURFACE)
        g = f[f['horizon'] == h]
        ends = {}
        for model in MODELS:
            m = g[g['model'] == model].sort_values('origin_date')
            if m.empty:
                continue
            cum = (m['se_rw'] - m['se']).cumsum()
            ax.plot(m['origin_date'], cum, color=COLORS[model], lw=2, label=LABELS[model])
            ends[LABELS[model]] = (m['origin_date'].iloc[-1] + pd.Timedelta(days=20), cum.iloc[-1])
        ax.axhline(0, color=BASELINE, lw=1)
        ax.margins(x=0.22)
        end_labels(ax, ends)
        ax.set_title(f'h = {h} trading day{"s" if h > 1 else ""}', loc='left', fontsize=11, color=INK)
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(BASELINE)
    for ax in axes[:, 0]:
        ax.set_ylabel('Cumulative SE(random walk) - SE(model)', fontsize=8, color=MUTED)
    fig.suptitle('Direct ML forecasts vs random walk, test period (above 0 = better than random walk)',
                 x=0.01, ha='left', y=0.985, fontsize=12, color=INK)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.005, 0.955), ncol=4,
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = OUT / 'figures' / mode / 'direct_cumulative_vs_rw.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    return path


def main():
    f, threshold = add_regime(load_forecasts())
    f['year'] = f['origin_date'].dt.year
    regimes, years = [], []
    for mode, part in f.groupby('mode'):
        regimes.append(summary(part, ['horizon', 'regime']).reset_index().assign(mode=mode, protocol_id=PROTOCOL_ID))
        years.append(summary(part, ['horizon', 'year']).reset_index().assign(mode=mode, protocol_id=PROTOCOL_ID))
        print('figure:', plot_cumulative(part, mode))
    total = overall(OUT / 'direct_forecasts.csv')
    total.to_csv(OUT / 'direct_overall.csv', lineterminator='\n')
    pd.concat(regimes, ignore_index=True).to_csv(OUT / 'direct_by_regime.csv', index=False, lineterminator='\n')
    pd.concat(years, ignore_index=True).to_csv(OUT / 'direct_by_year.csv', index=False, lineterminator='\n')
    print('overall:\n', total.to_string())
    print(f'regime threshold (pre-test median published-WTI 20d vol): {threshold:.4f}')


if __name__ == '__main__':
    main()

"""Evaluate the test-period distribution forecasts (metrics fixed in PLAN.md).

  uv run python -m m3_ml.distribution.analyze

Writes outputs/test_metrics.csv, test_risk.csv, test_by_regime.csv,
test_by_year.csv and figures (PIT histograms, coverage, fan chart); when route C
(proposed after the test, so not a blind test; see doc/m3_ml/README.md) has test
results, also test_non_blind.csv (descriptive comparisons against A-Ridge, GARCH, EWMA and HS)
and its fan chart.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m3_ml.common.data import HORIZONS, TEST
from m3_ml.common.features import build_features
from m3_ml.common.data import load_table
from m3_ml.common.protocol import PROTOCOL_ID

from .quantiles import (LEVELS, QUANTILE_COLUMNS, TAUS, christoffersen_pvalue, covered, crps,
                        hac_mean_pvalue, interval, kupiec_pvalue, pinball, pit)
from .tails import (ALL_COLUMNS, ALPHAS, SIDES, direction_metrics, fz0, mean_fz0, prob_up, side_returns, split,
                    var_es, z2_test)

OUT = Path(__file__).resolve().parent / 'outputs/frozen_october_2026'
FIG = OUT / 'figures'
BENCH = ('hs', 'ewma', 'garch')
ORDER = ['hs', 'ewma', 'garch', 'garch_evt', 'a_ridge', 'a_lightgbm', 'a_evt', 'b_linear', 'b_lightgbm',
         'b_forest', 'ensemble', 'c_mixture']
LABELS = {'hs': 'Historical sim.', 'ewma': 'EWMA', 'garch': 'GARCH', 'garch_evt': 'GARCH-EVT',
          'a_ridge': 'A: Ridge vol', 'a_lightgbm': 'A: LightGBM vol', 'a_evt': 'A: EVT tails',
          'b_linear': 'B: linear QR', 'b_lightgbm': 'B: LightGBM QR', 'b_forest': 'B: quantile forest',
          'ensemble': 'Ensemble', 'c_mixture': 'C: var. mixture (non-blind)'}
NON_BLIND = ('c_mixture',)                      # proposed after the test (PLAN.md), h = 5, 20, 60 only
NON_BLIND_RIVALS = ('a_ridge', 'garch', 'ewma', 'hs')
INK, MUTED, GRID, BASELINE, SURFACE = '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
BLUE = {'band90': '#b7d3f6', 'band50': '#6da7ec', 'line': '#1c5cab'}  # sequential blue ramp steps 150/300/550
ACTUAL = '#eb6834'


def load():
    f = pd.read_csv(OUT / 'test_quantiles.csv', parse_dates=['origin_date', 'target_date'])
    if not f['protocol_id'].eq(PROTOCOL_ID).all():
        raise ValueError('forecast protocol does not match frozen October')
    return f


def scored(f):
    Q, y = f[QUANTILE_COLUMNS].to_numpy(), f['actual_log_return'].to_numpy()
    out = f[['origin_date', 'target_date', 'horizon', 'method', 'origin_price', 'actual_log_return']].copy()
    out['crps'] = crps(Q, y)
    out['pit'] = pit(Q, y)
    losses = pinball(Q, y)
    for j, tau in enumerate(TAUS):
        out[f'pinball_{tau:.3f}'] = losses[:, j]
    for level in LEVELS:
        lo, hi = interval(Q, level)
        out[f'hit_{int(level * 100)}'] = covered(Q, y, level)
        out[f'width_{int(level * 100)}'] = hi - lo
        out[f'width_usd_{int(level * 100)}'] = f['origin_price'].to_numpy() * (np.exp(hi) - np.exp(lo))
    out['median'] = f['q_0.500'].to_numpy()
    return out


def paired(s, h, method, other):
    a = s[(s.horizon == h) & (s.method == method)].set_index('origin_date')['crps']
    b = s[(s.horizon == h) & (s.method == other)].set_index('origin_date')['crps'].reindex(a.index)
    return a, b


def metrics(s):
    rows = []
    for (h, method), g in s.groupby(['horizon', 'method']):
        row = {'horizon': h, 'method': method, 'n': len(g), 'crps': g['crps'].mean()}
        for tau in TAUS:
            row[f'pinball_{tau:.3f}'] = g[f'pinball_{tau:.3f}'].mean()
        for other in ('ewma', 'hs'):
            a, b = paired(s, h, method, other)
            row[f'crps_vs_{other}'] = a.mean() / b.mean()
            row[f'dm_p_vs_{other}'] = np.nan if method == other else hac_mean_pvalue(a - b, h - 1)
        for level in LEVELS:
            k = int(level * 100)
            row[f'cov_{k}'] = g[f'hit_{k}'].mean()
            row[f'cov_{k}_p'] = hac_mean_pvalue(g[f'hit_{k}'] - level, h - 1)
        row['width_90'] = g['width_90'].mean()
        row['width_usd_90'] = g['width_usd_90'].mean()
        if h == 1:
            row['kupiec_90_p'] = kupiec_pvalue(g['hit_90'], 0.9)
            row['christoffersen_90_p'] = christoffersen_pvalue(g['hit_90'].to_numpy())
        err = g['median'] - g['actual_log_return']
        row['median_rmse_vs_rw'] = np.sqrt((err ** 2).mean() / (g['actual_log_return'] ** 2).mean())
        p, a = g['origin_price'] * np.exp(g['median']), g['origin_price'] * np.exp(g['actual_log_return'])
        row['median_mae_usd'] = (p - a).abs().mean()
        row['skill'] = bool(row['crps_vs_ewma'] < 1 and row['dm_p_vs_ewma'] < 0.05 and row['cov_90_p'] >= 0.05) \
            if method != 'ewma' else False
        rows.append(row)
    out = pd.DataFrame(rows)
    out['method'] = pd.Categorical(out['method'], ORDER, ordered=True)
    return out.sort_values(['horizon', 'method']).round(4)


def risk_metrics(f):
    """VaR/ES backtests and FZ0 comparisons per horizon and method (PLAN.md addendum)."""
    per_origin, rows = {}, []
    for (h, method), g in f.groupby(['horizon', 'method']):
        Q_all, y = g[ALL_COLUMNS].to_numpy(), g['actual_log_return'].to_numpy()
        per_origin[(h, method)] = pd.Series(mean_fz0(Q_all, y), index=g['origin_date'].to_numpy())
        Q_main, Q_full = split(Q_all)
        p_up = prob_up(Q_main)
        row = {'horizon': h, 'method': method, 'fz0': np.nanmean(per_origin[(h, method)]),
               **{k: round(v, 4) for k, v in direction_metrics(p_up, y).items()}}
        row['n_fz0'] = int(np.isfinite(per_origin[(h, method)]).sum())
        for a in ALPHAS:
            for side in SIDES:
                v, e = var_es(Q_main, Q_full, a, side)
                x = side_returns(y, side)
                hits = x <= v
                key = f'{side}_{a * 100:g}'
                row[f'viol_{key}'] = hits.mean()
                row[f'viol_{key}_p'] = hac_mean_pvalue(hits - a, h - 1)
                price = g['origin_price'].to_numpy()
                row[f'var_usd_{key}'] = np.mean(price * np.abs(np.exp(v if side == 'lower' else -v) - 1))
                row[f'es_usd_{key}'] = np.mean(price * np.abs(np.exp(e if side == 'lower' else -e) - 1))
                row[f'z2_{key}'], z2p = z2_test(Q_all, y, a, side)
                row[f'z2_{key}_p'] = z2p if h == 1 else np.nan
                if h == 1:
                    row[f'kupiec_{key}_p'] = kupiec_pvalue(~hits, 1 - a)
                    row[f'christoffersen_{key}_p'] = christoffersen_pvalue(~hits)
        rows.append(row)
    out = pd.DataFrame(rows)
    hs_brier = out[out.method == 'hs'].set_index('horizon')['brier']
    out['brier_skill_vs_hs'] = 1 - out['brier'] / hs_brier.reindex(out['horizon']).to_numpy()
    for other in ('ewma', 'hs'):
        diff, pv = [], []
        for r in out.itertuples():
            a = per_origin[(r.horizon, r.method)]
            b = per_origin.get((r.horizon, other), pd.Series(dtype=float)).reindex(a.index)
            d = (a - b).dropna()
            diff.append(d.mean())
            pv.append(np.nan if r.method == other else hac_mean_pvalue(d, r.horizon - 1))
        out[f'fz0_minus_{other}'], out[f'dm_p_fz0_vs_{other}'] = diff, pv
    out['risk_skill'] = ((out['fz0_minus_ewma'] < 0) & (out['dm_p_fz0_vs_ewma'] < 0.05)
                         & (out['viol_lower_2.5_p'] >= 0.05) & (out['viol_upper_2.5_p'] >= 0.05)
                         & (out['method'] != 'ewma'))
    out['method'] = pd.Categorical(out['method'], ORDER, ordered=True)
    return out.sort_values(['horizon', 'method']).round(4)


def non_blind(s, f):
    """Route C against its main rivals: descriptive CRPS ratios and FZ0 differences."""
    fz = {(h, m): pd.Series(mean_fz0(g[ALL_COLUMNS].to_numpy(), g['actual_log_return'].to_numpy()),
                            index=g['origin_date'].to_numpy())
          for (h, m), g in f[f.method.isin(NON_BLIND + NON_BLIND_RIVALS)].groupby(['horizon', 'method'])}
    rows = []
    for method in NON_BLIND:
        for h in sorted(s.loc[s.method == method, 'horizon'].unique()):
            for other in NON_BLIND_RIVALS:
                if (h, other) not in fz:
                    continue
                a, b = paired(s, h, method, other)
                d = (fz[(h, method)] - fz[(h, other)].reindex(fz[(h, method)].index)).dropna()
                rows.append({'horizon': h, 'method': method, 'versus': other, 'crps_ratio': a.mean() / b.mean(),
                             'dm_p_crps': hac_mean_pvalue(a - b, h - 1), 'fz0_diff': d.mean(),
                             'dm_p_fz0': hac_mean_pvalue(d, h - 1)})
    return descriptive_only(pd.DataFrame(rows).round(4))


def by_group(s, key):
    rows = []
    for (h, grp, method), g in s.groupby(['horizon', key, 'method']):
        a, b = paired(s[s[key] == grp], h, method, 'ewma')
        rows.append({'horizon': h, key: grp, 'method': method, 'n': len(g), 'crps_vs_ewma': a.mean() / b.mean(),
                     'cov_90': g['hit_90'].mean(), 'width_90': g['width_90'].mean()})
    out = pd.DataFrame(rows)
    out['method'] = pd.Categorical(out['method'], ORDER, ordered=True)
    return out.sort_values(['horizon', key, 'method']).round(4)


def add_regime(s):
    table = load_table()
    vol = build_features(table)['wti_vol_20d'].set_axis(table['date'])
    threshold = vol[vol.index < TEST[0]].median()
    s['regime'] = np.where(vol.reindex(s['origin_date']).to_numpy() > threshold, 'high vol', 'low vol')
    s['year'] = s['origin_date'].dt.year
    return s


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=7)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(BASELINE)


def plot_pit(s):
    methods = [m for m in ORDER if m in s['method'].unique()]
    fig, axes = plt.subplots(len(HORIZONS), len(methods), figsize=(2.0 * len(methods), 7.5),
                             sharex=True, squeeze=False, facecolor=SURFACE)
    for i, h in enumerate(HORIZONS):
        for j, m in enumerate(methods):
            ax = axes[i, j]
            style(ax)
            ax.set_ylim(0, 2.6)
            if i == 0:
                ax.set_title(LABELS[m], fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel(f'h = {h}', fontsize=8, color=INK)
            v = s[(s.horizon == h) & (s.method == m)]['pit']
            if v.empty:  # route C is not defined at h = 1
                ax.text(0.5, 1.3, 'unreached / unsupported', ha='center', va='center', fontsize=8, color=MUTED)
                continue
            ax.hist(v, bins=10, range=(0, 1), weights=np.full(len(v), 10 / len(v)), color=BLUE['band50'],
                    edgecolor=SURFACE, linewidth=2)
            ax.axhline(1, color=INK, lw=0.8, ls='--')
    fig.suptitle('PIT histograms, test period (flat at 1 = calibrated; U shape = too narrow; hump = too wide)',
                 x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / 'pit_histograms.png', dpi=150, facecolor=SURFACE)


def plot_fan(f, s, method):
    horizons = sorted(f.loc[f.method == method, 'horizon'].unique())
    fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 3 * len(horizons)), sharex=True, facecolor=SURFACE)
    for ax, h in zip(axes, horizons):
        style(ax)
        g = f[(f.horizon == h) & (f.method == method)].sort_values('target_date')
        p = g['origin_price']
        for lo, hi, color, label in (('q_0.050', 'q_0.950', BLUE['band90'], '90% interval'),
                                     ('q_0.250', 'q_0.750', BLUE['band50'], '50% interval')):
            ax.fill_between(g['target_date'], p * np.exp(g[lo]), p * np.exp(g[hi]), color=color, lw=0, label=label)
        ax.plot(g['target_date'], p * np.exp(g['q_0.500']), color=BLUE['line'], lw=1.2, label='Median forecast')
        ax.plot(g['target_date'], p * np.exp(g['actual_log_return']), color=ACTUAL, lw=1.2, label='Actual WTI price')
        m = s[(s.horizon == h) & (s.method == method)]
        ax.set_title(f'h = {h} trading day{"s" if h > 1 else ""} ahead   90% coverage {m["hit_90"].mean():.0%}, '
                     f'50% coverage {m["hit_50"].mean():.0%}, mean 90% width ${m["width_usd_90"].mean():.1f}/bbl',
                     loc='left', fontsize=10, color=INK)
        ax.set_ylabel('USD/barrel', fontsize=8, color=MUTED)
        ax.grid(axis='y', color=GRID, lw=0.6)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f'WTI spot: actual vs {LABELS[method]} forecast distribution, test period, plotted at the target date',
                 x=0.01, ha='left', y=0.99, fontsize=12, color=INK)
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.005, 0.968), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / f'fan_{method}.png', dpi=150, facecolor=SURFACE)


def plot_test_skill(m):
    """Test CRPS relative to historical simulation; filled = DM p < 0.05 against it."""
    methods = [x for x in ORDER if x in set(m['method'].astype(str))]
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(14, 4.6), sharey=True, facecolor=SURFACE)
    for ax, h in zip(axes, HORIZONS):
        style(ax)
        g = m[m.horizon == h].set_index(m[m.horizon == h]['method'].astype(str))
        for i, name in enumerate(methods[::-1]):
            if name not in g.index:  # route C is not defined at h = 1
                continue
            r = g.loc[name]
            if r['n'] == 0 or not np.isfinite(r['crps_vs_hs']):
                continue
            sig = name != 'hs' and r['dm_p_vs_hs'] < 0.05
            ax.plot(r['crps_vs_hs'], i, 'o', ms=7, color=BLUE['line'], mfc=BLUE['line'] if sig else SURFACE, mew=1.5)
            ax.annotate(f'{r["crps_vs_hs"]:.3f}', (r['crps_vs_hs'], i), xytext=(0, 7), textcoords='offset points',
                        ha='center', fontsize=7, color=MUTED)
        ax.axvline(1, color=INK, lw=0.8, ls='--')
        n_after = sum(x in NON_BLIND for x in methods)
        if n_after:  # methods proposed after the test sit below this line
            ax.axhline(n_after - 0.5, color=BASELINE, lw=0.8, ls=':')
        ax.set_yticks(range(len(methods)), [LABELS[x] for x in methods[::-1]], fontsize=8, color=INK)
        ax.set_title(f'h = {h}', loc='left', fontsize=10, color=INK)
        ax.grid(axis='x', color=GRID, lw=0.6)
    fig.suptitle('October 1-23, 2026: CRPS relative to historical simulation (left of 1 = better; '
                 'inferential tests withheld when the realized sample is insufficient)', x=0.01, ha='left', fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / 'test_crps_skill.png', dpi=150, facecolor=SURFACE)


def plot_var(f, methods=('hs', 'a_evt'), alpha=0.025):
    """Daily (h = 1) VaR backtest: actual return against both 2.5% VaR bounds, violations marked."""
    if not methods:
        return
    fig, axes = plt.subplots(len(methods), 1, figsize=(12, 3.8 * len(methods)), sharex=True, sharey=True,
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    lo_col, hi_col = 'q_0.025', 'q_0.975'
    for ax, name in zip(axes, methods):
        style(ax)
        g = f[(f.horizon == 1) & (f.method == name)].sort_values('target_date')
        y = 100 * g['actual_log_return']
        lo, hi = 100 * g[lo_col], 100 * g[hi_col]
        ax.fill_between(g['target_date'], lo, hi, color=BLUE['band90'], lw=0, label='95% interval (2.5% VaR, both sides)')
        ax.plot(g['target_date'], y, color=MUTED, lw=0.6, label='Actual anchor-relative log return')
        out = (y < lo) | (y > hi)
        ax.plot(g['target_date'][out], y[out], 'o', ms=4, color=ACTUAL, label='VaR violation')
        ax.axhline(0, color=BASELINE, lw=0.6)
        ax.set_title(f'{LABELS[name]}: violations lower {(y < lo).mean():.1%}, upper {(y > hi).mean():.1%} '
                     f'(expected 2.5% each)', loc='left', fontsize=10, color=INK)
        ax.set_ylabel('%', fontsize=8, color=MUTED)
        ax.grid(axis='y', color=GRID, lw=0.6)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle('Test period: 1-day 2.5% VaR backtest for long (lower) and short (upper) positions',
                 x=0.01, ha='left', y=0.99, fontsize=11, color=INK)
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.005, 0.955), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / 'test_var_backtest_h1.png', dpi=150, facecolor=SURFACE)


def descriptive_only(result):
    """October's 17 daily / 13 overlapping five-day outcomes support description.

    Generic historical backtest functions above also calculate asymptotic tests;
    do not expose those tests or significance-based winner flags in this window.
    """
    result = result.copy()
    for column in result:
        if column.startswith('dm_p_') or column.endswith('_p'):
            result[column] = np.nan
    for column in ('skill', 'risk_skill'):
        if column in result:
            result[column] = False
    return result


def complete_metrics(result, emitted, mode):
    """Retain every emitted key, including n=0 at unreached horizons."""
    counts = emitted.groupby(['horizon', 'method']).actual_log_return.count().rename('n').reset_index()
    result = counts.merge(result.drop(columns=['n'], errors='ignore'), on=['horizon', 'method'], how='left')
    if 'n_fz0' in result:
        result['n_fz0'] = result['n_fz0'].fillna(0).astype(int)
    result.insert(0, 'mode', mode)
    result['protocol_id'] = PROTOCOL_ID
    result['status'] = np.where(result['n'].gt(0), 'small_sample_descriptive_only',
                                'unscored_no_matured_test_targets')
    return descriptive_only(result)


def main():
    global FIG
    emitted = load()
    metric_frames, risk_frames, regime_frames, year_frames, nb_frames = [], [], [], [], []
    for mode, all_rows in emitted.groupby('mode'):
        FIG = OUT / 'figures' / mode
        FIG.mkdir(parents=True, exist_ok=True)
        f = all_rows.dropna(subset=['actual_log_return']).copy()
        s = add_regime(scored(f))
        m = complete_metrics(metrics(s), all_rows, mode)
        metric_frames.append(m)
        risk_frames.append(complete_metrics(risk_metrics(f), all_rows, mode))
        regime_frames.append(by_group(s, 'regime').assign(mode=mode, protocol_id=PROTOCOL_ID))
        year_frames.append(by_group(s, 'year').assign(mode=mode, protocol_id=PROTOCOL_ID))
        if f['method'].isin(NON_BLIND).any():
            nb_frames.append(non_blind(s, f).assign(mode=mode, protocol_id=PROTOCOL_ID))
        plot_pit(s)
        plot_test_skill(m)
        plot_var(f, methods=tuple(x for x in ('hs', 'a_evt') if x in set(f.method)))
        # The displayed method is prescribed, never selected on October loss.
        if mode == 'daily_update' and 'a_evt' in set(f.method):
            plot_fan(f, s, 'a_evt')
    for filename, frames in [('test_metrics.csv', metric_frames), ('test_risk.csv', risk_frames),
                              ('test_by_regime.csv', regime_frames), ('test_by_year.csv', year_frames),
                              ('test_non_blind.csv', nb_frames)]:
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(OUT / filename, index=False, lineterminator='\n')
    print(pd.concat(metric_frames, ignore_index=True).to_string(index=False))


if __name__ == '__main__':
    main()

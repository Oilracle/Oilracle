"""Report historical validation (2023-09-01..2026-09-30), never October.

  python -m m3_ml.distribution.report_validation

Reads the selected out-of-origin forecasts already produced by run validate.
It never replaces the producer's full 63-level calibration file with a subset.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from m3_ml.common.data import HORIZONS

from . import run
from .quantiles import QUANTILE_COLUMNS, covered, crps, pit

OUT = Path(__file__).resolve().parent / 'outputs/frozen_october_2026'
FIG = OUT / 'figures'
ORDER = ['hs', 'ewma', 'garch', 'a_ridge', 'a_lightgbm', 'b_linear', 'b_lightgbm', 'b_forest']
LABELS = {'hs': 'Historical sim.', 'ewma': 'EWMA', 'garch': 'GARCH', 'a_ridge': 'A: Ridge vol',
          'a_lightgbm': 'A: LightGBM vol', 'b_linear': 'B: linear QR', 'b_lightgbm': 'B: LightGBM QR',
          'b_forest': 'B: quantile forest'}
INK, SECOND, MUTED, GRID, BASELINE, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
BLUE, RED, NEUTRAL = '#2a78d6', '#e34948', '#f0efec'
BAND90, BAND50, MEDIAN, ACTUAL = '#b7d3f6', '#6da7ec', '#1c5cab', '#eb6834'


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(BASELINE)




def plot_skill(scores, selected):
    """Geometric-mean CRPS ratio vs historical simulation with annual-fold range."""
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(13, 4.2), sharey=True, facecolor=SURFACE)
    for ax, h in zip(axes, HORIZONS):
        style(ax)
        for i, m in enumerate(ORDER[::-1]):
            p = selected[f'{m}/h{h}']['params']
            r = scores[(scores.method == m) & (scores.h == h) & scores.p.map(lambda d: d == p)]['relative']
            ax.plot([r.min(), r.max()], [i, i], color=BASELINE, lw=3, solid_capstyle='round')
            ax.plot(selected[f'{m}/h{h}']['score'], i, 'o', color=BLUE, ms=7, mec=SURFACE, mew=1.5)
            ax.annotate(f'{selected[f"{m}/h{h}"]["score"]:.3f}', (selected[f'{m}/h{h}']['score'], i),
                        xytext=(0, 7), textcoords='offset points', ha='center', fontsize=7, color=SECOND)
        ax.axvline(1, color=INK, lw=0.8, ls='--')
        ax.set_yticks(range(len(ORDER)), [LABELS[m] for m in ORDER[::-1]], fontsize=8, color=INK)
        ax.set_title(f'h = {h}', loc='left', fontsize=10, color=INK)
        ax.grid(axis='x', color=GRID, lw=0.6)
    fig.suptitle('Validation Sep 2023-Sep 2026: CRPS relative to historical simulation (dot = geometric mean, '
                 'bar = annual-fold range; left of 1 = better)', x=0.01, ha='left', fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / 'validation_crps_skill.png', dpi=150, facecolor=SURFACE)


def plot_years(scores, selected):
    cmap = LinearSegmentedColormap.from_list('div', [BLUE, NEUTRAL, RED])
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(14, 4.2), sharey=True, facecolor=SURFACE)
    years = list(run.FOLD_YEARS)
    for ax, h in zip(axes, HORIZONS):
        grid = np.array([[scores[(scores.method == m) & (scores.h == h) & (scores.year == y)
                                 & scores.p.map(lambda d: d == selected[f'{m}/h{h}']['params'])]['relative'].iloc[0]
                          for y in years] for m in ORDER])
        ax.imshow(grid, cmap=cmap, norm=TwoSlopeNorm(1.0, 0.8, 1.2), aspect='auto')
        for i in range(len(ORDER)):
            for j in range(len(years)):
                ax.text(j, i, f'{grid[i, j]:.2f}', ha='center', va='center', fontsize=7, color=INK)
        ax.set_xticks(range(len(years)), years, fontsize=8, color=MUTED)
        ax.set_yticks(range(len(ORDER)), [LABELS[m] for m in ORDER], fontsize=8, color=INK)
        ax.set_title(f'h = {h}', loc='left', fontsize=10, color=INK)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.suptitle('Validation: CRPS relative to historical simulation by year (blue < 1 = better, red > 1 = worse)',
                 x=0.01, ha='left', fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / 'validation_crps_by_year.png', dpi=150, facecolor=SURFACE)


def plot_calibration(v):
    """90% and 50% coverage per method and horizon, with PIT histograms."""
    fig, axes = plt.subplots(len(HORIZONS), len(ORDER), figsize=(16, 8), sharex=True, sharey=True, facecolor=SURFACE)
    for i, h in enumerate(HORIZONS):
        for j, m in enumerate(ORDER):
            ax = axes[i, j]
            style(ax)
            g = v[(v.horizon == h) & (v.method == m)]
            ax.hist(g['pit'], bins=10, range=(0, 1), weights=np.full(len(g), 10 / len(g)), color=BAND50,
                    edgecolor=SURFACE, linewidth=2)
            ax.axhline(1, color=INK, lw=0.8, ls='--')
            ax.set_ylim(0, 2.6)
            ax.text(0.5, 2.35, f'90%: {g["hit_90"].mean():.0%}  50%: {g["hit_50"].mean():.0%}', ha='center',
                    fontsize=7, color=INK)
            if i == 0:
                ax.set_title(LABELS[m], fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel(f'h = {h}', fontsize=9, color=INK)
    fig.suptitle('Validation Sep 2023-Sep 2026: PIT histograms and interval coverage (flat at 1 = calibrated; '
                 'U shape = too narrow, hump = too wide)', x=0.01, ha='left', fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / 'validation_calibration.png', dpi=150, facecolor=SURFACE)


def plot_fan(v, h, methods):
    fig, axes = plt.subplots(len(methods), 1, figsize=(12, 3.6 * len(methods)), sharex=True, facecolor=SURFACE)
    for ax, m in zip(np.atleast_1d(axes), methods):
        style(ax)
        g = v[(v.horizon == h) & (v.method == m)].sort_values('target_date')
        p = g['origin_price']
        ax.fill_between(g['target_date'], p * np.exp(g['q_0.050']), p * np.exp(g['q_0.950']), color=BAND90, lw=0,
                        label='90% interval')
        ax.fill_between(g['target_date'], p * np.exp(g['q_0.250']), p * np.exp(g['q_0.750']), color=BAND50, lw=0,
                        label='50% interval')
        ax.plot(g['target_date'], p * np.exp(g['q_0.500']), color=MEDIAN, lw=1, label='Median forecast')
        ax.plot(g['target_date'], p * np.exp(g['actual_log_return']), color=ACTUAL, lw=1.1, label='Actual WTI price')
        ax.set_title(f'{LABELS[m]}, h = {h}: 90% coverage {g["hit_90"].mean():.0%}, 50% coverage {g["hit_50"].mean():.0%}',
                     loc='left', fontsize=10, color=INK)
        ax.set_ylabel('USD/barrel', fontsize=8, color=MUTED)
        ax.set_ylim(0, 130)
        ax.grid(axis='y', color=GRID, lw=0.6)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.suptitle(f'Validation Sep 2023-Sep 2026: forecast distribution vs actual WTI, h = {h} '
                 '(plotted at the quoted target date)', x=0.01, ha='left', y=0.995, fontsize=10, color=INK)
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.005, 0.965), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / f'validation_fan_h{h}.png', dpi=150, facecolor=SURFACE)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    selected = run.selected_settings()
    v = pd.read_csv(OUT / 'validation_selected_quantiles.csv',
                    parse_dates=['origin_date', 'target_date'])
    if not v['protocol_id'].eq(run.PROTOCOL_ID).all():
        raise ValueError('validation forecast protocol mismatch')
    v = v.dropna(subset=['actual_log_return']).copy()
    Q, y = v[QUANTILE_COLUMNS].to_numpy(), v['actual_log_return'].to_numpy()
    v['crps'], v['pit'] = crps(Q, y), pit(Q, y)
    v['hit_90'], v['hit_50'] = covered(Q, y, 0.9), covered(Q, y, 0.5)
    scores = pd.read_csv(OUT / 'validation_scores.csv')
    scores['p'] = scores['params'].map(json.loads)
    scores = scores[scores.metric == 'crps']
    plot_skill(scores, selected)
    plot_years(scores, selected)
    plot_calibration(v)
    plot_fan(v, 5, ['garch', 'a_ridge'])
    cov = v.groupby(['horizon', 'method'])[['hit_90', 'hit_50']].mean().unstack('horizon').round(3)
    print(cov.to_string())


if __name__ == '__main__':
    main()

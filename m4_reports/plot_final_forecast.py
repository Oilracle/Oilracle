"""Render the frozen v6 rolling point forecasts without fitting or selection."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'm4_reports/final_strategy_manifest.json').read_text())
    assert manifest['active_version'] == 'short_bias_v6'
    selected = ROOT / manifest['model_output'] / 'selected.json'
    digest = hashlib.sha256(selected.read_bytes()).hexdigest()
    assert digest == manifest['selected_sha256']
    source = ROOT / manifest['forecasts']
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    audit = json.loads((source.parent / 'delivery_audit.json').read_text())
    assert audit['status'] == 'passed' and audit['forecast_sha256'] == source_digest
    frame = pd.read_csv(source, parse_dates=['origin_date', 'target_date'])
    scores = pd.read_csv(ROOT / manifest['metrics']).set_index('horizon')
    assert frame['mode'].eq('daily_update').all()
    folder = source.parent / 'figures'
    folder.mkdir(exist_ok=True)

    ink, muted, blue, orange = '#182638', '#607086', '#1463D6', '#D57919'
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#CBD5E1', 'axes.labelcolor': muted,
        'xtick.color': muted, 'ytick.color': muted,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    fig.set_facecolor('white')
    fig.subplots_adjust(left=.065, right=.975, top=.70, bottom=.275, wspace=.13)
    fig.text(.065, .935, 'FINAL v6  /  OCTOBER EVALUATION', color=blue,
             fontsize=11, fontweight='bold')
    fig.text(.065, .877, 'Rolling forecasts vs random walk and observed WTI',
             fontsize=25, color=ink, fontweight='bold')
    fig.text(.065, .833, '1–23 October 2026  ·  Price in USD per barrel  ·  Model parameters frozen on 30 September',
             fontsize=12, color=muted)
    handles = [
        Line2D([0], [0], color=ink, lw=2.5, marker='o', markersize=4, label='Observed spot price'),
        Line2D([0], [0], color=blue, lw=2.5, marker='o', markersize=4, label='v6 point forecast (median)'),
        Line2D([0], [0], color=orange, lw=2.1, ls='--', label='Random walk · latest published spot'),
    ]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.06, .805),
               ncol=3, frameon=False, fontsize=11, handlelength=3, columnspacing=3)

    evidence = []
    for ax, h, n in zip(axes, (1, 5), (17, 13)):
        g = frame[frame.horizon.eq(h) & frame.actual.notna()].sort_values('target_date').copy()
        assert len(g) == n and not g.target_date.duplicated().any()
        assert g.target_date.between('2026-10-01', '2026-10-23').all()
        assert (g.origin_date < g.target_date).all()
        np.testing.assert_allclose(g.rw_price, g.origin_price)
        rmse = float(np.sqrt(np.mean((g.forecast - g.actual) ** 2)))
        rw_rmse = float(np.sqrt(np.mean((g.rw_price - g.actual) ** 2)))
        np.testing.assert_allclose([rmse, rw_rmse], scores.loc[h, ['rmse_usd', 'rw_rmse_usd']].astype(float))
        improvement = 100 * (1 - rmse / rw_rmse)
        ax.plot(g.target_date, g.rw_price, color=orange, lw=2.1, ls='--', zorder=2)
        ax.plot(g.target_date, g.forecast, color=blue, lw=2.6,
                marker='o', markersize=4.2, markeredgecolor='white', markeredgewidth=.6, zorder=3)
        ax.plot(g.target_date, g.actual, color=ink, lw=2.4,
                marker='o', markersize=3.8, markeredgecolor='white', markeredgewidth=.6, zorder=4)
        ax.set_title(f'{h}-day ahead  |  {n} realized targets', loc='left',
                     fontsize=15, fontweight='bold', color=ink, pad=17)
        ax.set_ylim(82, 114)
        ax.set_yticks(np.arange(85, 115, 5))
        tick_indices = np.unique(np.linspace(0, n - 1, 5).round().astype(int))
        ax.set_xticks(g.target_date.iloc[tick_indices])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.tick_params(length=0, pad=9)
        ax.grid(axis='y', color='#E5EBF1', lw=.8)
        ax.set_axisbelow(True)
        ax.margins(x=.025)
        ax.set_xlabel('Forecast target date', labelpad=12)
        position = ax.get_position()
        fig.text(position.x0, .175, f'RMSE    v6 {rmse:.2f}    /    RW {rw_rmse:.2f} USD',
                 fontsize=15, color=ink, fontweight='bold')
        fig.text(position.x0, .138, f'{improvement:.1f}% lower RMSE than the published-spot RW',
                 fontsize=12, color=blue)
        evidence.append(dict(horizon=h, n=n, rmse_usd=rmse, rw_rmse_usd=rw_rmse,
                             rmse_reduction_pct=improvement,
                             first_target=g.target_date.min().date().isoformat(),
                             last_target=g.target_date.max().date().isoformat()))

    axes[0].set_ylabel('WTI spot price  /  USD per barrel', labelpad=12)
    fig.text(.065, .083, 'Each point is a separately issued daily rolling forecast, aligned to its target date. Horizons count quoted days.',
             fontsize=10, color=muted)
    fig.text(.065, .055, 'RW holds the latest spot quote available at issue time under the project publication protocol. October is retrospective evaluation only.',
             fontsize=10, color=muted)
    stem = folder / 'v6_vs_random_walk_actual'
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(stem.with_suffix('.' + suffix), dpi=200, facecolor='white')
    plt.close(fig)
    provenance = dict(selected_sha256=digest, forecast_sha256=source_digest,
                      forecast_source=manifest['forecasts'], x_axis='target_date',
                      forecast_mode='daily_update', fitting_performed=False,
                      baseline='Latest published spot price under the project availability protocol',
                      point_forecasts_unchanged_from_v5=True, metrics=evidence)
    stem.with_suffix('.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'figure': str(stem.with_suffix('.png')), 'metrics': evidence}, indent=2))


if __name__ == '__main__':
    main()

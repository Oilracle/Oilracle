"""Display existing historical v6 policy forecasts; never fit backward in time."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from m3_ml.common.protocol import TRAIN_END, VALIDATION_START, VALIDATION_END
from m3_ml.fusion_v6 import research as r, run


def main():
    r.protect()
    model = r.read(r.OUT / 'selected.json')
    assert model['input_sha256'] == run.hashes()
    source = r.OUT / 'point_crossfit.csv'
    outer_start = r.read(r.OLD / 'specification.json')['outer_start']
    raw = pd.read_csv(source, parse_dates=['origin_date', 'target_date', 'label_available_date'])
    frame = raw[raw.origin_date.ge(outer_start)].copy()
    assert frame.label_available_date.le(VALIDATION_END).all()
    assert (frame.origin_date < frame.target_date).all()
    assert frame['mode'].eq('historical_crossfit').all()
    assert not frame.duplicated(r.KEY).any()
    np.testing.assert_allclose(frame.forecast, frame.v5_forecast)
    np.testing.assert_allclose(frame.rw_price, frame.origin_price)
    for column in ('family_fit_cutoff', 'meta_fit_cutoff', 'guard_fit_cutoff'):
        assert (pd.to_datetime(frame[column]) <= frame.origin_date).all()
    expected = pd.read_csv(r.OUT / 'rolling/historical_point_metrics.csv').set_index('horizon')
    metrics = []
    for h, g in frame.groupby('horizon'):
        row = dict(horizon=int(h), n=len(g),
                   first_origin=g.origin_date.min().date().isoformat(),
                   last_origin=g.origin_date.max().date().isoformat(),
                   first_target=g.target_date.min().date().isoformat(),
                   last_target=g.target_date.max().date().isoformat())
        for label, column in [('v6', 'forecast'), ('rw', 'rw_price'), ('futures', 'futures_price')]:
            row[label + '_rmse_usd'] = float(np.sqrt(np.mean((g[column] - g.actual) ** 2)))
            row[label + '_mae_usd'] = float(np.mean(np.abs(g[column] - g.actual)))
        for baseline in ('rw', 'futures'):
            row['rmse_reduction_vs_' + baseline + '_pct'] = 100 * (1 - row['v6_rmse_usd'] / row[baseline + '_rmse_usd'])
        assert row['n'] == expected.loc[h, 'n']
        np.testing.assert_allclose([row['v6_rmse_usd'], row['rw_rmse_usd'], row['futures_rmse_usd']],
                                   expected.loc[h, ['v5_rmse_usd', 'rw_rmse_usd', 'futures_rmse_usd']].astype(float))
        metrics.append(row)
    scores = pd.DataFrame(metrics).set_index('horizon')
    folder = r.OUT / 'rolling/history_review'
    folder.mkdir(exist_ok=True)
    scores.to_csv(folder / 'point_metrics.csv')
    frame.to_csv(folder / 'point_forecasts.csv', index=False)

    ink, muted, blue, orange = '#182638', '#607086', '#1463D6', '#D57919'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#CBD5E1', 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True, sharey=True)
    fig.subplots_adjust(left=.065, right=.97, top=.73, bottom=.155, hspace=.37, wspace=.12)
    fig.text(.065, .947, 'v6  /  HISTORICAL WALK-FORWARD VALIDATION', fontsize=11, color=blue, fontweight='bold')
    fig.text(.065, .895, 'Forecast performance before the final September refit',
             fontsize=24, fontweight='bold', color=ink)
    fig.text(.065, .853, 'Outer evaluation origins: September 2025 onward  ·  Labels available by 30 September 2026  ·  USD per barrel',
             fontsize=11.5, color=muted)
    handles = [Line2D([0], [0], color=ink, lw=1.8, label='Observed spot price'),
               Line2D([0], [0], color=blue, lw=1.8, label='v6 policy point forecast (v5 retained)'),
               Line2D([0], [0], color=orange, lw=1.6, ls='--', label='RW · latest published spot')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.06, .829),
               ncol=3, frameon=False, fontsize=11, handlelength=3, columnspacing=3)
    for ax, h in zip(axes.flat, (1, 5, 20, 60)):
        g = frame[frame.horizon.eq(h)].sort_values('target_date')
        row = scores.loc[h]
        ax.plot(g.target_date, g.rw_price, color=orange, ls='--', lw=1.25, alpha=.85, zorder=2)
        ax.plot(g.target_date, g.actual, color=ink, lw=1.4, alpha=.9, zorder=3)
        ax.plot(g.target_date, g.forecast, color=blue, lw=1.6, zorder=4)
        ax.set_title(f'{h}-day ahead  |  n={int(row.n)}', loc='left', color=ink,
                     fontsize=14, fontweight='bold', pad=30)
        ax.text(0, 1.035, f'RMSE  v6 {row.v6_rmse_usd:.2f} / RW {row.rw_rmse_usd:.2f}    ·    {row.rmse_reduction_vs_rw_pct:.1f}% lower',
                transform=ax.transAxes, color=blue, fontsize=10.5)
        ax.set_ylim(50, 120)
        ax.set_yticks([60, 80, 100, 120])
        ax.set_xlim(pd.Timestamp('2025-09-01'), pd.Timestamp('2026-09-25'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.tick_params(length=0, pad=7, labelbottom=True)
        ax.grid(axis='y', color='#E5EBF1', lw=.8)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel('WTI spot price / USD per barrel', labelpad=8)
    for ax in axes[-1]:
        ax.set_xlabel('Forecast target date', labelpad=8)
    fig.text(.065, .087, 'Original stage-one window: Sep 2023–Sep 2026. The fusion layer requires earlier history for training; its outer evaluation starts Sep 2025.',
             fontsize=10, color=muted)
    fig.text(.065, .061, 'Historical policies refit forward in time. These data also informed strategy development; this is not a new independent test.',
             fontsize=10, color=muted)
    fig.text(.065, .035, 'Each point is aligned to its target date. RW follows the project publication protocol. No October outcomes or final-model backcasts are used.',
             fontsize=10, color=muted)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(folder / ('historical_point_forecasts.' + extension), dpi=200, facecolor='white')
    plt.close(fig)

    risk_source = r.OUT / 'rolling/historical_risk_forecasts.csv'
    risk = pd.read_csv(risk_source, parse_dates=['origin_date', 'target_date', 'label_available_date'])
    risk = risk[risk.horizon.isin([1, 5])].copy()
    assert risk.label_available_date.le(VALIDATION_END).all()
    risk['lower90'] = risk.origin_price * np.exp(risk['q_0.050'])
    risk['upper90'] = risk.origin_price * np.exp(risk['q_0.950'])
    risk_metrics = []
    reported = pd.read_csv(r.OUT / 'rolling/historical_risk_metrics.csv').set_index('horizon')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.subplots_adjust(left=.065, right=.97, top=.71, bottom=.23, wspace=.12)
    fig.text(.065, .925, 'v6 historical 90% prediction intervals', fontsize=23, color=ink, fontweight='bold')
    fig.text(.065, .855, 'Same June–September 2026 origins  ·  Monthly calibration from previously published forecast errors', color=muted, fontsize=11)
    for ax, h in zip(axes, (1, 5)):
        g = risk[risk.horizon.eq(h)].sort_values('target_date')
        coverage = float(g.actual.between(g.lower90, g.upper90).mean())
        np.testing.assert_allclose(coverage, reported.loc[h, 'coverage90'])
        assert len(g) == reported.loc[h, 'n']
        risk_metrics.append(dict(horizon=h, n=len(g), coverage90=coverage,
                                 width90_usd=float((g.upper90 - g.lower90).mean()),
                                 first_origin=g.origin_date.min().date().isoformat(),
                                 last_origin=g.origin_date.max().date().isoformat(),
                                 first_target=g.target_date.min().date().isoformat(),
                                 last_target=g.target_date.max().date().isoformat()))
        ax.fill_between(g.target_date, g.lower90, g.upper90, color=blue, alpha=.14, label='90% interval')
        ax.plot(g.target_date, g.forecast, color=blue, lw=1.8, label='v6 median')
        ax.plot(g.target_date, g.actual, color=ink, lw=1.8, label='Observed spot')
        ax.set_title(f'{h}-day | coverage {coverage:.1%} | n={len(g)}', loc='left', fontweight='bold', color=ink, pad=15)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.grid(axis='y', alpha=.2)
        ax.set_xlabel('Forecast target date', labelpad=9)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel('WTI spot price / USD per barrel')
    fig.text(.065, .105, 'Risk calibration has a shorter evaluation window than point forecasts. The 5-day interval is 35% wider than v5; coverage still falls below 90%.',
             fontsize=10, color=muted)
    fig.text(.065, .065, 'Development evidence only: the upgrade was selected using this history. Final October parameters remain frozen and unchanged.',
             fontsize=10, color=muted)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(folder / ('historical_risk_intervals.' + extension), dpi=200, facecolor='white')
    plt.close(fig)
    pd.DataFrame(risk_metrics).to_csv(folder / 'risk_metrics.csv', index=False)

    provenance = dict(selected_sha256=r.sha(r.OUT / 'selected.json'), point_source_sha256=r.sha(source),
                      risk_source_sha256=r.sha(risk_source), train_end=TRAIN_END,
                      original_stage_one_start=VALIDATION_START, original_stage_one_end=VALIDATION_END,
                      point_outer_origin_start=outer_start, risk_outer_origin_start='2026-06-01',
                      point_metrics=metrics, risk_metrics=risk_metrics, fitting_performed=False,
                      october_used=False, evidence='Historical development/validation; not a full-window independent stage-one test.',
                      inherited_selection_caveat='Base specifications and research amendments were informed by the historical development window.',
                      final_model_backcast=False)
    (folder / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    assert model['input_sha256'] == run.hashes()
    print(scores[['n', 'v6_rmse_usd', 'rw_rmse_usd', 'v6_mae_usd', 'rmse_reduction_vs_rw_pct', 'rmse_reduction_vs_futures_pct']].to_string())
    print('Report saved:', folder)


if __name__ == '__main__':
    main()

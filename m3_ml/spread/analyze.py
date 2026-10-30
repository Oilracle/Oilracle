"""Test-period evaluation of study 4 (metrics fixed in ml/spread/PLAN.md).

  uv run python -m m3_ml.spread.analyze [--spread brent_wti|crack_321]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m3_ml.common.protocol import MODES, PROTOCOL_ID
from m3_ml.risk.analyze import read_final

OUT = Path(__file__).resolve().parent / 'outputs' / 'frozen_october_2026'
FIG = OUT / 'figures'
INK, MUTED, GRID, BASELINE, SURFACE = '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
THRESHOLD, COST = 0.25, 0.05                # USD/bbl: trade hurdle and cost per unit position change
LABELS = {'rw': 'Random walk', 'drift': 'Drift', 'ou': 'OU / AR(1)', 'ma_dev': 'MA deviation', 'enet': 'Elastic Net',
          'lgbm': 'LightGBM'}


def accuracy(f):
    rows = []
    for (mode, h, m), block in f.groupby(['mode', 'horizon', 'method']):
        g = block.dropna(subset=['actual_change', 'forecast_change'])
        err = g['forecast_change'] - g['actual_change']
        rw = g['actual_change']
        row = {'protocol_id': PROTOCOL_ID, 'mode': mode, 'horizon': h, 'method': m, 'n': len(g),
               'rmse': np.sqrt((err ** 2).mean()), 'mae': err.abs().mean(),
               'rmse_vs_rw': np.sqrt((err ** 2).mean() / (rw ** 2).mean()) if (rw ** 2).mean() > 0 else np.nan,
               'inference': 'descriptive_short_window'}
        moved = (g['actual_change'] != 0) & (g['forecast_change'] != 0)
        up_true, up_pred = (g['actual_change'] > 0)[moved], (g['forecast_change'] > 0)[moved]
        row['direction_accuracy'] = (up_true == up_pred).mean() if moved.any() else np.nan
        for name, t, p in (('up', up_true, up_pred), ('down', ~up_true, ~up_pred)):
            tp = (t & p).sum()
            prec = tp / p.sum() if p.sum() else np.nan
            rec = tp / t.sum() if t.sum() else np.nan
            row[f'precision_{name}'], row[f'recall_{name}'] = prec, rec
            row[f'f1_{name}'] = 2 * prec * rec / (prec + rec) if prec + rec else np.nan
        rows.append(row)
    return pd.DataFrame(rows).round(4)


def trading(f, h):
    """Long the spread when the forecast change exceeds the hurdle, short below minus
    the hurdle, flat otherwise; P&L in USD/bbl per unit position."""
    daily = f[(f.horizon == 1) & (f.method == 'rw')].set_index('origin_date')['actual_change']
    rows, curves = [], {}
    for m, g in f[f.horizon == h].groupby('method'):
        if m in ('rw', 'drift'):
            continue
        g = g.set_index('origin_date').sort_index()
        pos = np.sign(g['forecast_change']).where(g['forecast_change'].abs() > THRESHOLD, 0.0)
        nxt = daily.reindex(g.index)
        ok = nxt.notna()
        cost = COST * pos.diff().abs().fillna(pos.abs())
        pnl = (pos * nxt - cost)[ok]
        eq = pnl.cumsum()
        entries = ((pos != 0) & (pos != pos.shift(1))).sum()
        in_market = pos[ok] != 0
        rows.append({'horizon': h, 'method': m, 'n': len(pnl), 'total_pnl': pnl.sum() if len(pnl) else np.nan,
                     'ann_sharpe': np.sqrt(252) * pnl.mean() / pnl.std() if pnl.std() > 0 else np.nan,
                     'max_drawdown': float((eq.cummax() - eq).max()),
                     'win_rate': float((pnl[in_market] > 0).mean()) if in_market.any() else np.nan,
                     'trades': int(entries), 'avg_holding_days': float(in_market.sum() / max(entries, 1)),
                     'profit_per_trade': float(pnl.sum() / max(entries, 1)), 'days_in_market': int(in_market.sum())})
        curves[m] = eq
    return pd.DataFrame(rows).round(4), curves


def plot(f, curves, kind, mode):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor=SURFACE, sharex=True)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.grid(axis='y', color=GRID, lw=0.6)
    g = f[(f.horizon == 20) & (f.method == 'rw')].sort_values('origin_date')
    axes[0].plot(g['origin_date'], g['spread'], color=MUTED, lw=1, label='Spread')
    for color, m in zip(SERIES, ('ma_dev', 'enet')):
        x = f[(f.horizon == 20) & (f.method == m)].sort_values('origin_date')
        axes[0].plot(x['origin_date'], x['spread'] + x['forecast_change'], color=color, lw=1.3,
                     label=f'{LABELS[m]}: forecast spread in 20 days')
    axes[0].set_ylabel('USD/bbl', fontsize=8, color=MUTED)
    axes[0].legend(frameon=False, fontsize=9, loc='upper left', ncol=3)
    name = {'brent_wti': 'Brent-WTI spread', 'crack_321': '3-2-1 crack spread'}[kind]
    axes[0].set_title(f'{name} and its 20-day-ahead forecast (plotted at the origin)', loc='left', fontsize=10, color=INK)
    for color, m in zip(SERIES, ('ma_dev', 'enet', 'ou', 'lgbm')):
        if m in curves:
            c = curves[m]
            axes[1].plot(c.index, c, color=color, lw=1.5, label=LABELS[m])
    axes[1].axhline(0, color=BASELINE, lw=0.8)
    axes[1].set_ylabel('cumulative P&L, USD/bbl', fontsize=8, color=MUTED)
    axes[1].legend(frameon=False, fontsize=9, loc='upper left', ncol=4)
    note = {'brent_wti': 'spot spread: London vs New York closes, see the asynchronous-close caveat',
            'crack_321': 'all legs assessed at the New York close'}[kind]
    axes[1].set_title(f'Spread mean-reversion rule from 20-day forecasts (hurdle USD {THRESHOLD}, cost USD {COST} per '
                      f'position change; {note})', loc='left', fontsize=10, color=INK)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.tight_layout()
    fig.savefig(FIG / f'{kind}_{mode}_test.png', dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(kind):
    FIG.mkdir(parents=True, exist_ok=True)
    f = read_final(OUT / f'{kind}_test_forecasts.csv')
    f['actual_change'] = f['actual'] - f['spread']
    acc = accuracy(f)
    acc.to_csv(OUT / f'{kind}_test_metrics.csv', index=False, lineterminator='\n')
    scored = f.dropna(subset=['actual_change'])
    years = scored.assign(year=scored.target_date.dt.year, se=(scored.forecast_change - scored.actual_change) ** 2,
                          rw=scored.actual_change ** 2)
    by_year = years.groupby(['mode', 'horizon', 'year', 'method']).apply(
        lambda g: np.sqrt(g.se.mean() / g.rw.mean()) if g.rw.mean() > 0 else np.nan,
        include_groups=False).unstack('method')
    by_year.to_csv(OUT / f'{kind}_test_by_year.csv', lineterminator='\n')
    trades = []
    for mode in MODES:
        block = f[f['mode'] == mode]
        tr20, curves = trading(block, 20)
        tr5, _ = trading(block, 5)
        tr = pd.concat([tr20, tr5], ignore_index=True)
        tr.insert(0, 'mode', mode)
        tr.insert(0, 'protocol_id', PROTOCOL_ID)
        tr['inference'] = 'descriptive_short_window'
        trades.append(tr)
        plot(block, curves, kind, mode)
    pd.concat(trades, ignore_index=True).to_csv(OUT / f'{kind}_test_trading.csv', index=False, lineterminator='\n')
    print(acc.to_string(index=False))
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spread', choices=['brent_wti', 'crack_321'], default='brent_wti')
    main(parser.parse_args().spread)

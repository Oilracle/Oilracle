"""Score only realized October targets, separately for each frozen forecasting mode.

uv run python -m m3_ml.risk.analyze
Unreached horizons remain in metric tables with n=0 and undefined scores.
October is a descriptive holdout, not an inferential experiment: at most 17
overlapping daily origins (one for fixed-origin paths) do not support the
asymptotic sampling argument for HAC/DM or Wald success claims. Keep validation
comparisons separate; final tables report losses/calibration summaries only.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from m3_ml.common.data import load_table, safe_log_price
from m3_ml.common.protocol import FIT_END, MODES, PROTOCOL_ID, TEST_END, TEST_START
from m3_ml.distribution import run as dist
from .volatility import OUT, qlike

FIG = OUT / 'figures'
TARGET_VOL, MAX_W, COSTS = 0.30, 2.0, (0.0, 0.0005, 0.001)
LIMIT, HEDGE_H = 0.10, 20


def read_final(path):
    f = pd.read_csv(path, parse_dates=['origin_date', 'target_date'])
    if not {'mode', 'protocol_id'}.issubset(f) or not f['protocol_id'].eq(PROTOCOL_ID).all():
        raise ValueError(f'{path} is not a frozen October forecast file')
    f.loc[~f.target_date.between(TEST_START, TEST_END), 'actual'] = np.nan
    return f




def mincer_zarnowitz(rv, s2):
    """Descriptive calibration slope/intercept and R-squared, without a Wald test."""
    if len(rv) < 3 or pd.Series(s2).nunique() < 2:
        return (np.nan,) * 3
    fit = sm.OLS(np.asarray(rv), sm.add_constant(np.asarray(s2))).fit()
    return float(fit.params[0]), float(fit.params[1]), float(fit.rsquared)


def study1():
    f = read_final(OUT / 'volatility_test_forecasts.csv')
    f['rv'] = f['actual']
    f['qlike'] = qlike(f.rv, f.sigma2)
    f['se'] = (f.rv - f.sigma2) ** 2
    rows = []
    for (mode, h, method), block in f.groupby(['mode', 'horizon', 'method']):
        g = block.dropna(subset=['actual', 'sigma2'])
        row = {'mode': mode, 'horizon': h, 'method': method, 'n': len(g),
               'qlike': g.qlike.mean(), 'mse_x1e6': 1e6 * g.se.mean()}
        for other in ('garch', 'ewma', 'ovx_raw', 'ovx_cal'):
            ref = f[(f['mode'] == mode) & (f.horizon == h) & (f.method == other)].set_index('origin_date').qlike
            d = g.qlike.to_numpy() - ref.reindex(g.origin_date).to_numpy()
            row[f'qlike_minus_{other}'] = float(np.mean(d)) if len(d) else np.nan
        row['mz_a'], row['mz_b'], row['mz_r2'] = mincer_zarnowitz(g.rv, g.sigma2)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.insert(0, 'protocol_id', PROTOCOL_ID)
    out['inference'] = 'descriptive_short_window'
    out.to_csv(OUT / 'volatility_test_metrics.csv', index=False, lineterminator='\n')
    f.assign(year=f.target_date.dt.year).groupby(['mode', 'horizon', 'year', 'method']).qlike.mean().to_csv(OUT / 'volatility_test_by_year.csv')
    for mode in MODES:
        fig, ax = plt.subplots(figsize=(10, 4))
        for method, g in f[(f['mode'] == mode) & (f.horizon == 20)].groupby('method'):
            ax.plot(g.origin_date, np.sqrt(252 / 20 * g.sigma2), label=method)
        ax.set_title(f'{mode}: frozen 20-row volatility forecasts (unreached targets are unscored)')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / f'{mode}_volatility_test_h20.png')
        plt.close(fig)
    return out


def study2():
    f = read_final(OUT / 'events_test_probabilities.csv')
    f['event'] = f['actual']
    f['brier'] = (f.prob - f.event) ** 2
    pc = f.prob.clip(1e-4, 1 - 1e-4)
    f['logloss'] = -(f.event * np.log(pc) + (1 - f.event) * np.log(1 - pc))
    rows = []
    for (mode, h, side, method), block in f.groupby(['mode', 'horizon', 'side', 'method']):
        g = block.dropna(subset=['actual', 'prob'])
        ref = f[(f['mode'] == mode) & (f.horizon == h) & (f.side == side) & (f.method == 'climatology')].set_index('origin_date').brier
        ref_loss = ref.reindex(g.origin_date)
        baseline = ref_loss.mean()
        skill = 1 - g.brier.mean() / baseline if baseline > 0 else np.nan
        rows.append({'mode': mode, 'horizon': h, 'side': side, 'method': method, 'n': len(g),
                     'events': int(g.event.sum()), 'brier': g.brier.mean(), 'brier_skill': skill,
                     'logloss': g.logloss.mean(),
                     'auc': roc_auc_score(g.event, g.prob) if g.event.nunique() == 2 else np.nan})
    out = pd.DataFrame(rows)
    out.insert(0, 'protocol_id', PROTOCOL_ID)
    out['inference'] = 'descriptive_short_window'
    out.to_csv(OUT / 'events_test_metrics.csv', index=False, lineterminator='\n')
    for mode in MODES:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], '--', color='gray')
        for method, g in f[(f['mode'] == mode) & (f.horizon == 5) & (f.side == 'down')].dropna(subset=['event']).groupby('method'):
            bins = pd.cut(g.prob, np.linspace(0, 1, 11), include_lowest=True)
            r = g.groupby(bins, observed=True).agg(pred=('prob', 'mean'), obs=('event', 'mean'))
            ax.plot(r.pred, r.obs, 'o', label=method)
        ax.set_title(f'{mode}: 5-row downside reliability (small retrospective sample)')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / f'{mode}_events_reliability.png')
        plt.close(fig)
    return out


def perf(pnl, w):
    if not len(pnl):
        return {k: np.nan for k in ('ann_return', 'ann_vol', 'sharpe', 'max_drawdown', 'es_97_5', 'worst_day', 'turnover_per_year')}
    eq, std = pnl.cumsum(), pnl.std()
    return {'ann_return': 252 * pnl.mean(), 'ann_vol': np.sqrt(252) * std,
            'sharpe': np.sqrt(252) * pnl.mean() / std if std > 0 else np.nan,
            'max_drawdown': float((eq.cummax() - eq).max()),
            'es_97_5': -float(pnl[pnl <= pnl.quantile(.025)].mean()), 'worst_day': float(pnl.min()),
            'turnover_per_year': 252 * float(np.abs(np.diff(w)).mean()) if len(w) > 1 else np.nan}


def study3a():
    table = load_table()
    returns = safe_log_price(table.target_wti).diff()
    from .protocol import available_return_mask
    historical = returns[available_return_mask(table, FIT_END)]
    sigma_train = np.sqrt(252) * historical.std()
    next_return = returns.shift(-1).set_axis(pd.DatetimeIndex(table.date))
    f = read_final(OUT / 'volatility_test_forecasts.csv')
    rows = []
    for mode in MODES:
        block = f[(f['mode'] == mode) & (f.horizon == 1)].dropna(subset=['actual'])
        dates = pd.DatetimeIndex(sorted(block.origin_date.unique()))
        curves = {}
        for method in ('unmanaged', 'constant', *sorted(block.method.unique())):
            if method == 'unmanaged':
                w = np.ones(len(dates))
            elif method == 'constant':
                w = np.full(len(dates), min(MAX_W, TARGET_VOL / sigma_train))
            else:
                s2 = block[block.method == method].set_index('origin_date').sigma2.reindex(dates).to_numpy()
                w = np.minimum(MAX_W, TARGET_VOL / np.sqrt(252 * s2))
            ret = next_return.reindex(dates).to_numpy()
            for cost in COSTS:
                turnover = np.abs(np.diff(np.concatenate([[0.0], w])))
                pnl = pd.Series(w * ret - cost * turnover, index=dates)
                rows.append({'mode': mode, 'rule': method, 'horizon': 1, 'n': len(pnl), 'cost_bp': cost * 1e4, **perf(pnl, w)})
                if cost == .0005:
                    curves[method] = pnl.cumsum()
        fig, ax = plt.subplots(figsize=(10, 4))
        for method, curve in curves.items():
            ax.plot(curve.index, curve, label=method)
        ax.set_title(f'{mode}: volatility sizing, 5bp turnover costs; retrospective illustration')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / f'{mode}_economic_vol_targeting.png')
        plt.close(fig)
    out = pd.DataFrame(rows)
    out.insert(0, 'protocol_id', PROTOCOL_ID)
    out['method'] = out['rule']
    out['inference'] = 'descriptive_short_window'
    out.to_csv(OUT / 'economic_vol_targeting.csv', index=False, lineterminator='\n')
    return out, sigma_train


def study3b():
    q = read_final(dist.OUT / 'test_quantiles.csv')
    q = q[q.horizon == HEDGE_H]
    rows = []
    for mode in MODES:
        block = q[q['mode'] == mode]
        for role, col, sign in (('producer (long)', 'q_0.050', -1), ('consumer (short)', 'q_0.950', 1)):
            for method in ('no hedge', 'full hedge', *sorted(block.method.unique())):
                source = block[block.method == ('hs' if method in ('no hedge', 'full hedge') else method)].dropna(subset=['actual'])
                loss = sign * source.actual_log_return.to_numpy()
                eta = np.zeros(len(source)) if method == 'no hedge' else np.ones(len(source)) if method == 'full hedge' else np.clip(1 - LIMIT / np.maximum(sign * source[col].to_numpy(), 1e-9), 0, 1)
                hedged = (1 - eta) * loss
                breach = hedged > LIMIT
                rows.append({'mode': mode, 'horizon': HEDGE_H, 'role': role, 'rule': method, 'n': len(source),
                             'breach_rate': breach.mean() if len(source) else np.nan,
                             'mean_hedge_ratio': eta.mean() if len(source) else np.nan,
                             'es_97_5_hedged_loss': np.sort(hedged)[-max(1, int(.025 * len(hedged))):].mean() if len(source) else np.nan,
                             'hedge_adjustments': int((np.abs(np.diff(eta)) > .05).sum())})
    out = pd.DataFrame(rows)
    out.insert(0, 'protocol_id', PROTOCOL_ID)
    out['method'] = out['rule']
    out['inference'] = 'descriptive_short_window'
    out.to_csv(OUT / 'economic_var_hedging.csv', index=False, lineterminator='\n')
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for name, result in (('volatility', study1()), ('events', study2()), ('volatility sizing', study3a()[0]), ('VaR hedging', study3b())):
        print(name, '\n', result.to_string(index=False))


if __name__ == '__main__':
    main()

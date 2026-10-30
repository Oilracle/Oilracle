"""Score frozen residual forecasts by mode and realized October target dates.

uv run python -m m3_ml.residual.analyze
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m3_ml.common.protocol import HORIZONS, MODES, PROTOCOL_ID
from m3_ml.distribution.quantiles import QUANTILE_COLUMNS, covered, crps
from m3_ml.risk.analyze import read_final

OUT = Path(__file__).resolve().parent / 'outputs' / 'frozen_october_2026'


def _ratio(value, baseline):
    return value / baseline if baseline > 0 else np.nan


def point_table(forecasts):
    f = forecasts.copy()
    f['se'] = (f.forecast_log_return - f.actual_log_return) ** 2
    f['ae_usd'] = (f.forecast - f.actual).abs()
    rows = []
    for (mode, h, model), block in f.groupby(['mode', 'horizon', 'model']):
        g = block.dropna(subset=['actual', 'forecast'])
        references = {}
        row = {'protocol_id': PROTOCOL_ID, 'mode': mode, 'horizon': h, 'model': model,
               'n': len(g), 'mae_usd': g.ae_usd.mean(), 'rmse': np.sqrt(g.se.mean()),
               'inference': 'descriptive_short_window'}
        for name, label in (('random_walk', 'rw'), ('base', 'base')):
            ref = f[(f['mode'] == mode) & (f.horizon == h) & (f.model == name)].set_index('origin_date').se.reindex(g.origin_date)
            references[label] = ref
            row[f'rmse_vs_{label}'] = np.sqrt(_ratio(g.se.mean(), ref.mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def risk_table(quantiles):
    rows = []
    for (mode, h), block in quantiles.groupby(['mode', 'horizon']):
        scored = block.dropna(subset=['actual_log_return', *QUANTILE_COLUMNS]).copy()
        if len(scored):
            scored['crps'] = crps(scored[QUANTILE_COLUMNS].to_numpy(), scored.actual_log_return.to_numpy())
            scored['covered_90'] = covered(scored[QUANTILE_COLUMNS].to_numpy(), scored.actual_log_return.to_numpy(), .9)
        else:
            scored['crps'], scored['covered_90'] = np.nan, np.nan
        common = None
        for method in block.method.unique():
            dates = set(scored[scored.method == method].origin_date)
            common = dates if common is None else common & dates
        scored = scored[scored.origin_date.isin(common or set())]
        for method in block.method.unique():
            g = scored[scored.method == method]
            row = {'protocol_id': PROTOCOL_ID, 'mode': mode, 'horizon': h, 'method': method,
                   'n': len(g), 'crps': g.crps.mean(), 'coverage_90': g.covered_90.mean(),
                   'inference': 'descriptive_short_window'}
            for name, label in (('ewma', 'ewma'), ('base_hs', 'base')):
                ref = scored[scored.method == name].set_index('origin_date').crps.reindex(g.origin_date)
                row[f'crps_vs_{label}'] = _ratio(g.crps.mean(), ref.mean())
            rows.append(row)
    return pd.DataFrame(rows)


def plot(point, risk):
    figures = OUT / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        for data, key, value, name in ((point, 'model', 'rmse_vs_rw', 'point_rmse'),
                                       (risk, 'method', 'crps_vs_ewma', 'risk_crps')):
            fig, ax = plt.subplots(figsize=(10, 5))
            block = data[(data['mode'] == mode) & data.horizon.isin(HORIZONS)]
            pivot = block.pivot(index='horizon', columns=key, values=value).reindex(HORIZONS)
            pivot.plot.bar(ax=ax)
            ax.axhline(1, color='gray', linewidth=.7)
            ax.set_title(f'{mode}: {value}; unreached horizons have n=0 and no bar')
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(figures / f'{mode}_{name}.png')
            plt.close(fig)
    return figures


def main():
    f = read_final(OUT / 'residual_forecasts.csv')
    f.loc[f.actual.isna(), 'actual_log_return'] = np.nan
    q = read_final(OUT / 'residual_quantiles.csv')
    q.loc[q.actual.isna(), 'actual_log_return'] = np.nan
    point, risk = point_table(f), risk_table(q)
    point.to_csv(OUT / 'point_metrics.csv', index=False, lineterminator='\n')
    risk.to_csv(OUT / 'risk_metrics.csv', index=False, lineterminator='\n')
    plot(point, risk)
    print(point.to_string(index=False))
    print(risk.to_string(index=False))
    return point, risk


if __name__ == '__main__':
    main()

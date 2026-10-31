"""Prepare paired first-stage STL evidence without changing fitted models."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / '.local/seasonality_slides'
SOURCE = ROOT / 'm2_seasonality/outputs/frozen_october_2026/validation_predictions.csv'
BUILD.mkdir(parents=True, exist_ok=True)
pred = pd.read_csv(SOURCE)
pred = pred[(pred['mode'] == 'daily_update') & pred.scored & pred.label_mature_at_selection.fillna(False)
            & (pred.target_type == 'price')]
assert (pd.to_datetime(pred.label_available_date) <= pd.Timestamp('2026-09-30')).all()
assert set(pred.fit_end) == {'2023-08-31'}


def bootstrap(frame, horizon):
    n, block = len(frame), max(20, horizon)
    e = (frame.forecast.to_numpy() - frame.actual.to_numpy()) ** 2
    b = (frame.forecast_base.to_numpy() - frame.actual.to_numpy()) ** 2
    rng = np.random.default_rng(5210 + horizon)
    sims = []
    for _ in range(50):
        starts = rng.integers(0, n - block + 1, size=(100, int(np.ceil(n / block))))
        indexes = (starts[:, :, None] + np.arange(block)).reshape(100, -1)[:, :n]
        sims.extend(100 * (1 - np.sqrt(e[indexes].mean(axis=1) / b[indexes].mean(axis=1))))
    lo, hi = np.percentile(sims, [2.5, 97.5])
    return dict(horizon=horizon, n=n, rmse=float(np.sqrt(e.mean())), baseline_rmse=float(np.sqrt(b.mean())),
                gain_pct=float(100 * (1 - np.sqrt(e.mean() / b.mean()))), ci_low=float(lo), ci_high=float(hi),
                block=block, bootstrap_reps=5000)


evidence = {'source': SOURCE.relative_to(ROOT).as_posix(),
            'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'metrics': [], 'paths': {}}
for horizon in [1, 5, 20, 60]:
    a = pred[(pred.horizon == horizon) & (pred.method == 'stl_monthly_daily_bridge')]
    b = pred[(pred.horizon == horizon) & (pred.method == 'random_walk')]
    d = a.merge(b[['origin_date', 'target_date', 'horizon', 'actual', 'forecast']],
                on=['origin_date', 'target_date', 'horizon'], suffixes=('', '_base'), validate='one_to_one').sort_values('origin_date')
    assert np.allclose(d.actual, d.actual_base)
    metrics = bootstrap(d, horizon)
    evidence['metrics'].append(metrics)
    if horizon in [20, 60]:
        evidence['paths'][str(horizon)] = d[['target_date', 'actual', 'forecast', 'forecast_base']].to_dict('records')
assert [m['n'] for m in evidence['metrics']] == [758, 754, 739, 699]
assert np.allclose([m['gain_pct'] for m in evidence['metrics']], [.2574387962, .1446821641, 2.2100285603, 6.7126869029])
assert all(m['ci_low'] < 0 < m['ci_high'] for m in evidence['metrics'])
(BUILD / 'evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
print(json.dumps(evidence['metrics']))

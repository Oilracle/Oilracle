"""Frozen modes, unreached targets and publication-aware residual state regressions."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from m3_ml.common.protocol import FIT_END, HORIZONS, PROTOCOL_ID
from m3_ml.common.walkforward import forecast_origins
from m3_ml.risk import protocol
from m3_ml.residual import base


def test_modes_and_unreached_targets():
    dates = pd.Series(pd.bdate_range('2026-09-01', '2026-10-23'))
    fixed = forecast_origins(dates, 'fixed_origin')
    daily = forecast_origins(dates, 'daily_update')
    assert len(fixed) == 1 and dates.iloc[fixed[0]] == pd.Timestamp(FIT_END)
    assert daily[0] == fixed[0] and dates.iloc[daily[-1]] < pd.Timestamp('2026-10-23')
    path = protocol.path_horizons(dates, 'fixed_origin')
    assert set(range(1, 18)).issubset(path) and set(HORIZONS).issubset(path)
    unseen = protocol.frame(dates, fixed, 60, 'fixed_origin', 'example', [1.2], [np.nan])
    assert unseen.target_date.isna().all() and unseen.actual.isna().all()
    assert unseen.status.eq('unscored').all() and unseen.protocol_id.eq(PROTOCOL_ID).all()
    one = protocol.frame(dates, fixed, 1, 'fixed_origin', 'example', [1.2], [1.1])
    assert one.target_date.eq(pd.Timestamp('2026-10-01')).all() and one.actual.eq(1.1).all()


def test_historical_selection_is_rejected(tmp_path):
    path = tmp_path / 'selection.json'
    path.write_text('{"old_model": {"params": {}}}', encoding='utf-8')
    with pytest.raises(ValueError, match='historical'):
        protocol.load_selection(path)
    selected = {'example/h1': {'params': {'k': 20}, 'score': .2}}
    protocol.save_selection(path, selected)
    assert protocol.load_selection(path) == selected


def test_residual_base_never_refits_october_and_bridges_unpublished_gap(monkeypatch):
    dates = pd.Series(pd.bdate_range('2025-01-01', '2026-10-23'))
    raw = 60 * np.exp(np.arange(len(dates)) * .0002 + np.sin(np.arange(len(dates)) / 9) * .01)
    table = pd.DataFrame({'date': dates, 'target_wti': raw,
                          'wti_spot_last_published': pd.Series(raw).shift(3).bfill()})
    monkeypatch.setattr(base, 'label_available_dates', lambda frame: frame.date + pd.offsets.BDay(3))
    monkeypatch.setattr(base, 'observable_price', lambda frame: frame.wti_spot_last_published)
    fit_lengths = []
    def fitted(adjusted):
        fit_lengths.append(len(adjusted))
        return .2, .001
    monkeypatch.setattr(base, '_fit_arima', fitted)
    full = base.build_base(table, refit_every=1000)
    fit_row = int(forecast_origins(dates, 'fixed_origin')[0])
    last_public = fit_row - 3
    assert fit_lengths and max(fit_lengths) <= last_public + 1
    assert np.isfinite(full.loc[fit_row, 'base_60'])
    changed = table.copy()
    changed.loc[changed.date > FIT_END, 'target_wti'] *= 2
    other = base.build_base(changed, refit_every=1000)
    assert np.allclose(full.loc[fit_row, ['base_1', 'base_5', 'base_20', 'base_60']].astype(float),
                       other.loc[fit_row, ['base_1', 'base_5', 'base_20', 'base_60']].astype(float))
    assert max(fit_lengths) <= last_public + 1


def test_volatility_fits_once_per_horizon_for_both_modes(monkeypatch, tmp_path):
    from m3_ml.risk import volatility
    dates = pd.Series(pd.bdate_range('2025-01-01', '2026-10-23'))
    rv = pd.DataFrame({h: np.full(len(dates), h * .0001) for h in range(1, 61)})
    ctx = SimpleNamespace(inp=SimpleNamespace(dates=dates), rv=rv)
    cutoff = int(forecast_origins(dates, 'fixed_origin')[0])
    calls = []
    def predictor(context, h, refit, train, origins):
        calls.append((h, refit))
        assert refit == cutoff and np.all(train + h < cutoff)
        return np.full(len(origins), h * .0002)
    monkeypatch.setattr(volatility, 'Context', lambda: ctx)
    monkeypatch.setattr(volatility, 'METHODS', {'example': (predictor, [{}])})
    monkeypatch.setattr(volatility, 'load_selection', lambda path: {f'example/h{h}': {'params': {}} for h in HORIZONS})
    monkeypatch.setattr(volatility, 'final_training_positions', lambda dates, ok, target, h: np.arange(cutoff - h))
    monkeypatch.setattr(volatility, 'OUT', tmp_path)
    result = volatility.test()
    assert len(calls) == len(protocol.path_horizons(dates, 'fixed_origin'))
    fixed = result[result['mode'] == 'fixed_origin']
    daily = result[result['mode'] == 'daily_update']
    for h in HORIZONS:
        assert fixed[fixed.horizon == h].forecast.iloc[0] == daily[daily.horizon == h].forecast.iloc[0]
    assert result[result.horizon == 60].actual.isna().all()


def test_october_tables_are_descriptive_only():
    from m3_ml.spread.analyze import accuracy
    from m3_ml.residual.analyze import point_table
    from m3_ml.risk.analyze import mincer_zarnowitz
    dates = pd.bdate_range('2026-09-30', periods=17)
    spread = pd.DataFrame({'mode': 'daily_update', 'horizon': 1, 'method': 'ou',
                           'forecast_change': np.linspace(.1, .2, 17),
                           'actual_change': np.linspace(.2, .3, 17)})
    scored = accuracy(spread)
    points = pd.concat([pd.DataFrame({'mode': 'daily_update', 'origin_date': dates,
                                     'horizon': 1, 'model': model,
                                     'forecast_log_return': predicted, 'actual_log_return': .02,
                                     'forecast': 80 * np.exp(predicted), 'actual': 80 * np.exp(.02)})
                        for model, predicted in (('random_walk', 0), ('base', .01), ('lightgbm', .015))],
                       ignore_index=True)
    point = point_table(points)
    for result in (scored, point):
        assert result.inference.eq('descriptive_short_window').all()
        assert result.n.eq(17).all()
        assert not any(name.startswith(('dm_', 'beats_')) for name in result.columns)
    assert len(mincer_zarnowitz(np.arange(17) + 1, np.arange(17) + 2)) == 3

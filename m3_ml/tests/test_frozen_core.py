"""Regression coverage for the frozen tabular/distribution protocol cutover."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from m3_ml.common import data
from m3_ml.common.features import build_features
from m3_ml.common.metrics import summarize
from m3_ml.common.preprocessing import make_preprocessor
from m3_ml.common.protocol import FIT_END, TEST_END, VALIDATION_END
from m3_ml.common.walkforward import (final_training_positions, fixed_path_horizons,
                                      forecast_origins, walk_forward)
from m3_ml.distribution import run
from m3_ml.distribution.data import load_inputs
from m3_ml.distribution.tails import ALL_TAUS
from m3_ml.tabular.run_direct import frozen_forecast, make_frame


@pytest.fixture(scope='module')
def table():
    return data.load_table()


def test_raw_target_is_not_an_origin_feature(table):
    changed = table.copy()
    changed['target_wti'] *= 4
    pd.testing.assert_frame_equal(build_features(table), build_features(changed))
    pd.testing.assert_series_equal(data.observable_price(table), data.observable_price(changed))


def test_anchor_uses_newest_observation_not_last_release(monkeypatch):
    evidence = pd.DataFrame({'date': pd.to_datetime(['2026-09-01', '2026-09-02', '2026-09-03']),
                             'target_wti': [70.0, 71.0, 72.0],
                             'available_date': pd.to_datetime(['2026-09-04', '2026-09-10', '2026-09-08'])})
    monkeypatch.setattr(data, '_publication_evidence', lambda: evidence.set_index('date'))
    frame = pd.DataFrame({'date': pd.to_datetime(['2026-09-03', '2026-09-04', '2026-09-08', '2026-09-10']),
                          'wti_spot_last_published': [69.0] * 4, 'target_wti': [np.nan] * 4})
    assert data.observable_price(frame).tolist() == [69.0, 70.0, 72.0, 72.0]


def test_all_training_window_labels_are_public(table):
    dates = table['date']
    for h in (1, 5, 20, 60):
        train = final_training_positions(dates, np.ones(len(table), dtype=bool),
                                          data.targets(table)[h].notna(), h)
        available = data.label_available_dates(table)
        for shift in range(h + 1):
            assert available.iloc[train + shift].le(FIT_END).all()
        assert dates.iloc[train + h].le(FIT_END).all()


def test_unavailable_interior_target_is_not_accepted(monkeypatch):
    dates = pd.Series(pd.bdate_range('2026-09-01', periods=12))
    available = dates.copy()
    available.iloc[5] = pd.Timestamp('2026-11-01')
    monkeypatch.setattr('m3_ml.common.walkforward.label_available_dates', lambda _: available)
    train = final_training_positions(dates, np.ones(12, bool), np.ones(12, bool), 5,
                                      cutoff='2026-09-30')
    assert not set(range(5)) & set(train)


def test_october_refitting_is_rejected(table):
    with pytest.raises(ValueError, match='historical validation'):
        list(walk_forward(table.date, np.ones(len(table), bool), np.ones(len(table), bool),
                          '2026-10-01', TEST_END, 1, 21))


def test_fixed_path_and_unreached_counts(table):
    fixed = forecast_origins(table.date, 'fixed_origin')
    assert table.date.iloc[fixed[0]] == pd.Timestamp(FIT_END)
    assert set(range(1, 18)) <= set(fixed_path_horizons(table.date))
    y = data.targets(table)[60]
    f = make_frame(table, y, 60, 'random_walk', pd.Series(0.0, index=fixed),
                   'fixed_origin', 'direct')
    assert f.actual.isna().all()
    assert f.target_date.isna().all()
    assert f.forecast.notna().all()
    assert summarize(f)['n'].iloc[0] == 0


def test_tabular_final_estimator_fits_once_and_freezes_scaler(table):
    calls = []

    class RecordedRidge(Ridge):
        def fit(self, X, y):
            calls.append(len(y))
            return super().fit(X, y)

    fitted_models = []

    def factory():
        model = make_pipeline(make_preprocessor('scaled'), RecordedRidge(alpha=10))
        fitted_models.append(model)
        return model

    X, y = build_features(table), data.targets(table)[5]
    origins = forecast_origins(table.date, 'daily_update')
    a = frozen_forecast(factory, {}, X, y, table.date, 5, origins)
    assert len(calls) == 1
    scaler_mean = fitted_models[0].steps[0][1].steps[0][1].mean_.copy()
    changed_X = X.copy()
    changed_X.loc[table.date.gt(FIT_END)] *= 10
    b = frozen_forecast(factory, {}, changed_X, y, table.date, 5, origins)
    assert np.isclose(a.iloc[0], b.iloc[0])
    assert np.array_equal(scaler_mean, fitted_models[1].steps[0][1].steps[0][1].mean_)
    changed_table = table.copy()
    changed_table.loc[table.date.gt(FIT_END), data.PRICE] *= 3
    altered_y = data.targets(changed_table)[5]
    c = frozen_forecast(factory, {}, X, altered_y, table.date, 5, origins)
    assert np.allclose(a, c)


@pytest.mark.parametrize('method', list(run.METHODS))
def test_distribution_fixed_origin_ignores_all_october_inputs(table, method):
    full = load_inputs(table)
    changed = table.copy()
    columns = changed.select_dtypes(include='number').columns
    changed.loc[changed.date.gt(FIT_END), columns] *= 1.5
    altered = load_inputs(changed)
    origins = forecast_origins(table.date, 'daily_update')
    refit = int(origins[0])
    params = run.METHODS[method][1][0]
    selected = {f'{m}/h5': {'params': run.METHODS[m][1][0], 'score': i}
                for i, m in enumerate(('a_ridge', 'a_lightgbm'))}
    a = run.fitted_block(full, method, params, 5, refit, origins, ALL_TAUS, selected)
    b = run.fitted_block(altered, method, params, 5, refit, origins, ALL_TAUS, selected)
    assert np.allclose(a[0], b[0]), method
    # Adding daily-update rows must not change fixed-origin Monte Carlo draws.
    c = run.fitted_block(full, method, params, 5, refit, origins[:1], ALL_TAUS, selected)
    assert np.allclose(a[0], c[0]), method


def test_validation_targets_mature_before_selection(table):
    inp = load_inputs(table)
    available = data.label_available_dates(table)
    for h in (1, 5, 20, 60):
        for year in run.FOLD_YEARS:
            origins = run.fold_origins(inp, h, year)
            assert table.date.iloc[origins + h].le(VALIDATION_END).all()
            assert available.iloc[origins + h].le(VALIDATION_END).all()

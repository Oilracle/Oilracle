"""News release calendars, stale-source rejection and October coverage."""
import numpy as np
import pandas as pd
import pytest

from m3_ml.common.news import (NEWS_POLICY_ID, news_availability,
                               news_features, require_complete_features)


@pytest.fixture
def news():
    days = pd.date_range('2025-01-01', '2026-11-30')
    values = np.arange(len(days), dtype=float) + 1
    return pd.DataFrame({'ai_gpr': values, 'gpr_keyword': values * 2, 'epu': values * 3}, index=days)


def expected(news, series, endpoint, window):
    return np.log1p(news[series]).loc[:endpoint].iloc[-window:].mean()


def test_monthly_rollover_and_within_month_freeze(news):
    dates = pd.Series(pd.to_datetime(['2026-09-30', '2026-10-01', '2026-10-22', '2026-11-01']))
    got = news_features(dates, news)
    for row, endpoint in enumerate(['2026-08-31', '2026-09-30', '2026-09-30', '2026-10-31']):
        assert got.ai_gpr_ma7.iloc[row] == pytest.approx(expected(news, 'ai_gpr', endpoint, 7))
        assert got.ai_gpr_shock.iloc[row] == pytest.approx(
            expected(news, 'ai_gpr', endpoint, 7) - expected(news, 'ai_gpr', endpoint, 90))


def test_gpr_monday_and_federal_holiday_deferral(news):
    dates = pd.Series(pd.to_datetime(['2026-10-05', '2026-10-11', '2026-10-12', '2026-10-13']))
    audit = news_availability(dates, news).query("series == 'gpr_keyword'")
    assert audit.data_through.dt.strftime('%Y-%m-%d').tolist() == [
        '2026-10-04', '2026-10-04', '2026-10-04', '2026-10-11']
    assert audit.available_date.iloc[-1] == pd.Timestamp('2026-10-13')


def test_future_news_cannot_change_earlier_features(news):
    dates = pd.Series(pd.to_datetime(['2026-09-30', '2026-10-01', '2026-10-12']))
    original = news_features(dates, news)
    changed = news.copy()
    changed.loc['2026-10-01':, 'ai_gpr'] *= 1000
    changed.loc['2026-10-05':, 'gpr_keyword'] *= 1000
    changed.loc['2026-10-12':, 'epu'] *= 1000
    pd.testing.assert_frame_equal(original, news_features(dates, changed))
    # Dataset truncation and extra future rows must not affect the same origins.
    pd.testing.assert_frame_equal(original, news_features(dates, news.loc[:'2026-10-11']))


def test_missing_month_or_week_does_not_extend_old_release(news):
    cut = news.copy()
    cut.loc['2026-10-01':, 'ai_gpr'] = np.nan
    cut.loc['2026-10-12':, 'gpr_keyword'] = np.nan
    dates = pd.Series(pd.to_datetime(['2026-10-13', '2026-10-19', '2026-11-01']))
    got = news_features(dates, cut)
    assert got.ai_gpr_ma7.iloc[:2].notna().all()
    assert pd.isna(got.ai_gpr_ma7.iloc[2])
    assert np.isfinite(got.gpr_keyword_ma7.iloc[0])
    assert got.gpr_keyword_ma7.iloc[1:].isna().all()
    with pytest.raises(ValueError, match='do not fill or drop origins'):
        require_complete_features(got, dates, np.arange(3))


def test_epu_next_day_and_bounded_carry(news):
    cut = news.loc[:'2026-10-01', ['epu']]
    dates = pd.Series(pd.to_datetime(['2026-10-01', '2026-10-02', '2026-10-09', '2026-10-10']))
    got = news_features(dates, cut)
    assert got.epu_ma7.iloc[0] == pytest.approx(expected(news, 'epu', '2026-09-30', 7))
    assert got.epu_ma7.iloc[1] == got.epu_ma7.iloc[2]
    assert got.iloc[3].isna().all()


def test_missing_calendar_days_are_not_compressed(news):
    cut = news.loc[:'2026-09-30', ['epu']].drop(pd.date_range('2026-09-25', '2026-09-29'))
    got = news_features(pd.Series([pd.Timestamp('2026-10-01')]), cut)
    assert pd.isna(got.epu_ma7.iloc[0])  # two observed days, below MIN_DAYS[7]
    assert np.isfinite(got.epu_ma30.iloc[0])


def test_origin_index_order_is_preserved(news):
    dates = pd.Series(pd.to_datetime(['2026-10-19', '2026-09-30', '2026-10-01']), index=[8, 3, 5])
    got = news_features(dates, news)
    assert got.index.tolist() == [8, 3, 5]
    assert got.ai_gpr_ma7.iloc[0] == got.ai_gpr_ma7.iloc[2]


@pytest.mark.parametrize('bad', [-1, np.inf])
def test_invalid_news_fails(news, bad):
    news.iloc[-1, 0] = bad
    with pytest.raises(ValueError, match='nonnegative'):
        news_features(pd.Series([pd.Timestamp('2026-10-01')]), news)


def test_real_data_covers_full_forecast_grid_and_audits_availability():
    from m3_ml.common.data import load_table
    from m3_ml.common.features import build_features
    from m3_ml.common.walkforward import forecast_origins
    table = load_table()
    X = build_features(table, news=True)
    for mode in ('fixed_origin', 'daily_update'):
        origins = forecast_origins(table.date, mode)
        require_complete_features(X, table.date, origins)
    audit = news_availability(table.date)
    assert audit.complete.all()
    assert audit.available_date.le(audit.origin_date).all()
    assert audit.data_through.lt(audit.available_date).all()
    assert audit.news_policy_id.eq(NEWS_POLICY_ID).all()


def test_distribution_rejects_settings_from_another_news_snapshot(monkeypatch):
    import json
    from m3_ml.distribution import run
    saved = run.manifest()
    saved['selected'] = {'a_ridge/h5': {'params': {'alpha': 1, 'features': 'base_vol_news'}}}

    class SavedFile:
        def __truediv__(self, name):
            return self

        def exists(self):
            return True

        def read_text(self, **kwargs):
            return json.dumps(saved)

    monkeypatch.setattr(run, 'OUT', SavedFile())
    assert run.selected_settings() == saved['selected']
    saved['news']['sources'][0]['sha256'] = 'old-snapshot'
    with pytest.raises(ValueError, match='news policy/source snapshot changed'):
        run.selected_settings()


def test_all_news_estimators_predict_full_grid_with_frozen_preprocessing():
    from m3_ml.common.data import load_table, targets
    from m3_ml.common.features import build_features
    from m3_ml.common.walkforward import forecast_origins
    from m3_ml.tabular.models import MODELS
    from m3_ml.tabular.run_direct import frozen_forecast
    table = load_table()
    origins = forecast_origins(table.date, 'daily_update')
    y = targets(table)[5]
    for denoise in (False, True):
        X = build_features(table, news=True, denoise=denoise)
        require_complete_features(X, table.date, origins)
        for factory, grid in MODELS.values():
            predictions = frozen_forecast(factory, grid[0], X, y, table.date, 5, origins)
            assert predictions.index.equals(pd.Index(origins))
            assert np.isfinite(predictions).all()

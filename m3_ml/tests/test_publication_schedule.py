"""Regression coverage for delayed labels and explicitly assumed availability."""
from pathlib import Path
import importlib

import pandas as pd
import pytest


pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / 'scripts/export_team_data.py').exists(),
    reason='Maintainer-only raw-data rebuild script is not present',
)


def test_export_preserves_delayed_archive_and_distinguishes_holiday_assumption(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / 'scripts'))
    module = importlib.import_module('export_team_data')
    table = pd.DataFrame({'date': pd.to_datetime(['2022-06-17', '2024-05-24']),
                          'target_wti': [109.56, 78.48]})
    full = pd.DataFrame({'forecast_date': table['date'],
                         'target_verified_available_date': pd.to_datetime(['2022-06-30', None]),
                         'target_publication_source_file': ['data/raw/eia_reports/2022-06-29/table_11.csv', None]})
    result = module.build_publication(table, full)
    assert result['available_date'].tolist() == list(pd.to_datetime(['2022-06-30', '2024-06-03']))
    assert result['evidence'].tolist() == ['matching_dated_archive', 'assumed_weekly_release']
    assert result.loc[1, 'source_file'] == ''
    pd.testing.assert_frame_equal(result[['date', 'target_wti']], table)

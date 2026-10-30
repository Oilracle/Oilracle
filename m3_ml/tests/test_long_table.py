"""The long table (ml/long/build.py) uses nothing published after each row's date: removing
every source observation dated on or after a cutoff leaves all rows up to the cutoff unchanged."""
import pandas as pd
import pytest

from m3_ml.long.build import DATA, build

CUTOFFS = ['1991-03-01', '1998-12-24', '2008-09-12', '2016-10-03', '2020-04-21', '2024-04-05']  # Fridays catch same-day use of weekly data

pytestmark = pytest.mark.skipif(not (DATA / 'long/eia_api/sources.csv').exists(),
                                reason='data/long not downloaded (ml/long/download.py)')


@pytest.fixture(scope='module')
def full():
    return build()[0]


def test_starts_complete(full):
    assert full.date.min() == pd.Timestamp('1991-02-15')
    assert not full.isna().any().any()


@pytest.mark.parametrize('cutoff', CUTOFFS)
def test_truncated_sources_leave_earlier_rows_unchanged(full, cutoff):
    truncated = build(pd.Timestamp(cutoff))[0]
    rows = full.date <= cutoff
    features = full.columns[2:]
    pd.testing.assert_frame_equal(truncated.loc[rows, features], full.loc[rows, features])

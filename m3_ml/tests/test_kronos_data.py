"""Kronos training cutoffs, historical validation purging and normalization."""
import numpy as np
import pandas as pd
import pytest

from m3_ml.kronos import config, corpus
from m3_ml.kronos.dataset import WindowSampler, normalize



@pytest.mark.skipif(not corpus.MANIFEST.exists(), reason='corpus not downloaded')
@pytest.mark.parametrize('stage', [1, 2])
def test_training_windows_respect_cutoff(stage):
    cfg = config.STAGES[stage]
    train = WindowSampler(cfg, 'train', seed=0)
    _, _, train_ends = train.sample(2000)
    assert max(train_ends) <= pd.Timestamp(cfg['train_end'])


@pytest.mark.skipif(not corpus.MANIFEST.exists(), reason='corpus not downloaded')
def test_historical_validation_windows_purge_all_future_labels():
    cfg = config.STAGES[1]
    val = WindowSampler(cfg, 'val', seed=0)
    _, _, val_ends = val.sample(2000)
    assert min(val_ends) > pd.Timestamp(cfg['train_end'])
    assert max(val_ends) <= pd.Timestamp(cfg['val_end'])
    for (_, _, _, dates, _), starts in zip(val.series, val.starts):
        assert (dates[starts + config.LOOKBACK] > pd.Timestamp(cfg['train_end'])).all()
        assert (dates[starts + config.WINDOW - 1] <= pd.Timestamp(cfg['val_end'])).all()


def test_final_fit_explicitly_rejects_validation():
    with pytest.raises(ValueError, match='Final fit has no validation split'):
        WindowSampler(config.STAGES[2], 'val', seed=0)


def test_normalization_ignores_the_future_part():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(config.WINDOW, 6)).astype(np.float32)
    a, _ = normalize(w)
    w2 = w.copy()
    w2[config.LOOKBACK:] *= 100
    b, _ = normalize(w2)
    assert np.allclose(a[:config.LOOKBACK], b[:config.LOOKBACK])

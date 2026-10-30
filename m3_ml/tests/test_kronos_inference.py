"""Kronos inference uses only bars up to the origin, generation keeps every path
reproducibly, and the cached decoder gives the same logits and paths as the full
recomputation (random-weight models are enough for these checks)."""
import numpy as np
import pandas as pd
import pytest
import torch

from m3_ml.kronos import config, corpus
from m3_ml.kronos.generate import CachedDecoder, return_quantiles, sample_paths, sample_paths_cached
from m3_ml.kronos.run import BDAY, contexts
from m3_ml.kronos.train import causal_cross_attention
from m3_ml.kronos.universe import WTI_INPUT
from m3_ml.kronos.dataset import normalize, WindowSampler
from m3_ml.kronos.vendor import Kronos, KronosTokenizer

pytestmark = pytest.mark.skipif(not corpus.MANIFEST.exists(), reason='corpus not downloaded')
DEVICE = 'cpu'   # GPU generation crashed this computer (PLAN.md bug log); tests stay on the CPU


def random_tokens(b, n, seed=1):
    g = torch.Generator().manual_seed(seed)
    s1, s2 = torch.randint(0, 1024, (b, n), generator=g), torch.randint(0, 1024, (b, n), generator=g)
    zeros = torch.zeros(b, n, dtype=torch.long)
    stamps = torch.stack([zeros, zeros, torch.randint(0, 7, (b, n), generator=g),
                          torch.randint(1, 32, (b, n), generator=g), torch.randint(1, 13, (b, n), generator=g)], dim=-1)
    return s1, s2, stamps


def test_cached_decoder_matches_full_recomputation():
    torch.manual_seed(0)
    model = Kronos(**config.MODEL).eval()
    n0, k = 20, 6
    s1, s2, stamps = random_tokens(2, n0 + k)
    with torch.no_grad():
        dec = CachedDecoder(model, s1[:, :n0], s2[:, :n0], stamps[:, :n0], paths=1, total=n0 + k)
        for j in range(k):
            n = n0 + j
            logits, context = model.decode_s1(s1[:, :n], s2[:, :n], stamps[:, :n])
            sibling = torch.cat([s1[:, 1:n], s1[:, n:n + 1]], dim=1)
            assert torch.allclose(dec.s1_logits, logits[:, -1], atol=1e-4)
            assert torch.allclose(dec.s2_logits(s1[:, n:n + 1]), model.decode_s2(context, sibling)[:, -1], atol=1e-4)
            dec.append(s1[:, n:n + 1], s2[:, n:n + 1], stamps[:, n:n + 1])


def test_cached_sampling_matches_full_recomputation():
    torch.manual_seed(0)
    tokenizer = KronosTokenizer(**config.TOKENIZER).eval()
    model = Kronos(**config.MODEL).eval()
    x, sc, sf, _, _ = contexts(pd.to_datetime(['2019-06-03', '2021-03-01']))
    sf = sf[:, :5]
    a = sample_paths(tokenizer, model, x, sc, sf, 1.0, 0.9, paths=4, device=DEVICE, seed=3)
    b = sample_paths_cached(tokenizer, model, x, sc, sf, 1.0, 0.9, paths=4, device=DEVICE, seed=3)
    assert np.allclose(a, b, atol=1e-5)


def test_validation_loss_cannot_see_later_tokens():
    """With causal cross-attention, changing tokens after position i leaves the
    fine-token logits at position i unchanged (in plain eval mode they change)."""
    torch.manual_seed(0)
    model = Kronos(**config.MODEL).eval()
    s1, s2, stamps = random_tokens(2, 30)
    t1, t2, _ = random_tokens(2, 30, seed=2)
    later = lambda a, b: torch.cat([a[:, :15], b[:, 15:]], dim=1)

    def fine_logits(a1, a2, causal):
        with torch.no_grad():
            if causal:
                with causal_cross_attention(model):
                    out = model(a1[:, :-1], a2[:, :-1], stamps[:, :-1], use_teacher_forcing=True, s1_targets=a1[:, 1:])
            else:
                out = model(a1[:, :-1], a2[:, :-1], stamps[:, :-1], use_teacher_forcing=True, s1_targets=a1[:, 1:])
        return out[1][:, :14]

    assert torch.allclose(fine_logits(s1, s2, True), fine_logits(later(s1, t1), later(s2, t2), True), atol=1e-5)
    assert not torch.allclose(fine_logits(s1, s2, False), fine_logits(later(s1, t1), later(s2, t2), False), atol=1e-5)
    assert not model.dep_layer.cross_attn.training


def test_context_never_passes_the_origin():
    bars = pd.read_parquet(corpus.path(WTI_INPUT)).index
    origins = pd.to_datetime(['2020-04-20', '2023-07-04', '2026-10-06'])   # includes a US holiday
    x, sc, sf, _, _ = contexts(origins)
    for d, stamps_future in zip(origins, sf):
        end = bars[bars <= d][-1]
        last = bars[bars.searchsorted(d, side='right') - 1]
        assert last == end and last <= d
        first_future = end + BDAY
        assert (stamps_future[0, 2:] == [first_future.weekday(), first_future.day, first_future.month]).all()
    assert x.shape == (3, config.LOOKBACK, 6) and np.abs(x).max() <= config.CLIP


def test_generation_keeps_paths_and_is_reproducible():
    torch.manual_seed(0)
    tokenizer = KronosTokenizer(**config.TOKENIZER).to(DEVICE).eval()
    model = Kronos(**config.MODEL).to(DEVICE).eval()
    x, sc, sf, mean, std = contexts(pd.to_datetime(['2019-06-03', '2021-03-01']))
    sf = sf[:, :5]
    a = sample_paths(tokenizer, model, x, sc, sf, 1.0, 1.0, paths=6, device=DEVICE, seed=1)
    b = sample_paths(tokenizer, model, x, sc, sf, 1.0, 1.0, paths=6, device=DEVICE, seed=1)
    assert a.shape == (2, 6, 6) and np.allclose(a, b)
    assert not np.allclose(a[:, 0], a[:, 1])
    q = return_quantiles(a, mean, std, [1, 5], np.array([0.1, 0.5, 0.9]))
    assert all(np.all(np.diff(v, axis=1) >= 0) for v in q.values())


def test_normalization_uses_checkpoint_statistics():
    values = np.arange(60, dtype=np.float32).reshape(10, 6)
    mean, std = np.arange(6, dtype=np.float32), np.ones(6, dtype=np.float32) * 20
    actual, stats = normalize(values, (mean, std))
    expected = np.clip((values - mean) / (std + 1e-5), -config.CLIP, config.CLIP)
    np.testing.assert_array_equal(actual, expected.astype(np.float32))
    np.testing.assert_array_equal(stats[0], mean)
    np.testing.assert_array_equal(stats[1], std)


def test_frozen_context_ignores_later_actual_bars(monkeypatch):
    frame = pd.read_parquet(corpus.path(WTI_INPUT))
    origin = frame.index[config.LOOKBACK + 100]
    stats = (np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32) * 100)
    original = contexts([origin], stats=stats)
    altered = frame.copy()
    altered.loc[altered.index > origin] = 999999
    monkeypatch.setattr(pd, 'read_parquet', lambda *args, **kwargs: altered)
    result = contexts([origin], stats=stats)
    for expected, actual in zip(original, result):
        np.testing.assert_array_equal(actual, expected)


def test_validation_windows_purge_all_future_labels(monkeypatch):
    dates = pd.bdate_range('2020-01-01', periods=800)
    frame = pd.DataFrame({column: np.arange(800, dtype=float) + 1
                          for column in ['open', 'high', 'low', 'close', 'volume']},
                         index=dates)
    import m3_ml.kronos.dataset as dataset
    monkeypatch.setattr(dataset, 'load_corpus', lambda cutoff: {WTI_INPUT: frame})
    cutoff = dates[500]
    sampler = WindowSampler(dict(train_end=str(cutoff.date()), val_end=str(dates[-1].date())),
                            'val', seed=0)
    for start in sampler.starts[0]:
        assert dates[start + config.LOOKBACK] > cutoff
        assert dates[start + config.WINDOW - 1] <= dates[-1]

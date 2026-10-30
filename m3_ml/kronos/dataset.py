"""Training windows for Kronos (PLAN.md sections 3-5).

A window is WINDOW consecutive bars of one ticker. Features: open, high, low,
close, volume, amount (= volume x mean OHLC). Normalization uses the first
LOOKBACK rows only (never the future part), then clips to +-CLIP. Time stamps
are minute, hour (always 0 for daily bars), weekday, day, month.

Training windows end on or before train_end; validation windows end in
(train_end, val_end]. Nothing after val_end is ever read.
"""
import numpy as np
import pandas as pd

from .config import CLIP, ENERGY_WEIGHT, FEATURES, LOOKBACK, SAMPLING, WINDOW
from .corpus import load_corpus
from .universe import ENERGY, GROUP_A, point_in_time_sp500


def to_arrays(frame):
    """(bars x 6) features and (bars x 5) time stamps for one ticker."""
    x = frame[['open', 'high', 'low', 'close', 'volume']].to_numpy(dtype=np.float64)
    amount = x[:, 4] * x[:, :4].mean(axis=1)
    feats = np.column_stack([x, amount]).astype(np.float32)
    idx = frame.index
    stamps = np.column_stack([np.zeros(len(idx)), np.zeros(len(idx)), idx.weekday, idx.day, idx.month])
    return feats, stamps.astype(np.int64)


def normalize(window, stats=None):
    """Apply frozen per-ticker training statistics; context-local only if requested."""
    if stats is None:
        head = window[:LOOKBACK]
        stats = (head.mean(axis=0), head.std(axis=0))
    mean, std = stats
    return np.clip((window - mean) / (std + 1e-5), -CLIP, CLIP).astype(np.float32), (mean, std)


class WindowSampler:
    """Random windows for one stage and split ('train' or 'val')."""

    def __init__(self, stage_cfg, split, seed):
        if split == 'val' and pd.Timestamp(stage_cfg['val_end']) <= pd.Timestamp(stage_cfg['train_end']):
            raise ValueError('Final fit has no validation split; use historical stage-1 validation for selection.')
        corpus = load_corpus(stage_cfg['val_end'])
        lo = None if split == 'train' else pd.Timestamp(stage_cfg['train_end'])
        hi = pd.Timestamp(stage_cfg['train_end'] if split == 'train' else stage_cfg['val_end'])
        self.series, self.starts, self.weights, self.groups = [], [], [], []
        self.statistics = {}
        eligible = set(GROUP_A) | set(point_in_time_sp500(stage_cfg['train_end']))
        for ticker, frame in corpus.items():
            if ticker not in eligible:
                continue
            feats, stamps = to_arrays(frame)
            ends = frame.index[WINDOW - 1:]
            # The entire future label segment must belong to the validation
            # interval, not merely its last row (purge boundary-crossing labels).
            first_future = frame.index[LOOKBACK:len(frame) - (WINDOW - LOOKBACK) + 1]
            ok = (ends <= hi) if lo is None else ((first_future > lo) & (ends <= hi))
            starts = np.flatnonzero(ok)
            if not len(starts):
                continue
            fit = feats[frame.index <= pd.Timestamp(stage_cfg['train_end'])]
            if not len(fit):
                continue
            stats = (fit.mean(axis=0), fit.std(axis=0))
            self.statistics[ticker] = stats
            group = 'A' if ticker in GROUP_A else 'B'
            self.series.append((ticker, feats, stamps, frame.index, stats))
            self.starts.append(starts)
            self.groups.append(group)
            self.weights.append(ENERGY_WEIGHT if ticker in ENERGY else 1.0)
        self.rng = np.random.default_rng(seed)
        groups = np.array(self.groups)
        w = np.array(self.weights)
        p = np.zeros(len(w))
        for g, share in SAMPLING.items():
            mask = groups == g
            p[mask] = share * w[mask] / w[mask].sum()
        self.p = p / p.sum()

    def reseed(self, seed):
        self.rng = np.random.default_rng(seed)

    def sample(self, n):
        """n windows: normalized features (n, WINDOW, 6), stamps (n, WINDOW, 5), end dates."""
        xs, ts, ends = [], [], []
        for k in self.rng.choice(len(self.series), size=n, p=self.p):
            ticker, feats, stamps, dates, stats = self.series[k]
            i = int(self.rng.choice(self.starts[k]))
            xs.append(normalize(feats[i:i + WINDOW], stats)[0])
            ts.append(stamps[i:i + WINDOW])
            ends.append(dates[i + WINDOW - 1])
        return np.stack(xs), np.stack(ts), ends

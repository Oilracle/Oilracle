"""Publication-aware historical fitting and frozen October forecast origins."""
import numpy as np
import pandas as pd

from .data import label_available_dates
from .protocol import FIT_END, TEST_END, VALIDATION_END


def final_training_positions(dates, features_ok, target_ok, horizon, cutoff=FIT_END):
    """Samples whose entire target window was public under the evidence policy.

    Target windows are also purged at the cutoff; missing publication evidence
    is not an assumption of immediate availability.
    """
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    positions = np.arange(len(dates))
    endpoint = positions + horizon
    public = label_available_dates(dates).le(pd.Timestamp(cutoff))
    # Include the origin boundary: the first future squared return also
    # depends on its raw quote, even when a direct price target uses an anchor.
    available = public.astype(int).rolling(horizon + 1).sum().shift(-horizon).eq(horizon + 1)
    endpoint_dates = dates.reindex(endpoint).reset_index(drop=True)
    eligible = (available & endpoint_dates.le(pd.Timestamp(cutoff))).to_numpy()
    return positions[eligible & np.asarray(features_ok, dtype=bool) & np.asarray(target_ok, dtype=bool)]


def forecast_origins(dates, mode):
    """Fixed origin at the final fit; daily origins precede each target day."""
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    fixed = np.flatnonzero(dates.le(FIT_END))
    if not len(fixed):
        raise ValueError('table has no final fitting origin')
    origin = int(fixed[-1])
    if mode == 'fixed_origin':
        return np.array([origin])
    if mode == 'daily_update':
        return np.flatnonzero(dates.ge(dates.iloc[origin]) & dates.lt(TEST_END))
    raise ValueError(f'unknown forecast mode: {mode}')


def fixed_path_horizons(dates):
    """All observed October path points plus the rubric's key horizons."""
    from .protocol import HORIZONS
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    origin = forecast_origins(dates, 'fixed_origin')[0]
    realized = np.flatnonzero(dates.gt(FIT_END) & dates.le(TEST_END)) - origin
    return tuple(sorted(set(HORIZONS) | set(realized.tolist())))


def walk_forward(dates, features_ok, target_ok, start, end, horizon, refit_every):
    """Historical validation splits only; October is never a refitting period.

    A block starts with information available on its first origin. Every target
    used to score validation must separately end by VALIDATION_END.
    """
    if pd.Timestamp(end) > pd.Timestamp(VALIDATION_END):
        raise ValueError('walk_forward refits are restricted to historical validation')
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    origins = np.flatnonzero(dates.between(start, end) & np.asarray(features_ok))
    for i in range(0, len(origins), refit_every):
        refit = int(origins[i])
        train = final_training_positions(dates, features_ok, target_ok, horizon, dates.iloc[refit])
        yield refit, train, origins[i:i + refit_every]

"""Model-specific preprocessing fitted inside historical training windows.

The final Pipeline is fitted once with pre-October information. Neither its
scaler nor estimator may be refitted by daily state/history updates.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Clip(BaseEstimator, TransformerMixin):
    """Cap standardized values so that crash days (2020) do not dominate."""

    def __init__(self, limit=5.0):
        self.limit = limit

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):
        return True  # stateless

    def transform(self, X):
        return np.clip(X, -self.limit, self.limit)


def make_preprocessor(kind):
    """'scaled' for linear, kernel and neural models; 'raw' for tree models,
    which are invariant to monotone rescaling."""
    if kind == 'scaled':
        return make_pipeline(StandardScaler(), Clip())
    if kind == 'raw':
        return 'passthrough'
    raise ValueError(f'unknown preprocessor kind: {kind}')

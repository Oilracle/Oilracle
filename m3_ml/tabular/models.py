"""Tabular ML models and their small hyperparameter grids."""
from itertools import product

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline

from m3_ml.common.compute import bounded_cpu, cpu_threads, lightgbm_params
from m3_ml.common.preprocessing import make_preprocessor

SEED = 0


class RelativeElasticNet(BaseEstimator, RegressorMixin):
    """Elastic Net whose penalty is a fraction of alpha_max, the smallest alpha
    that sets every coefficient to zero in this training window.

    An absolute alpha means different things at different horizons: alpha_max
    grows with the target's scale (about 40x from h=1 to h=60 here), so
    a fixed grid zeroes the model at short horizons and barely regularizes at
    long ones. alpha_ratio=1 is the all-zero model (the training-mean forecast).
    """

    def __init__(self, alpha_ratio=0.1, l1_ratio=0.5):
        self.alpha_ratio = alpha_ratio
        self.l1_ratio = l1_ratio

    @bounded_cpu
    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        Xc = X - X.mean(axis=0)
        self.alpha_max_ = np.abs(Xc.T @ (y - y.mean())).max() / (len(y) * self.l1_ratio)
        self.model_ = ElasticNet(alpha=self.alpha_ratio * self.alpha_max_, l1_ratio=self.l1_ratio,
                                 max_iter=50000).fit(X, y)
        return self

    @bounded_cpu
    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))


def elastic_net(alpha_ratio, l1_ratio):
    return make_pipeline(make_preprocessor('scaled'),
                         RelativeElasticNet(alpha_ratio=alpha_ratio, l1_ratio=l1_ratio))


def random_forest(max_depth, min_samples_leaf):
    return make_pipeline(make_preprocessor('raw'), RandomForestRegressor(
        n_estimators=200, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        max_features=0.33, n_jobs=cpu_threads(), random_state=SEED))


def lightgbm(num_leaves, n_estimators):
    return make_pipeline(make_preprocessor('raw'), LGBMRegressor(
        num_leaves=num_leaves, n_estimators=n_estimators, learning_rate=0.02,
        min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=SEED, verbose=-1, **lightgbm_params()))


def grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, values)) for values in product(*axes.values())]


# Low-signal targets: strong regularization, shallow trees, few combinations.
MODELS = {
    'elastic_net': (elastic_net, grid(alpha_ratio=[1.0, 0.3, 0.1, 0.03, 0.01], l1_ratio=[0.1, 0.5, 0.9])),
    'random_forest': (random_forest, grid(max_depth=[3, 6], min_samples_leaf=[20, 100])),
    'lightgbm': (lightgbm, grid(num_leaves=[4, 8], n_estimators=[100, 300])),
}

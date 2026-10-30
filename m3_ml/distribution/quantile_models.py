"""Route B: forecast the quantiles of the h-day log return directly."""
import warnings

import numpy as np
from lightgbm import LGBMRegressor
from quantile_forest import RandomForestQuantileRegressor
from sklearn.linear_model import QuantileRegressor

from m3_ml.common.compute import bounded_cpu, cpu_threads, lightgbm_params
from m3_ml.common.preprocessing import make_preprocessor

from .quantiles import TAUS

SEED = 0


@bounded_cpu
def predict_b_linear(inp, h, refit, train, origins, alpha, features, taus=TAUS):
    """One L1-penalized linear quantile regression per level. Features are
    standardized and clipped inside the training window; the target is divided
    by its training standard deviation so that alpha means the same at every h."""
    X = inp.features[features]
    pre = make_preprocessor('scaled').fit(X.iloc[train])
    Xt, Xo = pre.transform(X.iloc[train]), pre.transform(X.iloc[origins])
    y = inp.y[h].iloc[train].to_numpy()
    sd = y.std()
    out = np.empty((len(origins), len(taus)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for j, tau in enumerate(taus):
            if j == 0 or (j + 1) % 10 == 0 or j + 1 == len(taus):
                print(f'distribution method=b_linear h={h} refit={refit} '
                      f'quantile={j + 1}/{len(taus)} tau={tau:g}', flush=True)
            model = QuantileRegressor(quantile=tau, alpha=alpha, solver='highs',
                                      solver_options={'threads': cpu_threads(), 'parallel': False}).fit(Xt, y / sd)
            out[:, j] = model.predict(Xo) * sd
    return out


def predict_b_lightgbm(inp, h, refit, train, origins, num_leaves, n_estimators, features, taus=TAUS):
    """One quantile-loss LightGBM per level."""
    X = inp.features[features]
    y = inp.y[h].iloc[train]
    out = np.empty((len(origins), len(taus)))
    for j, tau in enumerate(taus):
        if j == 0 or (j + 1) % 10 == 0 or j + 1 == len(taus):
            print(f'distribution method=b_lightgbm h={h} refit={refit} '
                  f'quantile={j + 1}/{len(taus)} tau={tau:g}', flush=True)
        model = LGBMRegressor(objective='quantile', alpha=tau, num_leaves=num_leaves, n_estimators=n_estimators,
                              learning_rate=0.03, min_child_samples=50, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=SEED, verbose=-1, **lightgbm_params())
        out[:, j] = model.fit(X.iloc[train], y).predict(X.iloc[origins])
    return out


def predict_b_forest(inp, h, refit, train, origins, min_samples_leaf, max_features, features, taus=TAUS):
    """Quantile regression forest: one fit gives every level."""
    X = inp.features[features]
    model = RandomForestQuantileRegressor(n_estimators=300, min_samples_leaf=min_samples_leaf,
                                          max_features=max_features, n_jobs=cpu_threads(), random_state=SEED)
    model.fit(X.iloc[train], inp.y[h].iloc[train])
    return np.asarray(model.predict(X.iloc[origins], quantiles=list(taus)))


QUANTILE_MODELS = {
    'b_linear': (predict_b_linear, [{'alpha': a, 'features': f} for a in (0.001, 0.01, 0.1)
                                    for f in ('vol', 'base_vol', 'base_vol_news')]),
    'b_lightgbm': (predict_b_lightgbm, [{'num_leaves': l, 'n_estimators': n, 'features': f}
                                        for l in (4, 8) for n in (100, 300)
                                        for f in ('base_vol', 'base_vol_news')]),
    'b_forest': (predict_b_forest, [{'min_samples_leaf': m, 'max_features': x, 'features': f}
                                    for m in (20, 50, 100) for x in (0.33, 1.0)
                                    for f in ('base_vol', 'base_vol_news')]),
}

"""Route A: forecast h-day volatility, then attach a calibrated distribution shape.

1. A model predicts log(sum of the next h squared returns).
2. Inside the training window, purged blocked cross-fitting gives an
   out-of-sample volatility for every training sample; the empirical
   quantiles of z = y / sigma_oos are the distribution shape (this also
   absorbs any scale bias of the model).
3. Forecast quantiles = sigma_hat(origin) x quantiles of z.
"""
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from m3_ml.common.compute import bounded_cpu, lightgbm_params
from m3_ml.common.preprocessing import make_preprocessor

from .data import log_volatility
from .quantiles import TAUS
from .tails import fit_gpd_tails, semiparametric_ppf

FOLDS = 5
EMBARGO = 60  # longest feature lookback: later features would contain the block's returns
SEED = 0


def ridge(alpha):
    return make_pipeline(make_preprocessor('scaled'), Ridge(alpha=alpha))


def lightgbm(num_leaves, n_estimators):
    return LGBMRegressor(num_leaves=num_leaves, n_estimators=n_estimators, learning_rate=0.02,
                         min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                         random_state=SEED, verbose=-1, **lightgbm_params())


def purged_folds(train, h, folds=FOLDS):
    """Contiguous blocks of the (sorted) training positions. For each block,
    the fitting positions exclude every sample whose h-day target window
    overlaps the block's target windows (purge), and the samples right after
    the block whose features still look back into its returns (embargo)."""
    train = np.sort(np.asarray(train))
    for block in np.array_split(train, folds):
        a, b = block[0], block[-1]
        fit = train[(train <= a - h) | (train >= b + h + EMBARGO)]
        yield fit, block


@bounded_cpu
def scale_quantiles(make_model, X, inp, h, train, origins, taus=TAUS, tail=None):
    """sigma_hat(origin) x quantiles of the cross-fitted z; with `tail` (a tail
    fraction) the z distribution is an empirical body with GPD tails (A-EVT)."""
    vol_target = inp.vol_target[h]
    sigma_oos = np.empty(len(train))
    order = {p: i for i, p in enumerate(np.sort(train))}
    for fold, (fit, block) in enumerate(purged_folds(train, h), 1):
        print(f'distribution scale h={h} crossfit={fold}/{FOLDS}', flush=True)
        model = make_model().fit(X.iloc[fit], vol_target.iloc[fit])
        sigma_oos[[order[p] for p in block]] = np.sqrt(np.exp(model.predict(X.iloc[block])))
    z = inp.y[h].iloc[np.sort(train)].to_numpy() / sigma_oos
    z_q = np.quantile(z, taus) if tail is None else semiparametric_ppf(fit_gpd_tails(z, tail), np.asarray(taus))
    model = make_model().fit(X.iloc[train], vol_target.iloc[train])
    sigma = np.sqrt(np.exp(model.predict(X.iloc[origins])))
    return sigma[:, None] * z_q[None, :]


def predict_a_ridge(inp, h, refit, train, origins, alpha, features, taus=TAUS, tail=None):
    X = log_volatility(inp.features[features])
    return scale_quantiles(lambda: ridge(alpha), X, inp, h, train, origins, taus, tail)


def predict_a_lightgbm(inp, h, refit, train, origins, num_leaves, n_estimators, features, taus=TAUS, tail=None):
    return scale_quantiles(lambda: lightgbm(num_leaves, n_estimators), inp.features[features],
                           inp, h, train, origins, taus, tail)


def predict_a_evt(inp, h, refit, train, origins, tail, base_method, base_params, taus=TAUS):
    """Route A with GPD tails; the volatility model is fixed by validation (see run.resolve)."""
    base = predict_a_ridge if base_method == 'a_ridge' else predict_a_lightgbm
    return base(inp, h, refit, train, origins, **base_params, taus=taus, tail=tail)


SCALE_MODELS = {
    'a_ridge': (predict_a_ridge, [{'alpha': a, 'features': f} for a in (0.1, 1, 10, 100)
                                  for f in ('vol', 'base_vol', 'base_vol_news')]),
    'a_lightgbm': (predict_a_lightgbm, [{'num_leaves': l, 'n_estimators': n, 'features': f}
                                        for l in (4, 8, 16) for n in (200, 500)
                                        for f in ('base_vol', 'base_vol_news')]),
    'a_evt': (predict_a_evt, [{'tail': f} for f in (0.05, 0.10)]),
}

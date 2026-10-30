"""Classical models with frozen coefficients and publication-causal state updates.

Run: uv run python -m m3_ml.compare.part1
ARIMA-GARCH remains the documented two-step approximation, not R's joint fit.
"""
import json

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.vecm import VECM

from m3_ml.common.data import label_available_dates, observable_price
from m3_ml.distribution.tails import ALL_COLUMNS, ALL_TAUS

from .common import (
    FIT_END, MODES, OUT, PROVENANCE, TRAIN_END, VALIDATION_END,
    VALIDATION_START, forecast_horizons, forecast_origins, forecast_row, load_prices,
)

SIMULATIONS = 5000
SEED = 20261025
PRICE_FLOOR = 0.01
VECM_CANDIDATES = {
    'WTI_Brent_K4_R1': (['target_wti', 'brent_spot'], 4, 1),
    'WTI_Brent_K5_R1': (['target_wti', 'brent_spot'], 5, 1),
    'WTI_Brent_Gasoline_K3_R2': (['target_wti', 'brent_spot', 'ny_gasoline_spot'], 3, 2),
    'WTI_Brent_Gasoline_K4_R2': (['target_wti', 'brent_spot', 'ny_gasoline_spot'], 4, 2),
    'WTI_Brent_Heating_K4_R2': (['target_wti', 'brent_spot', 'ny_heating_oil_spot'], 4, 2),
    'WTI_Brent_Heating_K5_R2': (['target_wti', 'brent_spot', 'ny_heating_oil_spot'], 5, 2),
    'WTI_All4_K3_R3': (['target_wti', 'brent_spot', 'ny_gasoline_spot', 'ny_heating_oil_spot'], 3, 3),
    'WTI_All4_K4_R3': (['target_wti', 'brent_spot', 'ny_gasoline_spot', 'ny_heating_oil_spot'], 4, 3),
}


def available_prefix(available, date):
    """Retain the original quoted-row grid; never compress missing publications."""
    known = available.notna() & available.le(pd.Timestamp(date))
    missing = np.flatnonzero(~known.to_numpy())
    end = int(missing[0] - 1) if len(missing) else len(known) - 1
    if end < 100:
        raise ValueError(f'Insufficient contiguous published history by {date}; check publication evidence')
    return end


def fit_arima(y, sp=None):
    result = ARIMA(y, exog=sp, order=(0, 1, 1), trend='n').fit(method_kwargs={'maxiter': 500})
    if not result.mle_retvals.get('converged', False):
        raise RuntimeError('Frozen ARIMA parameter fit did not converge')
    return result


def fit_vecm(values, k, rank):
    return VECM(values, k_ar_diff=k - 1, coint_rank=rank, deterministic='ci').fit()


def vecm_path(result, history, steps):
    """Advance observed levels under the original frozen VECM's VAR representation."""
    coefficients = result.var_rep
    # statsmodels stores the restricted constant as (1, rank), not (rank, 1).
    intercept = (result.alpha @ result.const_coint.T).reshape(-1)
    p = len(coefficients)
    states = [row.copy() for row in history[-p:]]
    forecasts = np.empty((steps, history.shape[1]))
    for j in range(steps):
        value = intercept.copy()
        for lag, coefficient in enumerate(coefficients, 1):
            value += coefficient @ states[-lag]
        states.append(value)
        forecasts[j] = value
    return forecasts[:, 0]


def select_vecm(table, spot, available):
    """Train-only estimates; candidate choice uses historical validation targets only."""
    train_end = available_prefix(available, TRAIN_END)
    dates = table['date']
    y = table['target_wti'].to_numpy()
    anchors = observable_price(table).to_numpy()
    origins = np.flatnonzero(dates.between(VALIDATION_START, VALIDATION_END).to_numpy())
    last = int(np.flatnonzero(dates.le(VALIDATION_END).to_numpy())[-1])
    rows = []
    for name, (columns, k, rank) in VECM_CANDIDATES.items():
        values = spot[columns].to_numpy()
        result = fit_vecm(values[:train_end + 1], k, rank)
        errors = []
        for origin in origins:
            end = available_prefix(available, dates.iloc[origin])
            horizons = np.arange(1, min(20, last - origin) + 1)
            if not len(horizons):
                continue
            prediction = vecm_path(result, values[:end + 1], origin + int(horizons[-1]) - end)
            for h in horizons:
                target = origin + int(h)
                errors.append({'year': int(dates.iloc[target].year), 'horizon': int(h),
                               'squared_error': (prediction[target - end - 1] - y[target]) ** 2,
                               'rw_squared_error': (anchors[origin] - y[target]) ** 2})
        errors = pd.DataFrame(errors)
        for year, group in errors.groupby('year'):
            sums = group.groupby('horizon')[['squared_error', 'rw_squared_error']].mean()
            ratio = np.sqrt(sums['squared_error'] / sums['rw_squared_error'])
            rows.append({'model': name, 'year': int(year), 'n': len(group),
                         'mean_relative_rmse': float(ratio.mean())})
    scores = pd.DataFrame(rows)
    summary = scores.groupby('model')['mean_relative_rmse'].apply(lambda v: float(np.exp(np.log(v).mean()))).rename('score').reset_index()
    if summary.empty or not np.isfinite(summary['score']).all():
        raise ValueError('VECM historical validation did not produce finite candidate scores')
    return scores, summary.sort_values('score')


def garch_quantiles(parameters, variance, last_residual, theta, last_price, steps, rng):
    """Simulate frozen GARCH-t innovations and MA(1), including the publication gap."""
    omega, alpha, beta, nu = (parameters[name] for name in ('omega', 'alpha[1]', 'beta[1]', 'nu'))
    sigma2 = np.full(SIMULATIONS, variance)
    residual = np.full(SIMULATIONS, last_residual)
    price = np.full(SIMULATIONS, last_price)
    paths = np.empty((SIMULATIONS, steps))
    for j in range(steps):
        sigma2 = omega + alpha * residual ** 2 + beta * sigma2
        innovation = rng.standard_t(nu, SIMULATIONS) * np.sqrt((nu - 2) / nu * sigma2)
        price = price + innovation + theta * residual
        paths[:, j] = price
        residual = innovation
    return np.quantile(paths, ALL_TAUS, axis=0).T


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table, spot = load_prices()
    available = pd.to_datetime(label_available_dates(table))
    anchors = observable_price(table).to_numpy()
    y = table['target_wti'].to_numpy()
    sp = table['sp500_close'].to_numpy()
    fit_row = int(forecast_origins(table, 'fixed_origin')[0])
    fit_end = available_prefix(available, FIT_END)
    scores, summary = select_vecm(table, spot, available)
    scores.to_csv(OUT / 'vecm_selection_by_year.csv', index=False)
    summary.to_csv(OUT / 'vecm_selection.csv', index=False)
    chosen = summary['model'].iloc[0]
    columns, k, rank = VECM_CANDIDATES[chosen]
    values = spot[columns].to_numpy()
    vecm = fit_vecm(values[:fit_end + 1], k, rank)
    arima_fit = fit_arima(y[:fit_end + 1])
    arimax_fit = fit_arima(y[:fit_end + 1], sp[:fit_end + 1, None])
    residuals = np.asarray(arima_fit.resid)[1:]
    garch = arch_model(residuals, mean='Zero', vol='GARCH', p=1, q=1, dist='t', rescale=False).fit(disp='off')
    if garch.convergence_flag:
        raise RuntimeError('Frozen ARIMA-GARCH parameter fit did not converge')
    theta = float(arima_fit.params[arima_fit.param_names.index('ma.L1')])
    points, quantiles = [], []
    for mode in MODES:
        arima, arimax = arima_fit, arimax_fit
        state_end = fit_end
        variance = float(np.asarray(garch.conditional_volatility)[-1] ** 2)
        previous_residual = float(np.asarray(arima.resid)[-1])
        for origin in forecast_origins(table, mode):
            known = fit_end if mode == 'fixed_origin' else available_prefix(available, table['date'].iloc[origin])
            if known > state_end:
                arima = arima.append(y[state_end + 1:known + 1], refit=False)
                arimax = arimax.append(y[state_end + 1:known + 1], exog=sp[state_end + 1:known + 1, None], refit=False)
                for residual in np.asarray(arima.resid)[state_end + 1:known + 1]:
                    variance = float(garch.params['omega'] + garch.params['alpha[1]'] * previous_residual ** 2 + garch.params['beta[1]'] * variance)
                    previous_residual = float(residual)
                state_end = known
            horizons = forecast_horizons(table, origin, mode)
            steps = int(origin + max(horizons) - known)
            fc = arima.get_forecast(steps)
            # No future S&P observations: freeze at the last published model-state quote.
            xfc = arimax.get_forecast(steps, exog=np.full((steps, 1), sp[known]))
            vfc = vecm_path(vecm, values[:known + 1], steps)
            gq = garch_quantiles(garch.params, variance, previous_residual, theta, y[known], steps,
                                 np.random.default_rng(SEED + int(origin)))
            anchor = float(anchors[origin])
            if not np.isfinite(anchor) or anchor <= 0:
                raise ValueError(f'No positive observable anchor at origin {table["date"].iloc[origin]}')
            for h in horizons:
                j = int(origin + h - known - 1)
                for method, path in (('p1_arima', fc.predicted_mean), ('p1_arimax', xfc.predicted_mean), ('p1_vecm', vfc)):
                    row = forecast_row(table, method, mode, origin, h, anchor)
                    row.update(forecast=float(np.asarray(path)[j]), state_end_date=table['date'].iloc[known])
                    points.append(row)
                gaussian = float(np.asarray(fc.predicted_mean)[j]) + stats.norm.ppf(ALL_TAUS) * float(np.asarray(fc.se_mean)[j])
                for method, price_q in (('p1_arima', gaussian), ('p1_arima_garch', gq[j])):
                    row = forecast_row(table, method, mode, origin, h, anchor)
                    logq = np.log(np.maximum(price_q, PRICE_FLOOR) / anchor)
                    row.update(dict(zip(ALL_COLUMNS, logq)))
                    row['actual_log_return'] = np.log(row['actual'] / anchor) if pd.notna(row['actual']) and row['actual'] > 0 else np.nan
                    row['forecast'] = anchor * np.exp(np.interp(0.5, ALL_TAUS, logq))
                    row['state_end_date'] = table['date'].iloc[known]
                    quantiles.append(row)
    pd.DataFrame(points).to_csv(OUT / 'part1_forecasts.csv', index=False, date_format='%Y-%m-%d')
    pd.DataFrame(quantiles).to_csv(OUT / 'part1_quantiles.csv', index=False, date_format='%Y-%m-%d')
    metadata = {**PROVENANCE, 'selected_vecm': chosen, 'fit_origin_row': fit_row,
                'parameter_history_end': str(table['date'].iloc[fit_end].date()),
                'simulations': SIMULATIONS, 'seed': SEED, 'coefficient_refits_in_test': 0,
                'availability_limitation': 'WTI evidence plus explicitly assumed weekly releases for unmatched quotes; multivariate spot uses common WTI schedule assumption; retrospective current-vintage comparison',
                'garch_estimation': 'two-step ARIMA mean then GARCH(1,1)-t; not joint R estimation'}
    (OUT / 'part1_protocol.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print('wrote frozen classical forecasts to', OUT)


if __name__ == '__main__':
    main()

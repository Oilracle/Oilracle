# Residual correction: current protocol and archived results

## Current frozen October experiment

Initial training ends August 31, 2023; historical validation runs September 1,
2023–September 30, 2026. Final seasonal ARIMA, residual ML, scale regressions,
shock distributions and EWMA selections use only information published by
September 30. October updates states and causal features, not coefficients or
preprocessing. The current artifacts have been generated under
`outputs/frozen_october_2026/`; the older root outputs below remain historical.

The module supports fixed-origin full October quote-row paths and rolling
h1/5/20/60; the final product uses rolling mode. Non-core fixed-path horizons
inherit the next larger core horizon's validation setting, then fit on their own
labels. ARIMA innovations may absorb newly released history, while seasonal/MA
coefficients stay fixed. Missing unreleased quote rows remain in the forecast
distance. `observable_price(table)` uses the latest released observation date;
late older quotes cannot reverse the anchor.

Publication timing follows `data/processed/wti_publication.csv`: archive evidence
where matched and a labeled weekly assumption elsewhere. This is retrospective,
not complete real-time vintage evidence. Beyond observed rows, weekday scenarios
can define seasonal terms, but actual target dates remain missing.

Outputs include `residual_forecasts.csv`, `residual_quantiles.csv`,
`residual_params.json`, `point_metrics.csv` and `risk_metrics.csv`. Metrics score
realized October 1–23 targets separately by mode/horizon; rolling h1/5/20/60
counts are 17/13/0/0. They are marked `inference=descriptive_short_window` and
do not claim DM/HAC significance from a few overlapping outcomes.

```bash
uv run python -m m3_ml.residual.run
uv run python -m m3_ml.residual.analyze
```

## Archived January 2023–October 6, 2026 experiment

These scores used 2020–2022 selection and refits every 21 rows. Current commands
do not reproduce that retired protocol. Residual ML did not significantly
improve point predictions over time series plus seasonality: all models were
worse at h5, and both tree models were worse at h20. The baseline itself lost to
random walk at h1/h5/h20; driftless ARIMA(0,1,1) was essentially tied with RW.
Elastic Net selected zero feature coefficients at h1/h5/h20 and a nonzero h60
model that still lost 1.4% to baseline (p = 0.74).

### Point forecasts

MAE is USD/barrel; the RMSE ratios/DM statistics use archived log-return errors.
DM is two-sided with Newey–West lag h−1. See `outputs/point_metrics.csv` and
`outputs/figures/point_rmse.png`.

| h | Model | MAE (USD) | RMSE / RW | RMSE / Baseline | DM vs RW | DM vs Baseline |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Random walk | 1.42 | 1.000 | 0.990 | — | 0.004 |
| 1 | ARIMA(0,1,1) | 1.42 | 1.001 | 0.990 | 0.72 | 0.001 |
| 1 | Seasonality | 1.43 | 1.010 | 0.999 | 0.002 | 0.61 |
| 1 | Time series + seasonality | 1.44 | 1.010 | 1.000 | 0.004 | — |
| 1 | Baseline + Elastic Net | 1.44 | 1.011 | 1.000 | 0.004 | 0.39 |
| 1 | Baseline + Random Forest | 1.44 | 1.015 | 1.004 | 0.002 | 0.14 |
| 1 | Baseline + LightGBM | 1.44 | 1.017 | 1.007 | 0.005 | 0.099 |
| 5 | Random walk | 3.23 | 1.000 | 0.967 | — | 0.002 |
| 5 | ARIMA(0,1,1) | 3.23 | 1.000 | 0.968 | 0.73 | 0.002 |
| 5 | Seasonality | 3.33 | 1.033 | 1.000 | 0.002 | 0.51 |
| 5 | Time series + seasonality | 3.33 | 1.034 | 1.000 | 0.002 | — |
| 5 | Baseline + Elastic Net | 3.33 | 1.035 | 1.001 | 0.002 | 0.013 |
| 5 | Baseline + Random Forest | 3.43 | 1.059 | 1.025 | 0.0003 | 0.008 |
| 5 | Baseline + LightGBM | 3.39 | 1.055 | 1.020 | 0.0001 | 0.008 |
| 20 | Random walk | 5.71 | 1.000 | 0.921 | — | 0.010 |
| 20 | ARIMA(0,1,1) | 5.71 | 1.000 | 0.921 | 0.95 | 0.009 |
| 20 | Seasonality | 6.20 | 1.085 | 1.000 | 0.009 | 0.77 |
| 20 | Time series + seasonality | 6.20 | 1.086 | 1.000 | 0.010 | — |
| 20 | Baseline + Elastic Net | 6.21 | 1.089 | 1.003 | 0.008 | 0.060 |
| 20 | Baseline + Random Forest | 6.53 | 1.137 | 1.047 | 0.0006 | 0.007 |
| 20 | Baseline + LightGBM | 6.59 | 1.143 | 1.053 | 0.0004 | 0.009 |
| 60 | Random walk | 9.67 | 1.000 | 0.940 | — | 0.13 |
| 60 | ARIMA(0,1,1) | 9.67 | 1.000 | 0.940 | 0.12 | 0.13 |
| 60 | Seasonality | 10.86 | 1.064 | 1.000 | 0.13 | 0.086 |
| 60 | Time series + seasonality | 10.86 | 1.063 | 1.000 | 0.13 | — |
| 60 | Baseline + Elastic Net | 10.30 | 1.078 | 1.014 | 0.18 | 0.74 |
| 60 | Baseline + Random Forest | 11.08 | 1.149 | 1.080 | 0.022 | 0.14 |
| 60 | Baseline + LightGBM | 11.63 | 1.140 | 1.072 | 0.045 | 0.067 |

At h60, USD MAE and log-return RMSE rank models differently: Elastic Net lowers
price MAE but has 1.4% higher return RMSE, without a significant DM difference.

### Distribution and risk forecasts

CRPS approximates the score on 23 quantiles. Moving the residual distribution's
center did not reliably improve the baseline and remained worse than EWMA at
h1/h5/h20. At h1 scale correction improved baseline CRPS but still lost to EWMA.
At h20 it reduced nominal 90% coverage to 71.5%. See `outputs/risk_metrics.csv`
and `outputs/figures/risk_crps.png`.

| h | Method | CRPS / EWMA | CRPS / Baseline | 90% coverage | DM vs EWMA | DM vs Baseline |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | EWMA | 1.000 | 0.968 | 0.895 | — | <0.001 |
| 1 | Time series + seasonality | 1.033 | 1.000 | 0.901 | <0.001 | — |
| 1 | Baseline + Elastic Net | 1.033 | 1.000 | 0.899 | <0.001 | 0.51 |
| 1 | Baseline + Random Forest | 1.036 | 1.003 | 0.899 | <0.001 | 0.26 |
| 1 | Baseline + LightGBM | 1.039 | 1.006 | 0.896 | <0.001 | 0.091 |
| 1 | LightGBM + scale | 1.018 | 0.985 | 0.901 | <0.001 | 0.017 |
| 5 | EWMA | 1.000 | 0.946 | 0.891 | — | <0.001 |
| 5 | Time series + seasonality | 1.057 | 1.000 | 0.877 | <0.001 | — |
| 5 | Baseline + Elastic Net | 1.058 | 1.001 | 0.877 | <0.001 | 0.019 |
| 5 | Baseline + Random Forest | 1.085 | 1.026 | 0.870 | <0.001 | 0.006 |
| 5 | Baseline + LightGBM | 1.078 | 1.020 | 0.871 | <0.001 | 0.015 |
| 5 | LightGBM + scale | 1.060 | 1.003 | 0.877 | <0.001 | 0.76 |
| 20 | EWMA | 1.000 | 0.906 | 0.941 | — | 0.008 |
| 20 | Time series + seasonality | 1.104 | 1.000 | 0.905 | 0.008 | — |
| 20 | Baseline + Elastic Net | 1.107 | 1.003 | 0.904 | 0.007 | 0.077 |
| 20 | Baseline + Random Forest | 1.156 | 1.047 | 0.890 | <0.001 | 0.012 |
| 20 | Baseline + LightGBM | 1.167 | 1.058 | 0.889 | <0.001 | 0.010 |
| 20 | LightGBM + scale | 1.232 | 1.116 | 0.715 | <0.001 | 0.001 |
| 60 | EWMA | 1.000 | 0.917 | 0.894 | — | 0.18 |
| 60 | Time series + seasonality | 1.091 | 1.000 | 0.914 | 0.18 | — |
| 60 | Baseline + Elastic Net | 1.072 | 0.983 | 0.906 | 0.25 | 0.70 |
| 60 | Baseline + Random Forest | 1.141 | 1.046 | 0.881 | 0.025 | 0.40 |
| 60 | Baseline + LightGBM | 1.139 | 1.044 | 0.892 | 0.075 | 0.28 |
| 60 | LightGBM + scale | 1.139 | 1.044 | 0.892 | 0.075 | 0.28 |

At h60, validation reduced the scale model to a constant, so its output equals
the center-only LightGBM variant. These negative results remain part of the
research record and must not be substituted for current frozen-window scores.

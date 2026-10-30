# Risk forecasting results and historical archive

## Current frozen evaluation

The current runs are complete under `outputs/frozen_october_2026/`. Initial training ends August 2023; historical development ends September 2026; final fits use September 30 available information. October coefficients/preprocessing remain frozen. Each horizon fits once for both forecast modes; intermediate fixed-path horizons inherit the next larger key-horizon hyperparameters but fit their own labels. Event thresholds interpolate the prespecified key thresholds.

WTI state uses the contiguous published prefix without compressing quote time. Historical/EWMA/GARCH update state only. Futures bars follow the implemented observation cutoff. Archive evidence and assumed weekly releases remain distinct. Score only realized October target dates; long-horizon missing actuals remain empty. `inference=descriptive_short_window` suppresses October DM/HAC/Wald p-values and winner flags. Historical inference must not be borrowed as October significance evidence.

Run distribution validation/final prediction first, then `python -m m3_ml.risk.volatility validate`, `test`, corresponding `events` commands, and `python -m m3_ml.risk.analyze`. Outputs include volatility/event metrics, volatility-targeting and VaR-hedging tables, and mode-specific figures.

## Legacy experiment

The following values describe the old 2023-01-03–2026-10-06 monthly-refit test, not October-only scores. Root-level `outputs/` preserves them. Selection records were `cf03a75` and `36039a2`. Current commands execute the new protocol and must not be described as reproducing these old numbers.

### Volatility

| Horizon | Best QLIKE | HAR | Calibrated OVX | GARCH |
| --- | --- | ---: | ---: | ---: |
| 1 | HAR 1.367 | 1.367 | 1.371 | 1.421 |
| 5 | Calibrated OVX .314 | .322 | .314 | .390 |
| 20 | Raw OVX .177 | .206 | .185 | .258 |
| 60 | Calibrated OVX .189 | .224 | .189 | .237 |

Lower is better. Under the original rule, HAR beats GARCH at h1/h5 (DM p=.045/.036), using close and Garman–Klass inputs rather than implied volatility. No model meets the rule for beating calibrated OVX. HAR's validation advantage at h5/h20 does not persist. At h20, OVX Mincer–Zarnowitz R² is .57 versus HAR .45, with slope near one. These findings apply to this sample, not a universal efficiency claim.

### Events

Down-event counts are 79/77/34/27 and up counts 73/83/58/80. Many h1/h5 upside probabilities have 5–9% Brier skill with p<.05; A-Ridge h5 has 8.6%, p=.03. Downside significance is limited mostly to h1 (LightGBM classifier 2.6%, p=.026). No significant h20/h60 skill is found; high-probability bins miscalibrate during the 2026 shock. Direct classifiers do not outperform distribution-derived probabilities. Reuse of existing distribution-test predictions is post-test analysis, not new independent evidence.

### Exposure control

The target is 30% annual volatility with 5bp turnover costs.

| Rule | Annual volatility | Mean annual deviation from target, pp | Worst day | Daily 97.5% ES | Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unscaled w=1 | 40.7% | 10.3 | −17.5% | 7.8% | .15 |
| Ex-ante fixed | 22.9% | 10.9 | −9.9% | 4.4% | .15 |
| GARCH | 31.2% | 2.4 | −9.2% | 5.3% | .09 |
| HAR | 29.4% | 1.7 | −7.9% | 4.8% | .04 |
| Raw OVX | 27.5% | 2.7 | −7.6% | 4.6% | .05 |

Risk targeting reduces volatility deviations and worst daily losses but does not improve Sharpe. Reducing exposure during the high-volatility 2026 rise misses gains. The fixed rule uses 53% training volatility inflated by 2020, hence its lower realized exposure. The Moreira–Muir (2017) Sharpe motivation is not replicated here.

Producer VaR-limit exceedance is 1.6–2.7% against a 5% target, with 35–46% mean hedge share, indicating conservative lower tails. Consumer exceedance is 3.9–7.6%; GARCH-EVT is closest at 5.3% with 28.6% mean hedge share, versus 13% unhedged. Dynamic hedges adjust dozens of times per year but do not clearly beat fixed historical-simulation hedges in this sample.

The old test is about 3.8 years with few independent long-horizon windows. Spot-return exposure and spot-based hedging omit executable futures costs and roll returns; these are decision illustrations, not validated investment returns.

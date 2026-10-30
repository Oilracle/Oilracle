# Market residual fusion rolling research archive

This records `market_residual_v3`, not the current final model; see [v6](FUSION_V6.md). On 2026-10-30 the delivery scope changed to daily rolling forecasts at h1/5/20/60. The scope decision followed inspection of October, but changed no fitted parameters or sealed predictions. `rolling/delivery_scope.json` records it; the [pre-scope archive](FUSION_V3_PRE_ROLLING_ARCHIVE.md) preserves the unfavorable fixed path. The narrowed scope is not a new blind test.

At 00:00 New York time, forecasts use published inputs and frozen September coefficients, preprocessing, weights, and risk calibration. The rolling view has 68 rows (17 origins × four horizons), each with a price median, 63 quantiles, intervals, probabilities, and tail risk. Unobserved targets retain forecasts without fabricated actuals.

## Components and information

Actual M1 VECM, M2 monthly STL, three-seed LSTM, and original ML point forecasts feed residual features and combinations. Ridge, Elastic Net, and shallow LightGBM learn dollar residuals with calendar/state interactions. Risk chooses conditional scale/shape and calibrates fusion errors while preserving its median. There are 25 point candidates, eight combination rules, and 252 risk candidates; no member floors. h1/h20 use soft8, h5 soft32, and h60 best, locked using mature errors before 2025-09-01 and fitted at the final September cutoff.

Nearby WTI `CL=F` Close enters after two US federal business days. Basis aligns spot and futures on the same spot observation date and becomes usable only when both are published. Spot RW and futures hold are separate baselines. All four member signals have nonzero local sensitivity somewhere; this is evidence of actual use, not necessarily better accuracy or a positive scalar weight.

## Historical rolling evidence

| Horizon | n | v3 RMSE, USD | Spot RW RMSE | Futures hold RMSE | Improvement over spot RW |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 261 | 5.0700 | 7.9918 | 5.1788 | **36.6%** |
| 5 | 257 | 7.9234 | 9.7375 | 7.8619 | **18.6%** |
| 20 | 242 | 14.0127 | 14.8045 | 13.5973 | **5.3%** |
| 60 | 202 | 25.3211 | 25.1294 | 24.3050 | **−0.8%** |

The gain largely comes from more timely futures information. h5/h20/h60 do not beat futures hold. Inherited base specifications were already developed on the broader history; inference is conditional and exploratory. Risk forward selection begins March 2026:

| Horizon | n | CRPS / HS | CRPS / v2 | 90% interval coverage |
| --- | ---: | ---: | ---: | ---: |
| 1 | 139 | **0.5732** | 0.6121 | 79.14% |
| 5 | 135 | **0.7418** | 0.7773 | 83.70% |
| 20 | 120 | **0.9404** | 0.9536 | 81.67% |
| 60 | 80 | **1.3208** | 1.1181 | 56.25% |

Short-horizon CRPS improves, but nominal coverage remains unmet and h60 risk is weak.

## October rolling evaluation

| Horizon | n | v3 RMSE, USD | v2 RMSE | Spot RW RMSE | Futures hold RMSE | Improvement over spot RW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 17 | 6.6272 | 6.4710 | 9.1101 | 8.3154 | **27.3%** |
| 5 | 13 | 10.7855 | 6.2130 | 11.2532 | 12.4553 | **4.2%** |
| 20 | 0 | — | — | — | — | — |
| 60 | 0 | — | — | — | — | — |

h1/h5 improve about 20.3%/13.4% over futures hold but do not beat v2. October does not choose a fallback version. The sample is short and already viewed; h20/h60 have no realized targets.

## Artifacts and checks

`m3_ml/fusion_v3/outputs/frozen_october_2026/rolling/` contains forecasts, metrics, historical point/risk tables, inference, weights, sensitivity, figures, and delivery audit. The selected hash is `12d803737e1804b2e69312aede92113535b38f9f7fee743c9b903ecc409c209a`. The scope-only audit matches every rolling row to sealed forecasts and checks futures publication dates. A 3,174-observation EIA futures overlap agrees within tolerance but does not validate all later vendor data. Current vintages and assumed spot publication remain limitations. The current top-level manifest selects the final model; archived presentation files are provenance only.

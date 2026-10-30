# Long-history rolling fusion research archive

`rolling_long_history_v4` is an archived stage inherited by later versions. The official product is [v6](FUSION_V6.md). All fitted parameters were selected before October and frozen at the September information cutoff; daily updates change inputs only.

## Design

New market residual models train from 2012-10-01, using about 3,364–3,423 mature samples versus 698–757 common member samples. Twenty-six specifications per horizon include basis, mean corrections, Ridge, Elastic Net, shallow LightGBM, Huber, and dollar/volatility-standardized residuals. Sixteen combination rules include individual models, futures, windows, and regularization. A nonnegative regularized ensemble minimizes joint dollar error without member floors.

The accepted update is `v3 + beta * (new ensemble - v3)`, with beta estimated on the previous 252 mature errors and bounded to [0,1]. Risk evaluates 810 symmetric/asymmetric shape/scale specifications by past CRPS, including OVX and market-volatility scales. All 63 quantiles are monotone and retain the selected point median. Latest label publication used in the final fit is 2026-09-24.

| Horizon | Final new-ensemble weight β | Retained v3 weight |
| --- | ---: | ---: |
| 1 day | 0.0224% | 99.9776% |
| 5 day | 26.8771% | 73.1229% |
| 20 day | 21.0655% | 78.9345% |
| 60 day | 79.3336% | 20.6664% |

Actual VECM, STL, LSTM, ML point, and risk models remain connected through inherited residual features and new candidates. Nonzero input sensitivity establishes use, not independent positive contribution or a simple weight interpretation.

## Historical evidence and amendments

Quarterly cross-fits and combination-rule selection use mature labels; rules lock before 2025-09-01 and the table evaluates subsequent origins.

| Horizon | n | v3 RMSE, USD | v4 RMSE, USD | Improvement over v3 | v4 / Spot RW | v4 / Futures hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 261 | 5.0700 | 5.0932 | −0.46% | 0.6373 | 0.9835 |
| 5 | 257 | 7.9234 | 7.8453 | +0.98% | 0.8057 | 0.9979 |
| 20 | 242 | 14.0127 | 13.9933 | +0.14% | 0.9452 | 1.0291 |
| 60 | 202 | 25.3211 | 24.1627 | +4.57% | 0.9615 | 0.9941 |

On 202 common four-horizon origins, early-RW-MSE-normalized aggregate RMSE improves 2.465% versus v3 and 9.983% versus spot RW, but is 0.464% worse than futures hold. Paired bootstrap uses 20/60-origin blocks and 4,000 draws. The 60-origin improvement interval versus v3 is [-3.54%, +5.73%], one-sided p≈0.131; versus spot RW it is [+5.11%, +15.75%]. These are conditional development results without adjustment for all research selection.

The original gate required aggregate improvement and at most 2% deterioration at any horizon. The unshrunk proposal failed with 2.19% h1 deterioration. After a disclosed shrinkage amendment, h1 deterioration was 0.46% and the gate passed. Asymmetric-tail calibration was a second historical amendment. `guard_design.json` and `risk_tail_design.json` preserve both; the rejected proposal's 13.3% h60 gain is not the final result.

## Risk and October

| Horizon | n | v4 CRPS / HS | v4 CRPS / v3 | v3 90% coverage | v4 90% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 139 | 0.5760 | 1.0049 | 79.14% | 78.42% |
| 5 | 135 | 0.7462 | 1.0060 | 83.70% | 72.59% |
| 20 | 120 | 0.9269 | 0.9856 | 81.67% | 79.17% |
| 60 | 80 | 1.0561 | 0.7996 | 56.25% | 73.75% |

Risk forward evaluation starts March 2026. h60 CRPS improves 20.04% versus v3 and coverage by 17.5 percentage points, yet remains worse than HS and below 90% coverage. h5 coverage falls to 72.59%; lower CRPS does not imply complete calibration. Final sources are LightGBM/Ridge at h1/h5, OVX at h20, and HS at h60. [OVX](https://www.cboe.com/us/indices/dashboard/ovx/) is a 30-day USO-option volatility proxy; square-root-time scaling into a WTI spot distribution is a model assumption.

| Horizon | n | v4 RMSE, USD | v3 RMSE | v2 RMSE | v4 / Spot RW | v4 / Futures hold | 90% coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 17 | 6.6272 | 6.6272 | 6.4710 | 0.7275 | 0.7970 | 94.12% |
| 5 | 13 | 10.5906 | 10.7855 | 6.2130 | 0.9411 | 0.8503 | 61.54% |
| 20 | 0 | — | — | — | — | — | — |
| 60 | 0 | — | — | — | — | — | — |

October h1 is essentially unchanged versus v3 and h5 improves 1.81%. Both remain worse than v2. The model was retained under historical gates, not switched using October. There are no realized h20/h60 October targets.

## Artifacts

The archived directory is `m3_ml/fusion_v4/outputs/frozen_october_2026/`; rolling files contain 68 forecasts, metrics, historical evidence, component weights, sensitivity, figures, and delivery audits. Selected SHA-256: `4499a098f93ab7a727ce6c6b33f2302d22ac2b5ae2b5f943e0faa310a37e4272`. Eight targeted tests covered future perturbations, label maturity, zero weights, quantile ordering, median preservation, and target-independent inference. Explicit information rules do not prove a complete original-vintage archive.

Research order: `data → long_models → stack → guard → evaluate → risk → run seal → run test`. Use a separate reproduction directory. `run seal` rejects replacement after October evaluation. Focused checks: `python -B -m pytest -q m3_ml/tests/test_fusion_v4.py`.

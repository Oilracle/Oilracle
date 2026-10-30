# Market residual fusion before the rolling-only scope

This archive preserves the full v3 study before daily rolling became the only delivery mode. It is not the current official product. The [rolling record](FUSION_V3.md) explains the scope change, and the [manifest](../../m4_reports/final_strategy_manifest.json) selects current outputs.

## Method and timing

The existing Yahoo/Kronos WTI nearby-futures snapshot supplies raw Close, delayed by two US federal business days. No new download or historical-price modification was used for this upgrade. All 3,174 overlapping EIA contract-one observations agree within USD 0.011. This does not independently verify later prices. The September 30 origin uses September 28 futures at USD 92.60, while the common published spot anchor remains September 18 at USD 101.44. The conservative delay is separate from [CME settlement timing](https://www.cmegroup.com/trading/energy/files/NYMEX_Energy_Futures_Daily_Settlement_Procedure.pdf).

Basis uses same-observation-date spot/futures pairs, available only after both legs are published, with trailing medians over 20/60/250 observations. Twenty-five point candidates include spot RW, futures hold, basis, native member forecasts, convex combinations, and Ridge/Elastic Net/shallow LightGBM residual models. Calendar interactions use volatility, stocks, and spreads. The risk layer preserves the point median and emits 63 monotone quantiles. No member minimum weight is imposed.

The inherited common panel has 14,169 mature rows and actual v2 member refits. From March 2024, 475 horizon/candidate tasks generate quarterly cross-fits with 16 single-thread workers. Eight combination rules are simulated from September 2024 and locked by horizon using errors published before 2025-09-01. Risk scoring starts only after point-rule lock; forward risk selection begins March 2026. The 252 risk candidates combine seven sources, three windows, three mixtures, and four widths. Final fits consume only labels available by September 30, then seal before October evaluation.

## Historical full-path objective

The original main objective uses 245 complete origins × 17 horizons, normalized by early horizon-specific RW MSE. Paired block bootstrap keeps each origin's horizons together, with 4,000 draws and seed 5210.

| Comparator | v3 aggregate RMSE improvement | 95% interval, 60-origin blocks |
| --- | ---: | ---: |
| Original published spot RW | **14.24%** | **6.61%–21.52%** |
| Previous v2 fusion | **14.14%** | **6.50%–21.46%** |
| Lagged futures hold | **−3.05%** | **−7.38%–1.18%** |

The 20-origin spot-RW interval is 6.39%–21.71%; one-sided centered block-bootstrap p-values are 0.0167/0.0140 for 20/60-origin blocks. Repeated development and inherited specifications make these conditional exploratory results, not independent confirmation. The aggregate does not beat the stronger futures baseline.

| Horizon | n | v3 RMSE, USD | Spot RW RMSE | Futures hold RMSE | v3 / Spot RW |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 261 | 5.0700 | 7.9918 | 5.1788 | **0.6344** |
| 5 | 257 | 7.9234 | 9.7375 | 7.8619 | **0.8137** |
| 20 | 242 | 14.0127 | 14.8045 | 13.5973 | **0.9465** |
| 60 | 202 | 25.3211 | 25.1294 | 24.3050 | **1.0076** |

Per-horizon counts differ from the complete 17-horizon intersection. Forward risk results are:

| Horizon | n | CRPS / HS | CRPS / v2 | 90% interval coverage |
| --- | ---: | ---: | ---: | ---: |
| 1 | 139 | **0.5732** | 0.6121 | 79.14% |
| 5 | 135 | **0.7418** | 0.7773 | 83.70% |
| 20 | 120 | **0.9404** | 0.9536 | 81.67% |
| 60 | 80 | **1.3208** | 1.1181 | 56.25% |

## October full evidence

| Task | n | v3 RMSE, USD | v2 RMSE | Spot RW RMSE | Futures hold RMSE | v3 / Spot RW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rolling 1 day | 17 | 6.6272 | 6.4710 | 9.1101 | 8.3154 | **0.7274** |
| Rolling 5 days | 13 | 10.7855 | 6.2130 | 11.2532 | 12.4553 | **0.9584** |
| Complete fixed-origin path | 17 correlated targets | 8.5314 | 6.2349 | 5.3575 | 9.9805 | **1.5924** |

The fixed path stayed around USD 94–96 and missed the later October rise. Its 59.2% deterioration versus spot RW is retained, not hidden by the later scope change. All three tasks beat futures hold in this short window but lose to v2. No October outcome was used to alter weights. Fixed-path targets are correlated, not independent forecasts; h20/h60 have no realized October targets.

All real member inputs remain connected through residual baselines, features, or ensembles. Final h60 is an Elastic Net correction of M1. Rules are soft8 at h1/h20, soft32 at h5, and best at h60. Local +USD 0.01 member perturbations establish use, not positive independent accuracy contributions. h1/h5 risk uses ML scale and empirical fusion shape; h20 uses ML quantile shape; h60 uses widened HS shape.

Seventeen v2/v3 tests cover causal labels/features, target-independent inference, monotonic quantiles, unchanged medians, fixed/rolling first-origin agreement, and preserved v2 artifacts. Selected SHA-256 is `12d803737e1804b2e69312aede92113535b38f9f7fee743c9b903ecc409c209a`. Research order is `specification → point → risk → evaluate → tests → run seal → run test`. Sealing refuses an already evaluated directory. Reproduce research in a separate location; current presentation generation follows the final manifest rather than this archive.

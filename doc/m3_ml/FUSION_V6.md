# Integrated WTI Forecast final model

`short_bias_v6` is the sole official version. It retains every v5 point forecast and the one-, 20-, and 60-day distributions. The accepted change is five-day tail calibration. Actual M1, M2, LSTM, ML point, distribution, and risk modules remain in one system.

All selection and final fitting use information available by 2026-09-30. October updates inputs only. Earlier fitted objects and predictions remain preserved. No reliable one- or five-day point improvement passed the historical gates; the accepted risk improvement comes with wider intervals.

## Historical point performance

The original development window is September 2023–September 2026, with initial training through August 2023. Fusion needs earlier forecast errors for fitting/selection, so its reported outer origins begin 2025-09-02. These are historical forward fits, not retrospective predictions from the final September model. Repeated development and inherited base specifications make this development validation rather than a fully independent blind test.

| Horizon (quoted days) | n | v6 RMSE (USD/barrel) | Published spot RW RMSE | RMSE reduction vs RW | v6 MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 261 | 5.1196 | 7.9918 | 35.94% | 3.6439 |
| 5 | 257 | 7.8418 | 9.7375 | 19.47% | 5.3603 |
| 20 | 242 | 13.9357 | 14.8045 | 5.87% | 9.6062 |
| 60 | 202 | 23.4549 | 25.1294 | 6.66% | 18.1422 |

Labels are available by the September cutoff and target dates end 2026-09-18. Each horizon has a different target window. RW holds the latest published spot quote. Against the stronger futures-hold baseline, RMSE improves only 1.14%/0.26% at h1/h5, is 2.49% worse at h20, and 3.50% better at h60. The spot-RW gain is not a gain over every benchmark.

`rolling/history_review/point_metrics.csv` records dates and counts; `historical_point_forecasts.png` aligns actuals, fusion, and RW by target date. Risk plots use a separate June 2026 onward window with h1/h5 n=76/72 and coverage 93.42%/79.17%. Do not mix these counts with the point table. `python -B -m m4_reports.plot_historical_forecast` reads sealed historical predictions, checks hashes/maturity, and recomputes metrics without fitting or reading October outcomes.

## Rejected short-term corrections

The experiment evaluated 73 specifications at each of h1/h5: recent 20/63/126/252-error means/medians, dollar/volatility-standardized corrections, Ridge conditional bias, shrinkage, and fit-only magnitude caps. Sixteen workers generated monthly cross-fits from March 2025 and outer validation from September 2025. Each selection uses only previously published errors, at most 126 mature inner errors, at least 1% inner MSE improvement, and no more than 2% deterioration over the latest 63 errors. The final acceptance gate requires at least 0.5% RMSE improvement, no more than 2% MAE deterioration, and improvement in at least two quarters.

| Horizon | n | v5 historical RMSE | Proposed RMSE | Relative deterioration | Accepted |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 day | 261 | 5.1196 | 6.7882 | +32.59% | No |
| 5 day | 257 | 7.8418 | 11.8835 | +51.54% | No |

The proposals fail particularly in 2026 Q2 and are rejected. `candidates.parquet`, `point_audit.json`, `point_proposal.csv`, and `point_metrics.csv` retain the evidence. Both official entries in `point_final.json` are `p000` (zero correction); its inner `selection` evidence is not a claimed p000 gain. This does not establish a theoretical optimization limit.

## Risk calibration

Calibration starts from actual v5 distribution forecasts issued from March 2026 and their subsequently published errors. Evaluation begins June 2026, giving n=76/72, not v5's longer 139/135 sample. Monthly fits use up to 126 mature errors and separate conservative standardized lower/upper tail quantiles with a fixed 95% tail-calibration target. The point median and quantile monotonicity are preserved; original intervals are a lower bound and dollar-width expansion is capped.

The preregistered gate requires at least 60 targets, at least a two-percentage-point reduction in distance from nominal 90% coverage, at most 5% CRPS deterioration, and at most 40% mean-width growth. The initial 50% expansion proposal fails the width gate. One disclosed historical development amendment lowers the cap to 35%, keeping the acceptance gate unchanged; the initial proposal and `width_amendment.json` remain saved.

| Horizon | n | v5 coverage | Proposed coverage | Proposed CRPS / v5 | Proposed width / v5 | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 day | 76 | 93.42% | 96.05% | 1.0559 | 1.35 | No |
| 5 day | 72 | 65.28% | 79.17% | 0.9913 | 1.35 | **Yes** |

One-day expansion worsens already-high coverage and CRPS, so the original interval is retained. Five-day coverage improves by 13.89 percentage points, CRPS by 0.87%, and 90% interval score by 2.49%, at 35% greater width. Coverage remains below 90%. Interval score penalizes width plus 20 times either tail's missed distance. `rolling/historical_risk_metrics.csv` describes accepted outputs; `risk_metrics.csv` includes the rejected h1 proposal.

## Frozen October evaluation

| Horizon | n | v6 RMSE | v5 90% coverage | v6 90% coverage | v5 CRPS | v6 CRPS | v6 mean interval width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 day | 17 | 6.6272 | 94.12% | 94.12% | 0.032051 | 0.032051 | 21.27 USD |
| 5 day | 13 | 10.0658 | 76.92% | 100.00% | 0.050286 | 0.049941 | 30.95 USD |
| 20 day | 0 | — | — | — | — | — | — |
| 60 day | 0 | — | — | — | — | — | — |

Five-day coverage rises from 10/13 to 13/13, CRPS improves about 0.69%, and interval score 9.53%. Mean width rises from USD 22.92 to 30.95. Thirteen overlapping targets cannot support future coverage guarantees or a claim of better point accuracy. No h20/h60 October targets are realized. October was already viewed in earlier research; it is retrospective evaluation and did not select this amendment, its width cap, or a fallback version.

## Artifacts and reproduction

The official [manifest](../../m4_reports/final_strategy_manifest.json) points to `m3_ml/fusion_v6/outputs/frozen_october_2026/rolling/`. Its `presentation_data.json`, 68-row `forecasts.csv`, `metrics.csv`, historical point/risk tables, `figures/`, and `delivery_audit.json` are the final evidence. The path comparison shows separate daily rolling h1/h5 forecasts by target date, not a single September-origin future path. October RMSE reductions versus spot RW are 27.25%/10.55%.

Selected-model SHA-256: `5ddaa794e5b640689bfbd9b7e9842ab43e35d7c5cea798f5afac9b95d88f8fde`.

Seven targeted tests cover unpublished labels, target-independent inference, quantile median/ordering, width limits, and original hashes. Delivery audits independently recompute metrics, verify unchanged point predictions and h1/20/60 distributions, and check presentation inputs. They verify the declared information protocol, not the completeness of real-time vintages.

```powershell
uv run python -B -m m4_reports.final_strategy
uv run python -B -m m4_reports.plot_final_forecast
uv run python -B -m m4_reports.plot_historical_forecast
uv run python -B -m m4_reports.audit_strategy
uv run python -B -m pytest -q m3_ml/tests/test_fusion_v6.py
```

Research entry points are `m3_ml.fusion_v6.research` and `m3_ml.fusion_v6.tighten`. Use a separate reproduction workspace; do not overwrite sealed research. [Final slides and video](../m4_reports/README.md) use eight minutes plus two minutes Q&A; the pitch video is separate.

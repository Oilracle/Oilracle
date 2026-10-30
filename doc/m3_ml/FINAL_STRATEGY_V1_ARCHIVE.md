# Original global fusion research archive

This is the superseded `integrated_v1` study, not the final product. See [v6](FUSION_V6.md). The [diagnostic report](FUSION_DIAGNOSTICS.md) identifies a cross-horizon objective mismatch and an unsupported location-calibration step. Original parameters and unfavorable October outputs remain preserved.

## Components and objective

Actual member models were M1 VECM K3/rank2 (27.4813%), monthly STL bridge (25.9981%), three-seed 60-step LSTM with hidden 32/dropout 0.2 (41.5206%), and selected Elastic Net (5%). A-LightGBM supplies the conditional distribution. These were real native outputs, not replacement proxies. The four price models were converted to log returns from a common published anchor and blended with one global nonnegative weight vector. A five-percent floor was a design constraint, not a statistically established optimum. ML point predictions reached that floor and were strongly shrunk.

The common panel has 757 origins and 14,169 rows over 19 horizons. h1/5/20/60 contribute 757/753/738/698 rows, totaling 2,946 weight-fit rows. Quarterly forward weights start September 2024; penalties 0/0.1/1/10 select 0.1 using equal-horizon log-RMSE/RW, score 0.963109. Risk validation from September 2025 compares five ML distributions × three empirical-shape mixtures, selecting A-LightGBM/mixture zero, CRPS/HS 0.961819. Risk validation has 962 key-horizon rows, or 4,745 including intermediate horizons. Final calibration uses 9,438 forward errors, not fitted-in-sample fusion residuals.

## Location calibration and limitations

For log center c, ML quantiles, conditional scale s, and median standardized fusion error b, the selected rule is `Q_final(tau)=c+s*b+Q_ML(tau)-q50_ML`. Even mixture zero moves the center. The final point is `published_anchor * exp(Q_final(0.5))`, the constructed price median. All 63 quantiles are sorted jointly. Probabilities interpolate the grid bounded by 0.00125/0.99875; tail ES is a grid approximation. None of these constructions guarantees coverage.

Base specifications already used the complete development window, so forward fusion fitting is conditional rather than fully nested selection. October was previously inspected and is retrospective.

## October results

| Mode and target | n | Final price RMSE (USD) | RW RMSE (USD) | RMSE / RW | 90% interval coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rolling 1 day | 17 | 9.4469 | 9.1101 | 1.0370 | 76.47% |
| Rolling 5 days | 13 | 11.9092 | 11.2532 | 1.0583 | 84.62% |
| Fixed origin, complete October path | 17 targets | 7.2455 | 5.3575 | 1.3524 | 100.00% |
| 20 / 60-day horizons | 0 | Not realized | Not realized | Not scored | Not scored |

The September information set's latest WTI quote is September 18. Rolling h1/h5 are about 3.7%/5.8% worse than RW, and the full fixed path 35.2% worse. Its 17 targets are dependent. Broad fixed-path intervals do not establish overall calibration; rolling coverage is below nominal 90%.

## Historical removal ablations

Each ablation refits remaining weights and calibration while retaining the selected regularization/risk specification. Point and risk scores use their respective windows; lower is better.

| Removed module | Point center log RMSE/RW | Risk CRPS/HS |
| --- | ---: | ---: |
| None, complete policy | 0.963109 | 0.961819 |
| M1 | 0.967636 | 0.971400 |
| M2 | 0.958289 | 0.968661 |
| LSTM | 0.985465 | 0.956075 |
| ML point forecast | 0.961942 | 0.960799 |

M1/LSTM help this point score, M2 helps risk but not the point score, and ML point increment is weak. Integration has costs; no component improves every metric.

The archived `m3_ml/fusion/outputs/frozen_october_2026/` contains selected parameters, 87 forecast rows, metrics, cross-fit audits, selections, ablations, and original figures. Six targeted tests covered weights, label maturity, causal errors, coherent quantiles/risk, and target-independent October inference. These engineering checks did not prove forecasting superiority. A metadata-count correction left all selected values and forecasts unchanged. Historical replay uses `fusion.prepare spec`, native R replay, `fusion.prepare ml`, `fusion.prepare lstm`, `fusion.run select`, and `fusion.run test`; use a separate reproduction directory.

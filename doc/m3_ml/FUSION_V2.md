# Horizon-specific fusion research archive

`horizon_rw_v2` is an archived policy, superseded by [the current final model](FUSION_V6.md). All fitting/selection labels were published by 2026-09-30, parameters were sealed before October evaluation, and October did not change this version's weights or calibration.

## Point and risk construction

The price center is a nonnegative arithmetic combination of native M1, M2, LSTM, ML, and explicit RW prices, with weights summing to one and lower bounds zero. Dollar-price MSE replaces v1's global log-return objective. Ridge-style regularization shrinks toward RW, which is itself a candidate.

M1 is WTI–Brent–gasoline VECM K=3/rank=2; M2 is monthly STL; LSTM uses 60 inputs/outputs, hidden 32, 15 epochs, and seeds 42/43/44. The original selected Elastic Net participates in fitting/ablation but its final direct weight is zero at every horizon. ML's retained contribution is chiefly risk/distribution. There are 19 horizons, 1–17/20/60, with a single frozen lookup policy rather than October-dependent switching.

| Horizon | RW | M1 | M2 | LSTM | ML point forecast |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 day | 0 | 95.68 | 0 | 4.32 | 0 |
| 5 day | 0 | 91.67 | 0 | 8.33 | 0 |
| 20 day | 0 | 71.83 | 8.91 | 19.26 | 0 |
| 60 day | 100 | 0 | 0 | 0 | 0 |

## Historical selection

From September 2023, native M1/M2 refit every 63 origins (13 fits each), LSTM 13×3 times, and ML point/distribution use matching historical schedules. Specifications inherit earlier selection. The common panel has 14,169 rows and latest label publication 2026-09-24.

Quarterly weight fits start March 2024 with pure RW and penalties 0/0.01/0.1/1/10. From September 2024, each block selects its penalty from earlier mature candidate predictions, giving 9,438 rows over 508 origins. Risk candidates start September 2025: HS/five ML sources × shape mixtures 0/0.5/1 × widths 0.75/1/1.25 (54 choices). Forward risk selection starts March 2026, with 2,427 rows over 139 origins. Full-history development scores and past-only selection checks remain separate.

Calibration fixes the price median. With center c, conditional scale s, centered source shape and centered empirical standardized-error shape, `Q(tau) = c + width * s * ((1-mixture)*shape(tau) + mixture*empirical_shape(tau))`. The unmodified source shape/width remains a candidate. The same monotone 63-point curve supplies intervals, probabilities, VaR, and ES.

## Historical evidence

| Horizon | n | Fusion price RMSE/RW |
| --- | ---: | ---: |
| 1 day | 508 | 0.99428 |
| 5 day | 504 | 0.98898 |
| 20 day | 489 | 1.00000 |
| 60 day | 449 | 1.00010 |

The equal-horizon h1–17 mean ratio is about 1.00085; fusion does not dominate. M1 alone is better on these historical records, and some removal ablations improve metrics. Refit LSTM h60 RMSE/RW is about 1.07763, so the earlier apparent long-horizon advantage from stale coefficients disappears; h60 RW selection is historical, not October-driven.

| Horizon | Full-history risk development score | Forward risk selection using past scores only |
| --- | ---: | ---: |
| 1 day | 0.91437 | 0.93639 |
| 5 day | 0.91359 | 0.95425 |
| 20 day | 0.95934 | 0.98618 |
| 60 day | 0.97524 | 1.18131 |

The development and forward-risk windows differ (September 2025 versus March 2026), so their difference is not purely selection bias. Forward 90% coverage is about 69.1%/65.9%/87.5%/38.8%; lower CRPS is not guaranteed calibration.

## October evidence

| Mode | n | New price RMSE (USD) | RW RMSE (USD) | RMSE/RW |
| --- | ---: | ---: | ---: | ---: |
| Rolling 1 day | 17 | 6.4710 | 9.1101 | **0.7103** |
| Rolling 5 days | 13 | 6.2130 | 11.2532 | **0.5521** |
| Complete fixed-origin path | 17 targets | 6.2349 | 5.3575 | **1.1638** |
| Rolling 20/60 days | 0/0 | Not realized | Not realized | Not scored |

Rolling h1/h5 improve about 29.0%/44.8% versus RW. The fixed path remains 16.4% worse, compared with v1's ratio 1.3524. All three realized groups have 100% interval coverage, but samples are small and intervals wide. The fixed path's 17 targets share one origin and are dependent. All 87 forecast rows remain, including unrealized targets.

## Artifacts

`m3_ml/fusion_v2/outputs/frozen_october_2026/` contains selected weights, risk configuration, test forecasts/metrics/audit, point cross-fits/selections, risk development and forward-validation records, refit audits, and historical ablations. Eight targeted tests cover zero weights, arithmetic blending, pure RW, maturity, preserved medians, causal selection, and target-independent inference. Base hyperparameters used the full development period, so this is conditional development validation, not a fully nested blind experiment.

Historical replay order: `fusion_v2.prepare spec`, `Rscript m3_ml/fusion_v2/replay_members.R`, `fusion_v2.prepare lstm`, `fusion_v2.run select`, then `fusion_v2.run test`. Run in a separate copy and use `m3_ml/tests/test_fusion_v2.py` for checks. Current presentation generation uses the final manifest, not this archived policy.

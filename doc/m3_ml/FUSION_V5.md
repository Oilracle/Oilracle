# State reliability fusion research archive

This page records `state_reliability_v5`. The official [v6](FUSION_V6.md) inherits every v5 point forecast and h1/20/60 distributions, replacing only h5 risk tails. Earlier “final” status applies to the historical stage, not the current delivery.

## Design and accepted rules

Thirteen experts include actual M1 VECM, M2 STL, three-seed LSTM, ML point forecasts, v4, futures, basis, and long-history residual models. Market state includes volatility, basis, spot age, trend, expert disagreement, and calendar features. Ridge or shallow LightGBM predicts standardized expert errors and converts them to reliability weights. Static ensembles and v4 remain candidates. There is no minimum component weight.

The initial point grid contains 64 specifications × four horizons; support guards produce 253 candidate rules reusing cross-fits. Risk has 216 specifications: nine sources, four windows, three tail targets, and symmetric/asymmetric calibration. Risk selection seeks at least 90% empirical coverage, at most 7.5% one-sided misses, and width at most twice HS, minimizing CRPS among feasible candidates; infeasibility is explicitly recorded.

Final rules: h1 retains v4; h5 uses a regularized static ensemble on 126 mature errors; h20 uses LightGBM conditional reliability; h60 uses a regularized static ensemble on 252 errors. Updates are `v4 + 0.5 * (proposal - v4)`. This is a cap on the overall change, not a minimum member share. h20 weights change with daily inputs while the fitted map stays frozen. Input perturbations show all four member signals affect some outputs, not that each improves accuracy independently.

## Historical development

Quarterly cross-fits start March 2025 and outer selection September 2025. Only published errors enter selection, using at most 252 mature errors and requiring 1% inner MSE improvement. Latest final-fit label publication is 2026-09-24. Original acceptance requires at least 0.5% aggregate RMSE improvement and no horizon worse by over 2%.

Three disclosed historical amendments preserve those gates: shrink out-of-support states toward v4 using fit-only 1%–99% ranges; take half the proposed change; and protect intervals using the outer envelope of the new distribution and the v4 shape shifted to the new median, capped at 1.5 times the shifted original width. Monotonicity and the point median are preserved. `support_design.json`, `damping_design.json`, `risk_safety_design.json`, and `initial_proposal/` retain these decisions.

| Historical proposal | aggregate RMSE Improvement over v4 | h1 deterioration | Point upgrade gate |
| --- | ---: | ---: | --- |
| Initial state/reliability policy | 2.15% | 2.34% | Rejected |
| With support-domain guard | 2.36% | 2.34% | Rejected |
| Final half update | 1.36% | 0.52% | Passed |

Inherited member specifications and repeated historical amendments prevent interpretation as a fully nested blind test. October did not select the design or upgrade.

## Point evidence

| Horizon | n | v4 RMSE | v5 RMSE | Improvement over v4 | v5 / Spot RW | v5 / Futures hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 261 | 5.0932 | 5.1196 | −0.52% | 0.6406 | 0.9886 |
| 5 | 257 | 7.8453 | 7.8418 | +0.05% | 0.8053 | 0.9974 |
| 20 | 242 | 13.9933 | 13.9357 | +0.41% | 0.9413 | 1.0249 |
| 60 | 202 | 24.1627 | 23.4549 | +2.93% | 0.9334 | 0.9650 |

The aggregate uses 202 common origins, horizon errors normalized by early RW MSE, and equal horizon weights. It improves 1.356% over v4, 11.204% over spot RW, and 0.898% over futures hold; it is not the arithmetic mean of table percentages. A paired 60-origin block bootstrap gives:

| Comparator | Improvement | 95% interval |
| --- | ---: | --- |
| v4 | +1.36% | [−0.06%, +2.42%] |
| Spot RW | +11.20% | [+7.32%, +15.99%] |
| Futures hold | +0.90% | [−4.00%, +5.02%] |

The one-sided p≈0.0497 versus v4 does not establish robust significance: the two-sided interval contains zero and repeated research selection is uncorrected. The futures-baseline increment is also unproven.

## Risk and October evidence

Forward risk selection begins March 2026; lower CRPS is better.

| Horizon | n | v5 CRPS / v4 | v5 CRPS / HS | v4 90% coverage | v5 90% coverage | Mean interval width increase |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 139 | 1.0027 | 0.5776 | 78.42% | 79.14% | +2.66% |
| 5 | 135 | 0.9990 | 0.7455 | 72.59% | 74.81% | +2.58% |
| 20 | 120 | 0.9784 | 0.9068 | 79.17% | 82.50% | +5.04% |
| 60 | 80 | 0.9514 | 1.0048 | 73.75% | 87.50% | +29.17% |

Mean coverage deviation improves by 5.01 percentage points and mean CRPS ratio is 0.9829. The h60 improvement has a 29.17% width cost; its CRPS remains 0.48% worse than HS. All historical coverages remain below 90%. Final raw sources are ML Ridge at h1/h5, OVX scale at h20, and linear ML quantiles at h60, followed by interval protection. h20/h60 have `constraints_satisfied=false` in `risk_final.json`; constraints are not a guarantee.

Calibration is informed by [Conformalized Quantile Regression](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html) and time-varying coverage concerns in [Adaptive Conformal Inference](https://proceedings.neurips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html). This implementation does not establish their assumptions or claim their theoretical guarantees; October calibration stays frozen.

| Horizon | n | v5 RMSE | v4 RMSE | v2 RMSE | v5 / Spot RW | v5 / Futures hold | v5 90% coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 17 | 6.6272 | 6.6272 | 6.4710 | 0.7275 | 0.7970 | 94.12% |
| 5 | 13 | 10.0658 | 10.5906 | 6.2130 | 0.8945 | 0.8082 | 76.92% |
| 20 | 0 | — | — | — | — | — | — |
| 60 | 0 | — | — | — | — | — | — |

October h5 RMSE improves 4.96% over v4 and coverage rises from 61.54% to 76.92%; CRPS falls from 0.05622 to 0.05029. h1 is unchanged. Neither point result beats v2's October score, but October is not used to switch versions. These are short retrospective observations.

## Reproduction record

Artifacts are under `m3_ml/fusion_v5/outputs/frozen_october_2026/`, including rolling forecasts/metrics, historical evidence, expert weights, member sensitivity, and delivery audits. Selected SHA-256: `ff2f527014bea685cede195ba71f765a2204ad8c0ac84fca0d245114f12bb031`. Ten targeted tests covered causal selection, state weights, support guards, tails, width limits, and target-independent inference. Futures retain the two-federal-business-day delay; other publication assumptions remain unchanged.

Research order: `data → point → evaluate → risk → support → evaluate → damp → evaluate → risk → risk_safety → run fit → tests → run seal → run test`. `run fit` and `run seal` reject replacing an already evaluated model. Run `python -B -m pytest -q m3_ml/tests/test_fusion_v5.py` for focused checks. Current delivery is always read from the top-level manifest.

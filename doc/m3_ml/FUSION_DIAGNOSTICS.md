# Fusion diagnostics

The first sections concern the superseded global v1 policy; the final section compares current v6 with native VECM. Diagnostics preserve official model/data/prediction artifacts. October is used for error decomposition only, never parameter selection.

## Original fixed-path loss

No date, unit, model-loading, or October-input leak explained the original 35.2% fixed-path gap. Instead, the historical selection objective and risk-location calibration were insufficiently aligned with the delivered point target.

| Same Sep 30 origin and 17-target path | Price RMSE (USD/barrel) |
| --- | ---: |
| RW at common published anchor 101.44 | 5.3575 |
| Four-component fusion before risk location calibration | 6.6786 |
| Original delivered median after location calibration | 7.2455 |

The ratio is 7.2455/5.3575−1, or USD 1.8880 additional RMSE. Over the first ten targets, original/RW RMSE is 2.6604/4.8323; over the last seven, 10.8343/6.0290. Individual fixed-path RMSE is M1 6.6765, M2 5.3600, LSTM 11.9632, and ML 5.3598. LSTM falls from 96.8681 to 86.6890 while final actual is 109.72; its 41.52% weight lowers fusion. At step 17, the center is 96.5028 before and 95.4448 after location calibration.

| Historical horizon | LSTM log-return RMSE/RW | Fusion-center log-return RMSE/RW |
| --- | ---: | ---: |
| 1 day | 1.01920 | 0.99726 |
| 5 day | 1.01867 | 0.99480 |
| 20 day | 0.97718 | 0.95773 |
| 60 day | **0.85126** | **0.90265** |

Equal weighting of h1/5/20/60 rewards LSTM's long-horizon result, then applies that same global mixture to a 1–17-step path. Log-return selection loss also differs from the reported dollar loss. Historical native models mostly retained August 2023 coefficients while ML refit every 63 origins, unlike the final all-member September fit. This unequal model age is a validation-design issue; October alone cannot quantify its cost.

## Location-calibration control

Mixture zero still added median standardized error, so no candidate preserved the already-selected point center. Same-sample historical evidence from September 2025 shows:

| Horizon | Price before location calibration RMSE/RW | Price after location calibration RMSE/RW |
| --- | ---: | ---: |
| 1 day | 0.99291 | 1.00487 |
| 5 day | 0.99181 | 1.00694 |
| 20 day | 0.95755 | 0.99134 |
| 60 day | 0.90700 | 0.93231 |

The h1–17 mean ratio worsens from 0.98223 to 1.00828. Holding shape/width fixed and removing only the location shift gives key-horizon CRPS/HS 0.92998 versus 0.96182; h1–17 values are 0.93805 versus 0.96998. This is a pre-October diagnostic control, not a production replacement.

Independent checks matched CSV/Parquet/target/publication values, reconstructed observable anchors, reconciled all member origin/target/horizon/row bases, and replayed native R/LSTM/ML models. A-LightGBM quantiles at h1/5/17/20/60 reproduce within 2.23e-16. Future/unpublished-input perturbations leave checked fixed-origin predictions unchanged, and protected hashes remain unchanged.

The origin is September 30 at 00:00 New York, not after the close. Under the next-business-day WPSR rule it uses September 18 at 101.44. October 1's switch to September 25 at 85.23 matches local FRED/WPSR sources. The October 7 report URL/page-date conflict was quarantined, so affected quotes use the declared release assumption. This later evidence gap cannot cause the September fixed-path loss. See [September archive](https://www.eia.gov/petroleum/supply/weekly/archive/2026/2026_09_30/wpsr_2026_09_30.php), [schedule](https://www.eia.gov/petroleum/supply/weekly/schedule.php), and [conflicting page](https://www.eia.gov/petroleum/supply/weekly/archive/2026/2026_10_07/wpsr_2026_10_07.php).

The resulting historical design requirements were horizon-aligned dollar loss, a center-preserving risk baseline, consistent native refit schedules, and removal of arbitrary member floors. Current later versions implement these changes. Diagnostic artifacts remain under `m3_ml/fusion/outputs/frozen_october_2026/diagnostics/`; reproduction uses `python -m scripts.audit_fusion_diagnostics` and `Rscript scripts/audit_fusion_r_objects.R`.

## Current v6 versus native VECM

No alignment, price-conversion, member-loading, or median-shift bug was found. October h5 exposes downward-biased futures-related forecasts and a conservative static blend that limits VECM's higher signal. This is a real weakness amplified by a short window, not evidence that VECM should replace fusion based on October.

| Evaluation window | Horizon | n | Final fusion | VECM | Latest published spot RW |
| --- | ---: | ---: | ---: | ---: | ---: |
| Common historical outer window from Sep 2025 | 1 | 261 | 5.1196 | 7.7282 | 7.9918 |
| Common historical outer window from Sep 2025 | 5 | 257 | 7.8418 | 9.3612 | 9.7375 |
| October retrospective | 1 | 17 | 6.6272 | 6.3309 | 9.1101 |
| October retrospective | 5 | 13 | 10.0658 | 5.5742 | 11.2532 |

October h1 MAE favors fusion (5.5493 versus 5.8183) despite VECM's slightly better RMSE. At h5, VECM wins RMSE/MAE and has smaller absolute error at 11 of 13 origins.

All 30 scored forecasts match native VECM inputs, actuals, and anchors; original row differences equal horizons after explicit R/Python index conversion. State/futures availability precedes each origin. Frozen weighted predictions replay exactly and risk retains the same median. Current historical M1 comes from v2's 63-origin refits, not the stale original v1 panel. A quarterly K3/rank2 refit diagnostic gives h1/h5 RMSE 7.7334/9.3760, close to 7.7282/9.3612, preserving the historical ranking.

The final h5 outer equation is `0.860746*v4 + 0.077874*VECM + 0.061379*(futures + historical mean residual)`. v4 also contains VECM: explicit linear paths sum to about 13.67%, plus inherited ML feature effects. Weights use 126 mature errors, regularization, and a half update. October mean signed errors are futures −10.8631, fusion −8.4180, and VECM −2.5879. VECM is above futures at all 13 origins. The recent historical 126-error window instead favors the original ensemble (9.3354 RMSE) over VECM (10.8324), as do the full common window and 2025 Q4–2026 Q3 quarters.

Frozen decomposition gives October h5 RMSE 10.5906 for v4, 9.5501 for the undamped fitted proposal, and 10.0658 final. Damping has a cost in this window, but removing it still does not reach VECM's 5.5742; no parameter is changed.

Among 245 overlapping historical 13-origin windows, VECM wins 67 (27.35%). Only three (1.22%), all around May 2026, reach the October fusion/VECM ratio 1.8058. October 16/19/20 targets account for 74.81% of excess squared loss. These overlapping-window frequencies are not p-values or future probabilities. Thirteen correlated targets cannot establish long-run dominance, and historical results remain conditional on repeated development.

Current artifacts are under `m4_reports/diagnostics/vecm_fusion/`: diagnosis plot/JSON, metrics, and daily error decomposition. Reproduction: `Rscript --vanilla m4_reports/diagnose_vecm_refits.R`, then `python -B m4_reports/diagnose_vecm_fusion.py`. Protected production hashes were unchanged.

# Kronos results: current artifacts and historical negative result

Current frozen-protocol artifacts are under `outputs/frozen_october_2026/`.
Use the protocol and commands at the start of [PLAN.md](PLAN.md), with portable
`data/kronos/corpus/` and `m3_ml/kronos/checkpoints/`. Current stage cutoffs,
normalization, weights and separate evaluation modes differ from the archive.
The following scores and stop decision are historical; they are not October
1–23 results and do not describe current checkpoint completion status.

## Archived training and stop decision

The old validation CRPS exceeded HS by more than 5% at all horizons, triggering
the registered stop rule. The old test was not run and no further tuning was
added. Two implementation bugs were corrected before validation results became
available, as documented in PLAN.md. Root `outputs/validation_scores.csv`,
`selected.json` and `diagnostic_2021.csv` retain the old evidence.

| | Stage 1 (training through 2018; early stopping in 2019) | Stage 2 (training through 2021; early stopping in 2022) |
|---|---|---|
| Training bars | About 2.43 million (578 assets) | About 2.87 million (582 assets) |
| Tokenizer validation reconstruction MSE (30,000 steps) | 0.00528 | 0.00588 |
| Best predictor validation cross entropy | 2.969 (step 2,000) | 3.083 (step 2,000) |
| At early stopping (step 12,000) | 3.028 | 3.163 |

The predictor overfit quickly: its first evaluation at step 2,000 was best;
coarse-token loss then rose. By that point each bar had effectively been seen
14–17 times. The local corpus had fewer than three million bars, versus the
roughly twelve-billion-bar scale reported for official pretraining. Correcting
validation loss reduced its rise from 0.19 to 0.06 but did not change the best
step. Archived stage 2 was trained but unused after the stop decision.

## Archived validation: 2020–2022 geometric mean CRPS / HS

| Temperature T | top-p | h = 1 | h = 5 | h = 20 | h = 60 |
|---|---|---|---|---|---|
| 0.8 | 0.9 | 1.091 | 1.112 | 1.373 | 1.583 |
| 0.8 | 1.0 | 1.069 | 1.087 | 1.310 | 1.534 |
| 1.0 | 0.9 | 1.070 | 1.088 | 1.308 | 1.560 |
| 1.0 | 1.0 | **1.061** | **1.075** | **1.251** | **1.482** |

Temperature 1.0 without top-p truncation was best at every horizon. Its annual
ratios were:

| Year | h = 1 | h = 5 | h = 20 | h = 60 |
|---|---|---|---|---|
| 2020 | 1.128 | 1.084 | 1.345 | 2.106 |
| 2021 | 1.063 | 1.085 | 1.112 | 1.140 |
| 2022 | 0.996 | 1.055 | 1.309 | 1.356 |

Only h1 in 2022 approximately matched HS. Truncated sampling was consistently
worse. In the separate seven-fold distribution study, simple EWMA/GARCH/route A
improved short-horizon CRPS/HS by 2–4%.

## Validation-only diagnosis (2021)

| h | CRPS (Kronos / Historical simulation) | 50% interval coverage | 90% interval coverage | 90% interval width (log return) | Median | Actual mean |
|---|---|---|---|---|---|---|
| 1 | 0.0112 / 0.0105 | 53.4% / 45.4% | 89.2% / 92.0% | 0.079 / 0.073 | 0.001 / 0.001 | 0.002 |
| 5 | 0.0243 / 0.0223 | 55.0% / 45.8% | 89.6% / 91.2% | 0.161 / 0.156 | 0.004 / 0.002 | 0.009 |
| 20 | 0.0566 / 0.0511 | 43.4% / 34.3% | 80.1% / 88.1% | 0.313 / 0.320 | 0.009 / 0.009 | 0.037 |
| 60 | 0.0935 / 0.0814 | 74.1% / 49.8% | 90.4% / 90.8% | 0.634 / 0.699 | 0.037 / 0.020 | 0.116 |

Widths, 90% coverage and medians were of plausible scale, giving no indication
of a unit or inverse-standardization error. Shape was problematic: at h1/h5 the
center was too broad while tails were thin; h20's 90% coverage was only 80.1%;
at h60 the center was much too broad. State-dependent adjustments added noise
on average. Possible explanations, not separately tested, are limited corpus
size, discrete-token/autoregressive error accumulation, and the mismatch between
rolled futures inputs and spot-return targets. This is a useful negative research
result. Official pretrained weights could not provide a fair comparison because
their training history overlapped the original evaluation window.

# Direct ML forecasts: frozen protocol and archived results

## Current protocol

Initial training ends August 31, 2023. Historical validation spans September 1,
2023–September 30, 2026; final fitting uses information available by September 30.
Evaluation targets are October 1–23. Elastic Net, Random Forest and LightGBM
retain their historical grids. Every raw quote in a label window must have been
published by its training/selection cutoff.

Each method/horizon is fitted once for final inference. Standardization,
clipping, relative Elastic Net penalties and all model coefficients remain
fixed. The historical mean is also frozen; random walk is zero return relative
to the latest published spot anchor. The module retains fixed-origin direct
h1–17 and h20/h60 paths, plus rolling h1/5/20/60 forecasts. The final product
uses rolling mode only. Extra path horizons have their own historical selection;
they are neither interpolated nor based on October features.

Raw contemporaneous `target_wti` is a label, not an observable predictor.
`observable_price(table)` chooses the released quote with the latest observation
date, with a pre-sample cold start only before the first available quote. A late
older release cannot reverse the anchor. Targets are log target price minus log
published anchor. The WPSR evidence/weekly-rule publication schedule is explicit
and retrospective; it is not a complete real-time vintage archive.

`--denoise` selects variables and estimates Kalman variances in the initial
training period, then only filters states. `--news` uses the approved
`news-release-calendar-v1`: previous-month-end AI-GPR statistics available on the
next month's first day; prior-Sunday keyword GPR on Monday (federal holidays
defer release); EPU on the following day with at most seven days' carry-forward.
Missing monthly/weekly releases cannot perpetually reuse old states.

```bash
uv run python -m m3_ml.tabular.run_direct
uv run python -m m3_ml.tabular.run_direct --denoise
uv run python -m m3_ml.tabular.run_direct --news
uv run python -m m3_ml.tabular.run_direct --denoise --news
uv run python -m m3_ml.tabular.analyze_direct
```

Outputs use `outputs/frozen_october_2026/` and protocol ID
`qf5210-frozen-october-2026`. Prefixes distinguish direct, denoised and news
variants. Metrics separate mode/horizon/model and retain n=0 for unrealized
horizons. Rolling counts are 17/13/0/0; unknown target dates stay missing while
numeric forecasts and row positions remain available. The two news variants
were rerun after the October 29 calendar repair: each retained 437 candidate
evaluations, 57 final selections, 435 forecasts and 30,424 availability-audit rows.
Historical source revisions required rerunning historical selection as well as
final fitting. See the packaged news audit/output files. October is retrospective,
not a newly untouched blind test.

## Archived expanding-window experiment

The following scores belong to the January 3, 2023–October 6, 2026 test with
monthly refits and 2020–2022 validation. Root outputs retain them; current commands
run the new protocol. No ML model reliably beat random walk across horizons,
years and regimes. Elastic Net selected all-zero feature coefficients at
h1/h5/h20, reducing to the historical mean. At h60 a sparse model still lost
3.9% in test, without significance. Trees deteriorated with horizon; h≥5 DM
comparisons were generally significant. Direction accuracy largely reflected
the class base rate rather than useful timing.

### Overall archived results

Price MAE is USD/barrel. RMSE ratios and DM comparisons use the archived return
loss definition; DM is two-sided with Newey–West lag h−1. See
`outputs/direct_overall.csv`.

| h | Model | MAE (USD) | MAPE | RMSE / RW | Direction | DM p |
|---|---|---|---|---|---|---|
| 1 | Random walk | 1.42 | 1.82% | 1.000 | — | — |
| 1 | Elastic Net = Historical mean | 1.42 | 1.82% | 1.000 | 52.3% | 0.64 |
| 1 | Random Forest | 1.42 | 1.82% | 1.003 | 49.5% | 0.20 |
| 1 | LightGBM | 1.43 | 1.83% | 1.006 | 47.9% | 0.22 |
| 5 | Random walk | 3.23 | 4.12% | 1.000 | — | — |
| 5 | Elastic Net = Historical mean | 3.23 | 4.13% | 1.001 | 46.6% | 0.04 |
| 5 | Random Forest | 3.31 | 4.23% | 1.023 | 46.6% | 0.007 |
| 5 | LightGBM | 3.30 | 4.21% | 1.024 | 44.8% | 0.001 |
| 20 | Random walk | 5.71 | 7.26% | 1.000 | — | — |
| 20 | Elastic Net = Historical mean | 5.72 | 7.26% | 1.004 | 48.1% | 0.14 |
| 20 | Random Forest | 6.30 | 7.95% | 1.080 | 42.6% | 0.003 |
| 20 | LightGBM | 6.05 | 7.66% | 1.044 | 42.3% | 0.01 |
| 60 | Random walk | 9.67 | 12.04% | 1.000 | — | — |
| 60 | Historical mean | 9.64 | 11.96% | 1.005 | 57.9% | 0.48 |
| 60 | Elastic Net | 9.92 | 12.22% | 1.039 | 57.0% | 0.32 |
| 60 | Random Forest | 10.91 | 13.40% | 1.121 | 55.6% | 0.04 |
| 60 | LightGBM | 11.51 | 14.23% | 1.163 | 53.3% | 0.05 |

Actual up-window proportions were 52.1%, 51.8%, 48.9% and 42.9% at h1/5/20/60.
The volatility split used a pre-test median daily volatility of 1.81%, producing
577 high- and 359 low-volatility days. Neither regime showed reliable gains.
Elastic Net's low-volatility h60 ratio was 0.987 (p = 0.76); low-volatility h20
tree ratios were 1.10–1.14. The cumulative-loss chart is
`outputs/figures/direct_cumulative_vs_rw.png`.

Tree losses concentrated in 2023's bearish bias (about −8% predicted mean h60
return versus actual near zero) and delayed responses to 2026 reversals. Around
the April peak, h20 predictions averaged +5.8%, followed by an approximately
18% May decline. These are explanations of observed errors, not tuning targets.

### Elastic Net search correction (October 17)

A fixed alpha grid was not comparable across horizons: alpha_max at h60 was
roughly 40 times h1 (about 0.103 versus 0.0027 for l1_ratio 0.5). A preprocessing
audit found the design issue. The replacement computes alpha_max inside each
training window and uses ratios 1/0.3/0.1/0.03/0.01 with l1_ratio 0.1/0.5/0.9.
Selection still uses validation only. h1/h5/h20 stayed all-zero; h60 selected
ratio 0.3. Trees were unaffected. Relative validation RMSE changes at l1_ratio
0.5 are:

| h | × 0.3 | × 0.1 | × 0.03 | × 0.01 |
|---|---|---|---|---|
| 1 | +0.5% | +1.8% | +4.1% | +6.6% |
| 5 | +3.2% | +7.2% | +12.4% | +18.5% |
| 20 | +5.3% | +11.0% | +26.6% | +38.1% |
| 60 | −5.2% | +2.1% | +19.7% | +35.0% |

### Post-test, non-blind Kalman denoising experiment

Pre-2019 negative lag-one autocorrelation of weekly flow changes identified
eight noisy inputs using threshold −2/sqrt(n), approximately −0.096. No stock
variable was selected. Local-level variances were fitted using pre-2019 data and
then frozen. Filtering, not smoothing, replaced changes in eight inputs; the
remaining 48 features, grids and evaluation stayed unchanged. Estimated new-data
weights ranged from 0.09–0.10 for distillate/jet supply to 0.19 for imports and
0.66 for production. Post-2020 lag-one correlations shifted from −0.15…−0.55 to
−0.17…+0.30; truncation checks confirmed causality.

| Model | Validation RMSE change | Test RMSE / RW (original → denoised) |
|---|---|---|
| Elastic Net | 0 (identical) | Unchanged |
| Random Forest | −0.1% to +0.05% | h = 5: 1.023 → 1.017; other changes ≤ 0.002 |
| LightGBM | −0.3% to +0.6% | h = 1: 1.006 → 1.009; h = 20: 1.044 → 1.049; other changes ≤ 0.006 |

Removing measurement noise barely changed forecast accuracy. The eight inputs
accounted for only 1–7% of tree importance, versus their 14% feature share, and
were absent from the selected Elastic Net. Denoising did not resolve weak
return predictability.

### Post-test, non-blind news experiment (old timing convention)

This experiment followed inspection of test results and knowledge of the 2026
geopolitical shock. It used AI-GPR oil/threat/action/regional indices, keyword GPR
and daily US EPU. Twenty-four features combine log1p 7/30-day means and 7-minus-90
day shocks. The old next-day article rule and short carry-forward convention
described this archived experiment only; the current approved monthly/weekly
release calendar above supersedes them. AI-GPR retrospective scoring and EPU
revisions still limit vintage fidelity. GPR's retrospective event labels are
never features.

| Model | h | Validation RMSE change | Test RMSE / RW (original → news) | DM p |
|---|---|---|---|---|
| Elastic Net | 1, 5, 20 | 0 (all coefficients remain zero) | Unchanged | — |
| Elastic Net | 60 | −8.9% | 1.039 → 1.032 | 0.92 |
| Random Forest | 5 | −0.1% | 1.023 → 1.015 | 0.04 |
| Random Forest | 20 | +0.1% | 1.080 → 1.054 | 0.04 |
| Random Forest | 1, 60 | ≤ 1% | 1.003 → 1.003; 1.121 → 1.114 | 0.91, 0.75 |
| LightGBM | All | −2.7% to +0.9% | worse at each horizon by 0.003–0.008 | 0.28–0.80 |

Only h60 Elastic Net gained clearly in validation, selecting six mostly negative
news coefficients, but its test gain was small and insignificant. Random Forest
improved versus its own baseline at h5/h20 but still lost to random walk; selected
trees were also shallower (depth three instead of six), confounding attribution.
News importance was only 11–21%, below its 30% feature share. These results did
not overturn the archived negative point-forecast conclusion.

### Deferred proposal: reduce the influence of 2020 extremes (not run)

The following training/validation diagnostics motivated a post-test proposal,
not an implemented modification to the archived result:

| | h = 1 | h = 5 | h = 20 | h = 60 |
|---|---|---|---|---|
| Feb–Jul 2020 share of squared training targets (4.5% of days) | 47% | 50% | 49% | 17% |
| Same period share of validation RW squared errors (15% of days) | 73% | 75% | 74% | 33% |

The proposal would winsorize training labels at their own window's 1st/99th
percentiles (never actual evaluation labels), use equal-weight annual 2016–2022
validation RMSE ratios, and choose the simplest model within one standard error.
It would retain 2020 observations and report any later test alongside the
original as non-blind research. The expected effect was reduced tree bias and
overreaction, not a promised significant victory over random walk. It remained
deferred. Historical test results must not become tuning data; later fusion
development uses pre-October validation and reports October separately.

# Distribution forecasts: current protocol and archived research plan

## Current frozen October protocol

The initial training cutoff is 2023-08-31. Historical validation covers
2023-09-01–2026-09-30; final fitting uses only information available by 2026-09-30.
Evaluation includes actual target dates from October 1 to October 23, 2026.
Shared constants are in `m3_ml/common/protocol.py`.

For each method and horizon, `fitted_block` estimates the model once at the final
cutoff and predicts the union of required origins. Weights, scaling, empirical
residual distributions, GPD tails, cross-fitted calibration and ensemble weights
remain fixed in October. Internal cross-fitting uses only the final training set.
The archived module supports both `fixed_origin` and `daily_update`; the final
Oilracle product uses daily rolling forecasts only. Fixed-origin output covers
h1–17 and h20/h60; route C starts at h2. Rolling output uses h1/5/20/60.

All quotes in a training target window must have been published by the fitting
cutoff. `data/processed/wti_publication.csv` gives matched WPSR evidence where
available and an explicit weekly release assumption elsewhere: the fifth US
federal business day after the quote week's Friday. This is a retrospective
availability rule, not a complete vintage archive.

`observable_price(table)` selects the most recent observation date among released
quotes. A late older quote cannot move the anchor backward. The pre-sample
published price is used only before the first available quote. Targets are
`log(target_price) - log(published_anchor)`; contemporaneous raw `target_wti`
is not an observable feature. Published-price changes supply WTI/ML-volatility
features; raw quote returns are retained separately for realized-variance labels.

GARCH/EWMA estimation uses the continuous released raw-quote prefix. New releases
advance filtering; missing internal releases prevent advancement rather than
compressing time. Simulation adds the complete quote-row gap from the last known
quote to the origin, then forecasts h rows and converts to the shared anchor.
GARCH-EVT tails use final training residuals. Independent origin seeds prevent
future origins from changing an earlier simulation.

Historical validation refits every 63 origins within annual 2023–2026 segments,
retaining the full original grids. Ordinary methods minimize the geometric mean
of annual CRPS ratios to expanding historical simulation (HS). EVT methods
minimize the mean annual FZ0 difference to HS; A-EVT first selects its scale model
on the same historical validation. Neither target dates nor label availability
may pass September 30.

Current artifacts live in `outputs/frozen_october_2026/`, with protocol ID
`qf5210-frozen-october-2026`. They include 63 quantiles, point medians, origins,
target rows/dates, actuals and cutoff metadata. Score modes separately. October
rolling sample counts are 17/13/0/0 at h1/5/20/60; unreached targets retain numeric
forecasts and missing actuals/dates. Do not infer significance from this short
window or choose a displayed method using its October loss.

```bash
uv run python -m m3_ml.distribution.run validate
uv run python -m m3_ml.distribution.run test
uv run python -m m3_ml.distribution.analyze
uv run python -m m3_ml.distribution.report_validation
```

Validation saves complete 63-level calibration forecasts and `selected.json`.
Reporting only reads these files. Full `test` deterministically refreshes all
methods while fitting only September information; `--methods` may append a
previously untested method, but existing methods require a full rerun to preserve
ensemble consistency. Historical root outputs remain separate. Previously viewed
October data cannot be relabeled as an untouched blind test.

## Archived plan: written before implementation on October 17, 2026

The original question was whether intervals and distributions were easier to
forecast than return direction. Training-only diagnostics (August 2012–December
2022) found correlations of current volatility with future direction of 0.02–0.23,
but correlations with future realized volatility of 0.36–0.64 and OVX correlations
of 0.38–0.73. Excluding February–July 2020 changed little.

The original target was h-day log return at horizons 1/5/20/60. The main quantile
grid contains 23 levels: 0.01, 0.025, 0.05–0.95 in increments of 0.05, 0.975 and
0.99. Price quantiles are `P_t * exp(q)`. Rearrangement prevents crossings.
Features comprise 56 shared predictors, causal 1/5/20/60-day return RMS plus OVX
and VIX, and an optional 24-feature news block. Revised AI-GPR/EPU data limit
real-time interpretation.

| Method | Specification and original grid |
|---|---|
| HS | Empirical target quantiles; expanding, latest 750 or 1,500 origins |
| EWMA | Lambda 0.94/0.97/0.99; normal or empirical standardized-return shape |
| GARCH | Zero-mean Student-t GARCH(1,1) or GJR(1,1,1); parametric or filtered bootstrap; 2,000 validation paths |
| A-Ridge | Log realized-variance target; alpha 0.1/1/10/100; volatility-only, shared + volatility, or + news |
| A-LightGBM | Squared loss, learning rate 0.02, minimum leaf 50, row/column sampling 0.8; leaves 4/8/16, 200/500 trees; with/without news |
| B-linear quantile | Separate L1 quantile regression per level, highs solver; alpha 0.001/0.01/0.1; three feature blocks |
| B-LightGBM | Quantile loss, learning rate 0.03, minimum leaf 50, row/column sampling 0.8; leaves 4/8, 100/300 trees; with/without news |
| B-quantile forest | 300 trees, minimum leaf 20/50/100, max features 0.33/1.0; with/without news |

Route A predicts `log(sum(r_future**2) + 1e-6*h)`. Five blocked cross-fitting
folds purge h origins on either side. Its predictive shape is the empirical
distribution of out-of-fold standardized targets. Linear models standardize and
clip features at ±5; quantile regression scales targets by the training standard
deviation. The ensemble averages quantiles of the five A/B ML models equally,
then sorts; its weights are not tuned.

The archived validation used seven annual folds, 2016–2022, with 63-origin
expanding refits. Labels had to mature before fitting; scored targets could not
pass December 31, 2022. Selection minimized the geometric mean of annual CRPS/HS
ratios, giving years equal weight. The archived test used January 2023–October 6,
2026 origins with 21-origin refits. These dates no longer describe the current
frozen experiment.

Metrics include mean pinball loss times two on the 23-level main grid, individual
pinball losses, central 50/80/90/95% coverage and width, ten-bin PIT, median
MAE/RMSE and comparisons by year and volatility regime. DM and coverage tests use
Newey–West lag h−1; h1 additionally has Kupiec/Christoffersen checks. The original
distribution-skill rule required CRPS below EWMA with DM p < 0.05 and no rejection
of nominal 90% coverage at the 5% level.

### Tail risk and direction: added before the original test

Both tails use alpha 1%, 2.5% and 5%. Long-position VaR is `-q_alpha` and short
VaR is `q_(1-alpha)`; price changes use `P_t * abs(exp(q)-1)`. ES integrates tail
quantiles using 20 midpoint levels between 0 and 0.05 and their upper-tail
reflections (40 additional levels). These levels are sorted separately and do
not redefine main-grid CRPS. Quantile linear/tree models fit the additional
levels; extreme samples are scarce and must be reported as uncertain.
GARCH test paths increase to 10,000 to reduce tail Monte Carlo error.

GARCH-EVT fits separate maximum-likelihood GPD tails to standardized training
residuals, retaining the empirical center. Its grid is GARCH/GJR × 5%/10% tail
thresholds. A-EVT replaces the two tails of cross-fitted route-A residuals; its
base is whichever of A-Ridge/A-LightGBM has lower validation CRPS. EVT selection
minimizes the annual mean FZ0 difference to HS, averaged over three alphas and
both tails. FZ0 may be negative, so use differences, not ratios.

Risk reporting includes breaches, HAC coverage tests, h1 Kupiec/Christoffersen,
Acerbi–Szekely Z2 (10,000 simulated replications), and joint VaR/ES FZ0. The
independent-period Z2 p value is used only at h1; longer overlapping horizons
receive the statistic only. The archived risk-skill rule required lower mean
FZ0 than EWMA with DM p < 0.05 and no rejection of either 2.5% tail's coverage.
Direction uses `1-F(0)`, Brier score versus HS, accuracy and class-specific
precision/recall/F1 at threshold 0.5.

### Route C: post-test variance mixture, written October 18

This proposal followed inspection of distribution and volatility test results.
It is an explicitly non-blind research addition and does not replace the original
methods or enter their ensemble. The results document clarifies which OVX
choices were influenced by the old test.

Validation-only diagnostics showed excess kurtosis of 2.3/13.4/23.5 after HAR
standardization at h5/20/60, versus −0.8/−0.4/0.05 after realized-variance
standardization. Route C therefore models a zero-mean normal mixture over future
variance. It excludes h1. The 18-setting grid combines three variance cores
(HAR-GK, HAR-GK plus implied volatility, equal-weight HAR-GK/calibrated OVX),
Ridge alpha 1/10/100, and empirical/conditional variance uncertainty.

Five purged folds produce log-variance errors. Their mean corrects bias.
Empirical mixing uses every cross-fitted error with equal probability.
Conditional mixing regresses log squared centered errors, adds 1.2704 (the
negative expectation of log chi-square with one degree of freedom), and clips
the log-variance shock scale to [0.2, 2.0]. Quantiles invert the mixture CDF by
bisection; conditional mixing uses 40-point Gauss–Hermite integration.
The original validation/test rules and all 63 levels are retained, with extra
DM comparisons to A-Ridge and GARCH.

### Corrections before the original test

The 2020 negative quote produced two undefined log-return targets per horizon.
Using an ordinary mean accidentally discarded the entire annual fold. `nanmean`
now omits only undefined targets, retaining all seven folds. Parametric GARCH
simulation is seeded through its Student-t distribution object; the forecast
`random_state` parameter alone only seeded bootstrap simulation. Both corrections
have regression checks in `test_reproducibility.py`. Superseded validation
outputs are retained as history; all methods were revalidated after correction.

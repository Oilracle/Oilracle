# Distribution forecast results

## Current frozen October experiment

Run the commands in [PLAN.md](PLAN.md). Current outputs are exclusively in
`outputs/frozen_october_2026/`, with the shared September 30 cutoff and October
1–23 target window. Validation produces complete 63-level calibration forecasts;
test metrics separate fixed-origin and daily-update modes. Rolling h1/5/20/60
counts are 17/13/0/0. Long-horizon forecasts remain numeric, with missing unrealized
actuals and no invented quotation dates. The final product uses rolling mode.
October is a short retrospective evaluation, not a fresh blind experiment.

The following tables retain the older experiment's scores. They must not be
presented as frozen October results. Current scores are in the dedicated output
directory and the final fusion documentation.

## Archived expanding-window experiment

The original test ran January 3, 2023–October 6, 2026, with monthly refits and
2016–2022 annual validation folds. Root `outputs/test_metrics.csv`,
`test_risk.csv`, `test_by_regime.csv` and `test_by_year.csv` retain its evidence.

Volatility-aware methods lowered short-horizon CRPS versus HS by about 2–3%
(DM p < 0.01); differences at h20/h60 were not significant. A-EVT at h1 satisfied
the original distribution-skill rule: CRPS 0.6% below EWMA (p = 0.022), 90%
coverage 90.4% (p = 0.69). A-Ridge had the same CRPS but slightly excessive
coverage, 91.8% (p = 0.0495). A-Ridge and linear quantile regression passed the
risk-skill rule at h1 (FZ0 p = 0.039/0.035); A-EVT narrowly did not (p = 0.053).

EVT improved upper-tail risk estimates. At h1, GARCH's 1% upper-tail breach rate
was 1.9% (Kupiec p = 0.012; Z2 p = 0.006); GARCH-EVT's was 1.4% (p = 0.26) and
A-EVT's 0.8% (p = 0.65; Z2 p = 0.61). HS breaches clustered during the 2026 shock
despite acceptable overall rates. Long horizons remained asymmetric: 2.5%
lower-tail breaches were 0–0.9%, versus 4–7% in the upper tail. Overlapping
targets limited the tests' power. Direction forecasts had little skill, and
median point forecasts were close to the random walk; direct quantile trees
were worse.

### Archived h1 excerpt

| Method | CRPS / Historical simulation | CRPS / EWMA (DM p) | 90% coverage | FZ0 − EWMA (p) | 2.5% ES，long / short (USD/barrel) |
|---|---|---|---|---|---|
| Historical simulation | 1.000 | 1.023 (0.0002) | 90.8% | +0.308 (<0.001) | 7.01 / 7.57 |
| EWMA | 0.977 | — | 89.7% | — | 5.24 / 4.55 |
| GARCH | 0.977 | 1.000 (0.90) | 88.8% | −0.015 (0.35) | 5.12 / 4.29 |
| GARCH-EVT | 0.979 | 1.002 (0.32) | 89.3% | +0.021 (0.33) | 5.06 / 4.41 |
| A-Ridge | 0.971 | 0.994 (0.022) | 91.8% | −0.055 (0.039) | 5.10 / 4.54 |
| A-EVT | 0.971 | 0.994 (0.022) | 90.4% | −0.053 (0.053) | 5.17 / 4.71 |
| Linear quantile regression | 0.976 | 0.998 (0.57) | 90.7% | −0.058 (0.035) | 4.93 / 4.69 |
| Ensemble | 0.973 | 0.996 (0.11) | 91.2% | −0.041 (0.081) | 5.14 / 4.68 |

HS was worse than EWMA in each year and volatility regime at h1/h5. A-EVT's h1
advantage was concentrated in high volatility (relative CRPS 0.992) and 2026
(0.985), with little difference in quiet periods. The 3.8-year test contains few
independent long-horizon windows. Revised news data are not historical vintages.

Figures in `outputs/figures/`: `test_crps_skill.png`, `test_var_backtest_h1.png`,
`fan_garch.png` and `pit_histograms.png`.

## Post-test research: route-C variance mixture

Route C was proposed after observing the original distribution and risk results.
It passed future-truncation causality checks, but its design was not blind.
The 18-setting grid was chosen on 2016–2022 validation only; it was tested once,
appended without changing other methods' scores, and excluded from the ensemble.
All candidates use the common OVX-supported sample from May 2007.

The decision to offer an OVX combination at h5/h20 was influenced by the old
test: calibrated OVX already led HAR at h60 in validation, but led at h5/h20
only in test. This qualifies the earlier plan's broader claim.

| h | Validation HAR | Validation calibrated OVX | Test HAR | Test calibrated OVX |
|---|---|---|---|---|
| 5 | **0.406** | 0.429 | 0.322 | **0.314** |
| 20 | **0.394** | 0.456 | 0.206 | **0.185** |
| 60 | 0.471 | **0.426** | 0.224 | **0.189** |

The mixture idea and kurtosis diagnostics came from validation data. Route A's
weak long-horizon validation result (h60 CRPS/HS 1.017) was also known before test.

### Validation selection

| h | Variance core | Ridge α | Uncertainty | Validation CRPS / Historical simulation |
|---|---|---|---|---|
| 5 | Equal-weight HAR-GK / calibrated OVX | 1 | State dependent (conditional) | 0.956 |
| 20 | Equal-weight HAR-GK / calibrated OVX | 100 | Empirical errors | 0.972 |
| 60 | Equal-weight HAR-GK / calibrated OVX | 100 | Empirical errors | 0.953 |

Grid scores were close: 0.956–0.957 at h5, 0.972–0.980 at h20 and 0.953–0.962
at h60. The best originally registered methods were A-Ridge at h5 (0.961), and
GARCH at h20/h60 (0.978/0.938).

### Archived test

| h | CRPS / EWMA (DM p) | / GARCH (p) | / A-Ridge (p) | / Historical simulation (p) | 50% coverage (p) | 90% coverage (p) | FZ0 − EWMA (p) | 2.5% breach rate, lower / upper |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.997 (0.47) | 0.991 (0.057) | 0.997 (0.31) | 0.968 (<0.001) | 44.2% (0.008) | 90.8% (0.58) | −0.16 (0.13) | 2.2% / 1.9% |
| 20 | 0.995 (0.69) | 0.992 (0.38) | **0.983 (<0.001)** | 0.981 (0.22) | 51.0% (0.78) | 93.2% (0.16) | −0.20 (0.43) | 0.9% / 1.7% |
| 60 | 0.974 (0.18) | 0.983 (0.16) | **0.964 (<0.001)** | 0.972 (0.027) | 56.6% (0.30) | 92.6% (0.61) | −0.60 (0.31) | 0.0% / 6.2% |

Route C had the lowest long-horizon CRPS and FZ0, but improvements over EWMA and
GARCH were not significant (smallest p = 0.057). It did not meet the registered
skill rule at any horizon. It significantly improved A-Ridge at h20/h60, by
1.7%/3.6% (p < 0.001). The contribution of GK, OVX and symmetric zero-mean shape
was not separately identified.

Its h5 50% interval undercovered at 44.2% (p = 0.008), even though 90% coverage
was acceptable. Heavy tails coexisted with an overly sharp center. At h60 the
2.5% lower/upper breach rates remained 0%/6.2%; a symmetric distribution could
not anticipate the upward 2026 shock. Gains concentrated in high volatility
and 2026; the validation ranking against GARCH did not persist. Its zero median
provides no directional signal. See `test_non_blind.csv` and
`outputs/figures/fan_c_mixture.png`. This remains exploratory evidence, not an
independent confirmation of the selected design.

# Time-series regression

Run commands from the repository root or the module directory. The former `part1/` implementation is now `m1_time_series/`.

| Entry point | Purpose |
| --- | --- |
| `classical.R` | Exploratory plots, initial-training diagnostics, candidate models, and expanding/sliding historical validation; `--final-evaluation` invokes the final protocol without plots |
| `for_test_set.R` | Standalone historical selection, final fit, frozen forecasts, and dated evaluation |
| `validation_selection.R` | Automatically loaded historical selector; saves candidate scores, selected configuration, input/code signatures, and package versions |
| `daily_spot_df.R` | Rebuilds the contemporaneous spot table using forward fill only |

## Evaluation protocol

`qf5210-frozen-october-2026` matches `m3_ml/common/protocol.py`: initial information cutoff 2023-08-31, historical target window 2023-09-01–2026-09-30, final information cutoff 2026-09-30, and October targets 2026-10-01–2026-10-23. Horizons count quoted rows, not calendar days.

The scripts retain two modes for audit. `fixed_origin` issues one 60-step path from September 30 without October inputs. `daily_update`, the final product mode, updates observable state but reuses frozen coefficients, innovation variance, GARCH shape, and VECM dynamics. Only realized target dates inside the evaluation window are scored. Unobserved 20/60-step targets retain numerical forecasts but have missing target dates/actuals and `n=0` metrics.

### Availability and state

`data/processed/wti_publication.csv` provides quote availability. Matched dated WPSR evidence uses the next federal business day; unmatched quotes use an assumed fifth federal business day after the quote week's Friday. The latter is not independently verified historical publication evidence.

At each origin, models consume the longest contiguous published prefix of the original quote grid. They stop at an unpublished internal gap rather than deleting that row and compressing time. Forecasts bridge the gap before extracting the requested origin-plus-horizon target. For example, September 21–25 quotes have declared availability on October 1. A September 30 fit therefore does not observe the September 30 spot price.

Companion VECM spot series use the same publication schedule as WTI as a shared-calendar assumption. ARIMAX uses processed S&P 500, OVX, and dollar candidates; future exogenous paths hold origin-known values constant. The canonical `origin_price` and RW use the latest quote date actually available at the origin, not a potentially stale `wti_spot_last_published` feature. Late older reports never rewind that anchor. On October 9, this gives the October 2 quote 97.89 rather than the old feature 85.23. A model's contiguous state can still end earlier than that anchor.

The analysis uses current historical data vintages and explicit publication assumptions. It is not a complete real-time vintage backtest, and previously inspected October data are not a new blind test.

## Historical selection

1. Fit each candidate once on the contiguous prefix available by 2023-08-31. During selection validation, freeze coefficients and update only causal state.
2. Forecast 1/5/20/60 quoted-day targets at historical origins. Score only targets inside the historical window and available by 2026-09-30.
3. Use a common target schedule across candidates. Average the four horizon-specific RMSE/RW-RMSE ratios equally; lower is better. Break ties by complexity and candidate name.
4. Select one valid candidate in each family, then refit the four winners and independent RW at the final cutoff. Failed/incomplete candidates retain their failure reasons and cannot win.

The grid has 37 candidates: five ARIMA specifications including drifted RW; four S&P500/OVX/dollar combinations times five ARIMAX orders; four differenced ARMA–GARCH(1,1) Student-t models; and eight restricted-constant VECMs with two to four variables and the original K/rank combinations. Expanding/sliding exploratory validation is a separate workflow.

`validation_selection_2023-09-01_2026-09-30.rds` stores the rule, grid, input/code MD5s, package versions, and selection object. Cached selection is reused only when these signatures match. Input signatures exclude post-September data. A changed `--as-of` date does not select a new model; incompatible revisions require an intentional new output directory.

Forecasts are conditional means, with `q025/q05/q50/q95/q975` also exported. GARCH uses 10,000 simulated paths by default and seed 5210 plus the origin row; its simulated median is not mislabeled as a mean. Other models use conditional Gaussian quantiles. Parameter-estimation uncertainty is excluded.

GARCH uses `nlminb+nm` on the same Student-t likelihood, fit-period scaling restored to dollar units, and `control=list(xscale=TRUE,tol1=1e4,tol2=1)`. The relative/parameter tolerance is 1e-10 and the Nelder–Mead relative tolerance 1e-11. Convergence code zero, finite likelihood, and stationarity are required. Optimizer, tolerance, scale, and convergence evidence are saved. See the fGarch [solver source](https://github.com/cran/fGarch/blob/master/R/garch-Solver.R) and [likelihood implementation](https://github.com/cran/fGarch/blob/master/R/loglik.R). Coefficients, unstandardized residuals, and conditional standard deviations are read from the declared S4 slots; execution does not depend on interactive search-path changes.

## Running and outputs

Install R packages `forecast`, `fGarch`, `vars`, and `urca`, then run:

```powershell
Rscript m1_time_series/for_test_set.R --as-of=2026-10-23
Rscript m1_time_series/classical.R --final-evaluation
```

Options include `--root`, `--output`, `--as-of`, `--simulations`, `--seed`, `--bundle`, and `--selection-only=true`. Default output is `m1_time_series/outputs/frozen_october_2026/`. Preserve the approved folder; use a separate output directory for experiments.

| Output | Contents |
| --- | --- |
| `validation_selection_scores.csv`, `validation_selection_by_horizon.csv`, `selected_config.csv` | Scores, sample counts, failures, and selected specifications |
| `validation_selection_2023-09-01_2026-09-30.rds` | Reproducible selection state |
| `frozen_fit_selected_2026-09-30.rds` | Final fitted models and covariance/state |
| `forecasts_2026-10-01_2026-10-23_asof_<date>.csv` | Mode, origin/target, forecasts, quantiles, actuals, score flags, and publication audit; row indexes are zero-based for Python alignment |
| `metrics_...csv`, `metrics_all_horizons_...csv` | RMSE, MAE, MAPE, 95% coverage/width/interval score; `n_mape` excludes zero actuals |
| `frozen_parameters_2026-09-30.csv`, `manifest_...csv` | Parameters, dates, assumptions, sample counts, and random settings |

## Recorded results

The 2026-10-29 integration run completed actual 37-candidate selection, final fitting, and evaluation in 1,182.41 seconds. These winners supersede earlier fixed-specification smoke runs.

| Family | Selected method | Mean historical RMSE/RW ratio |
| --- | --- | ---: |
| ARIMA | `ARIMA211`, no drift | 0.992457724405346 |
| ARIMAX | `SP500_OVX_ARIMA011`, no drift | 0.990580567049404 |
| GARCH | `GARCH_ARIMA111`, Student-t | 0.998642876556623 |
| VECM | `WTI_Brent_Gasoline_K3_R2` | 0.974629475488100 |

Selection input MD5: `06519504cbeffe7450db9542e13b1135`; selection signature: `ecb3bdea29dcc1fcebaaa3eb16c97184`.

| Method | Fixed h1 RMSE | Fixed h5 RMSE | Rolling h1 RMSE | Rolling h5 RMSE |
| --- | ---: | ---: | ---: | ---: |
| RW | 1.670000 | 6.420000 | 9.110146 | 11.253163 |
| ARIMA211 | 2.148129 | 6.898142 | 8.200725 | 10.096210 |
| SP500_OVX_ARIMA011 | 2.538284 | 7.288284 | 8.450119 | 10.161849 |
| GARCH_ARIMA111 | 1.758797 | 6.509267 | 8.907202 | 10.999953 |
| WTI_Brent_Gasoline_K3_R2 | 4.318543 | 10.003619 | 6.330858 | 5.574177 |

Units are USD/barrel. Fixed-origin counts are 1/1/0/0 and rolling counts 17/13/0/0 at h1/5/20/60. A single fixed-origin target cannot support the same statistical interpretation as a multi-target rolling sample.

Recorded checks reused the frozen checkpoint with `--as-of=2026-10-22` (rolling n=16/12), perturbed all October prices/exogenous inputs by +1000 without changing the September fixed path, and perturbed unpublished quotes without changing the checked origins. All checked means/quantiles and checkpoint hashes remained unchanged. These finite checks do not prove all historical vintages or release assumptions.

## Exploratory workflow

`classical.R` without `--final-evaluation` additionally uses `ggplot2`, `dplyr`, `fBasics`, `tseries`, `lmtest`, `FinTS`, `moments`, `tidyr`, and `MTS`. Diagnostics use initial available training data. Expanding/sliding validation refits at historical origins with the same publication-prefix rule, ends before October, and uses common complete 60-step paths, excluding the last 59 potential origins. Older plots/slides remain historical artifacts and are not new frozen-protocol results.

Inputs are the main table, dictionary, publication schedule, and `data/time_series/daily_spot.csv`. `Rscript m1_time_series/daily_spot_df.R` overwrites the contemporaneous table; run it only when rebuilding that table intentionally.

## Public source export

The serialized `validation_selection_2023-09-01_2026-09-30.rds` cache is not distributed because it embeds machine-specific source metadata. The selection runner regenerates it if needed. Selection summaries, the chosen specification and the final frozen fit remain available.

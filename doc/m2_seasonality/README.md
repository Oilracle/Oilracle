# Seasonality

`m2_seasonality/seasonality.R` studies month/weekday effects, 3×IQR sensitivity, HAC inference, and periodic STL on monthly/weekly means.

## Frozen evaluation protocol

| Stage | Fit information cutoff | Evaluation target dates |
| --- | --- | --- |
| Historical validation | 2023-08-31 | 2023-09-01–2026-09-30 |
| Final evaluation | 2026-09-30 | 2026-10-01–2026-10-23 |

The protocol ID is `qf5210-frozen-october-2026`. Each stage fits once. October never refits coefficients, IQR thresholds, STL templates, residual quantiles, or model selection. Historical `daily_update` RMSE selects methods separately by target type and horizon. All methods remain exported with `selected_on_validation` flags.

`fixed_origin` retains one complete 1–60 quoted-row path for audit; `daily_update`, the final product mode, issues h1/5/20/60 forecasts at the cutoff and subsequent quoted-day origins. State updates use published data only. R row indexes are one-based and `target_row = origin_row + horizon`. `effective_steps_from_anchor` records the full span from the observable anchor; unavailable rows are never deleted to compress time.

## Publication and label maturity

Inputs are `data/processed/wti_daily.csv` and `wti_publication.csv`, matched by date and target value. Every fit/origin uses the longest contiguous available prefix. An internal unpublished quote blocks later isolated quotes from the model state, preserving genuinely adjacent changes. `matching_dated_archive` identifies WPSR evidence; `assumed_weekly_release` uses the fifth US federal business day after the observation week's Friday. Output provenance records effective fit end, anchor evidence, and assumed/matched counts.

Selection labels must be available by 2026-09-30 as well as fall inside the target window. A price label needs its quote; an absolute daily change needs the target and immediately preceding original quote; a weekly/monthly mean needs every quote in the complete period. Missing evidence or an immature quote makes the label unscorable. Such rows retain forecasts but have missing actuals, `scored=FALSE`, and `status=label_unavailable_at_selection`. Audit columns include `label_available_date`, `selection_cutoff`, and `label_mature_at_selection`. October actuals can be scored retrospectively after publication; the historical selection gate is not incorrectly applied to them.

This uses current historical vintages and declared release assumptions, not a complete real-time data archive. Previously viewed October outcomes are retrospective evaluation.

## Models and targets

- Month, weekday, and joint month-plus-weekday mean regressions, on raw data and samples filtered by fit-only 3×IQR thresholds.
- Means, standard deviations, median-based Levene tests, and Newey–West HAC inference with `lag=10, prewhite=FALSE`. HAC changes coefficient covariance, not OLS predictions.
- Absolute daily change forecasts from constant means, weekday means, and joint calendar regressions. The target is `abs(P_t-P_(t-1))`, not cumulative change, price, or annualized volatility; its RMSE must not be ranked against price RMSE.
- Price forecasts from RW, accumulated calendar mean changes, and STL seasonal-adjusted RW: `anchor - S(anchor_date) + S(target_date)`. The daily bridge uses a seasonal template and is distinct from predicting a monthly/weekly mean.
- Native monthly/weekly mean RW and STL-adjusted RW. STL is robust and periodic, with frequencies 12 and 52. Weekly indexing uses a continuous weekly grid rather than ISO week numbers with occasional week 53. Only fully completed periods in the published prefix are aggregated, with `n_quotes` recorded. Updates change the latest completed-period mean; they never refit STL during validation or October.

Frozen empirical residuals produce `q05/q50/q95`. Price residuals use the complete anchor-to-target span, absolute changes use one-day targets, and aggregate residuals use period steps. These are empirical uncertainty estimates, not guaranteed October coverage.

## Unobserved targets and counting

Beyond the observed quote table, forecasts remain numerical but `target_date` and `actual` are missing and `scored=FALSE`. Calendar methods use explicitly marked future-weekday scenarios (`scenario_date`, `calendar_source=assumed_future_weekdays`), not claimed knowledge of future trading holidays. Native aggregates must be complete and wholly inside the evaluation window; a partial October monthly mean is not an actual monthly target.

Metrics distinguish `n_forecasts`, scored `n`, `n_distinct_targets`, and `n_distinct_periods`. Repeated origins predicting the same week are not independent weeks. Empty samples have missing error metrics, not zeros.

## Running

Install R packages `car`, `sandwich`, and `lmtest`; the script does not install packages itself.

```powershell
Rscript -e "install.packages(c('car','sandwich','lmtest'), repos='https://cloud.r-project.org')"
Rscript m2_seasonality/seasonality.R
Rscript m2_seasonality/seasonality.R m2_seasonality/outputs/frozen_october_2026_smoke
```

Default outputs are under `m2_seasonality/outputs/frozen_october_2026/`: validation/final predictions and metrics, selected methods, `frozen_fits.rds`, calendar/IQR diagnostics, OLS/Levene/HAC tests, coefficients, STL components/strength, and fit provenance. Use a separate output directory for new runs.

### Rescoring mature validation labels without fitting

```powershell
Rscript m2_seasonality/seasonality.R --rescore-validation
Rscript m2_seasonality/seasonality.R --rescore-validation m2_seasonality/outputs/frozen_october_2026_smoke
```

This requires a compatible frozen RDS and both prediction CSVs. It reapplies `apply_label_maturity()`, masks immature validation actuals, recomputes validation selection, updates final selection flags, and synchronizes RDS selection metadata. It does not call fitting, forecasting, or diagnostics, and does not change fit objects, forecasts, quantiles, quote rows, or origins. It rewrites prediction/metric/selection files and the RDS, so the complete RDS hash can change even when the fitted objects do not. Compare serialized `initial_fit` and `final_fit` objects with serialization version 2: R formula environments can have different reference identities after reading. The comparison includes coefficients, residuals, QR, model frames, STL, IQR, attributes, and captured bindings without numerical tolerances.

Rescoring uses the same table/publication evidence, does not restore previously masked actuals from changed inputs, and cannot join incompatible protocols or quote grids. The recorded integration check preserved both serialized fits, all forecasts, and October scores; mature historical price counts were 758/754/739/699 at h1/5/20/60. The check proves the compared snapshots, not an unavailable earlier snapshot.

## Recorded validation

The complete script ran successfully in about 144 seconds. Frozen-state mutation checks took about 72.97 seconds: adding 1000 to all October prices/absolute changes left fixed-origin forecasts and quantiles unchanged; changing an unpublished final quote left both checked modes unchanged. The fitted RDS hash did not change.

| Mode | Horizons | Scored targets |
| --- | --- | --- |
| daily_update | 1 / 5 / 20 / 60 | 17 / 13 / 0 / 0 |
| fixed_origin | 1 / 5 / 17 | One each |
| fixed_origin | 20 / 60 | Zero each |

No complete native monthly target is scored. Native weekly rolling h1 has ten scored forecasts for two distinct target weeks. Final fit provenance ends at 2026-09-18: 3,778 quotes (3,716 matched archives and 62 assumed releases), 180 complete months, and 788 complete weeks. The September 30 information cutoff is not the latest observable price date.

Earlier `wti_*.csv` exports and legacy module slides describe historical workflows. Expanding-window STL decompositions must not be mixed with frozen-template results. The final presentation reads the approved frozen outputs.

## Reproduction check

The following retained R check independently verifies maturity dates and unchanged fitted objects during a rescore. Run it from the repository root on a copy of the output directory if preserving exact artifact bytes is required.

```r
out <- "m2_seasonality/outputs/frozen_october_2026"
load_csv <- function(name) read.csv(file.path(out, name), stringsAsFactors = FALSE,
                                    na.strings = c("", "NA"))
before_fit <- readRDS(file.path(out, "frozen_fits.rds"))
frozen_bytes <- function(fit) serialize(fit, NULL, version = 2)
before_initial <- frozen_bytes(before_fit$initial_fit)
before_final <- frozen_bytes(before_fit$final_fit)
before_v <- load_csv("validation_predictions.csv")
before_f <- load_csv("final_predictions.csv")
before_metrics <- load_csv("final_metrics.csv")
stopifnot(system2(file.path(R.home("bin"), "Rscript"),
                 c("m2_seasonality/seasonality.R", "--rescore-validation", out)) == 0L)
after_fit <- readRDS(file.path(out, "frozen_fits.rds"))
v <- load_csv("validation_predictions.csv")
f <- load_csv("final_predictions.csv")
stopifnot(identical(before_initial, frozen_bytes(after_fit$initial_fit)),
          identical(before_final, frozen_bytes(after_fit$final_fit)))
mutable <- c("actual", "scored", "status", "label_available_date",
             "selection_cutoff", "label_mature_at_selection", "selected_on_validation")
same <- function(a, b, columns) {
  isTRUE(all.equal(a[columns], b[columns], check.attributes = FALSE))
}
stopifnot(same(before_v, v, setdiff(names(before_v), mutable)),
          same(before_f, f, setdiff(names(before_f), mutable)),
          same(before_f, f, c("actual", "scored", "status")),
          isTRUE(all.equal(before_metrics, load_csv("final_metrics.csv"),
                           check.attributes = FALSE)))
quotes <- read.csv("data/processed/wti_daily.csv")
quotes <- quotes[order(as.Date(quotes$date)), ]
quotes <- quotes[as.Date(quotes$date) <= as.Date("2026-10-23"), ]
pub <- read.csv("data/processed/wti_publication.csv")
dates <- as.Date(quotes$date)
available <- as.Date(pub$available_date[match(quotes$date, pub$date)])
cutoff <- as.Date("2026-09-30")
expected <- rep(as.Date(NA), nrow(v))
for (i in seq_len(nrow(v))) {
  kind <- v$target_type[i]
  row <- v$target_row[i]
  needed <- if (kind == "price") row else if (kind == "absolute_daily_change") {
    c(row - 1L, row)
  } else {
    which(dates >= as.Date(v$period_start[i]) & dates <= as.Date(v$period_end[i]))
  }
  if (length(needed) && all(needed >= 1L & needed <= length(dates))) {
    releases <- available[needed]
    if (!anyNA(releases)) expected[i] <- max(releases)
  }
}
# Unreached daily targets and incomplete future periods have no actual label.
expected[is.na(v$target_date) | v$target_date == ""] <- as.Date(NA)
aggregate <- v$target_type %in% c("monthly_average", "weekly_average")
expected[aggregate & !is.na(v$period_end) &
         as.Date(v$period_end) > as.Date("2026-10-23")] <- as.Date(NA)
stopifnot(isTRUE(all.equal(as.Date(v$label_available_date), expected)),
          all(v$label_mature_at_selection == (!is.na(expected) & expected <= cutoff)),
          all(!v$scored | v$label_mature_at_selection),
          all(is.na(v$actual[!v$label_mature_at_selection])))
selected <- after_fit$selection
key <- paste(selected$target_type, selected$horizon, selected$method)
stopifnot(all(f$selected_on_validation ==
              (paste(f$target_type, f$horizon, f$method) %in% key)))
print(subset(load_csv("validation_metrics.csv"),
             mode == "daily_update" & target_type == "price" & method == "random_walk"))
```

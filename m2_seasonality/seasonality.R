# QF5210 WTI calendar/HAC/STL models: frozen-fit validation and final evaluation.
# Run with Rscript m2_seasonality/seasonality.R [--rescore-validation] [output_directory].

TRAIN_END <- as.Date("2023-08-31")
VALIDATION_START <- as.Date("2023-09-01")
VALIDATION_END <- as.Date("2026-09-30")
FIT_END <- VALIDATION_END
TEST_START <- as.Date("2026-10-01")
TEST_END <- as.Date("2026-10-23")
HORIZONS <- c(1L, 5L, 20L, 60L)
MODES <- c("fixed_origin", "daily_update")
PROTOCOL_ID <- "qf5210-frozen-october-2026"
AVAILABILITY_POLICY <- paste(
  "WTI publication evidence; unmatched weekly-release dates are explicit assumptions;",
  "latest contiguous available quoted-row prefix; current-vintage retrospective evaluation")

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[1]), mustWork = TRUE))
} else if (file.exists("m2_seasonality/seasonality.R")) {
  normalizePath("m2_seasonality")
} else {
  normalizePath(".")
}
root <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)
args <- commandArgs(trailingOnly = TRUE)
rescore_only <- length(args) > 0L && args[1] == "--rescore-validation"
if (rescore_only) args <- args[-1L]
output_dir <- if (length(args)) args[1] else {
  file.path(root, "m2_seasonality", "outputs", "frozen_october_2026")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
export <- function(x, name) {
  write.csv(x, file.path(output_dir, name), row.names = FALSE, na = "")
}
calendar <- function(dates) {
  data.frame(
    date = dates,
    month = factor(as.integer(format(dates, "%m")), levels = 1:12),
    weekday = factor(as.POSIXlt(dates)$wday, levels = 1:5,
                     labels = c("Mon", "Tue", "Wed", "Thu", "Fri"))
  )
}
raw <- read.csv(file.path(root, "data", "processed", "wti_daily.csv"),
                stringsAsFactors = FALSE)
oil <- data.frame(date = as.Date(raw$date), price = as.numeric(raw$target_wti))
oil <- oil[order(oil$date), ]
rownames(oil) <- NULL
if (anyNA(oil) || anyDuplicated(oil$date) || any(!is.finite(oil$price))) {
  stop("WTI quoted rows require unique valid dates and finite prices; no test-time imputation is allowed.")
}
if (any(!as.POSIXlt(oil$date)$wday %in% 1:5)) stop("Unexpected weekend quote.")
if (!any(oil$date == TRAIN_END) || !any(oil$date == FIT_END)) {
  stop("The initial and final fit cutoff quotes must exist.")
}
oil <- oil[oil$date <= TEST_END, ]
oil$diff_price <- c(NA_real_, diff(oil$price))
oil$abs_change <- abs(oil$diff_price)
oil <- cbind(oil, calendar(oil$date)[, c("month", "weekday")])
publication <- read.csv(file.path(root, "data", "processed", "wti_publication.csv"),
                        stringsAsFactors = FALSE)
publication$date <- as.Date(publication$date)
publication$available_date <- as.Date(publication$available_date)
if (anyDuplicated(publication$date)) stop("Duplicate dates in WTI publication evidence.")
match_index <- match(oil$date, publication$date)
if (anyNA(match_index) ||
    any(oil$price != as.numeric(publication$target_wti[match_index]))) {
  stop("Publication evidence must match every input WTI date and value.")
}
oil$available_date <- publication$available_date[match_index]
oil$availability_evidence <- publication$evidence[match_index]
observable_prefix <- function(origin_date) {
  eligible <- !is.na(oil$available_date) & oil$available_date <= origin_date & oil$date <= origin_date
  unavailable <- which(!eligible)
  last <- if (length(unavailable)) unavailable[1] - 1L else nrow(oil)
  if (last < 2L) stop("Insufficient contiguous published WTI history at ", origin_date)
  last
}

# Labels used to choose a method must be published by the selection cutoff.
# Final-window actuals are retrospective; this restriction never masks October.
# This same function is used for fresh forecasts and frozen-output replay.
apply_label_maturity <- function(predictions) {
  available <- rep(as.Date(NA), nrow(predictions))
  daily <- predictions$target_type %in% c("price", "absolute_daily_change")
  known <- daily & !is.na(predictions$target_date) &
    predictions$target_row >= 1L & predictions$target_row <= nrow(oil)
  rows <- predictions$target_row[known]
  if (any(as.Date(predictions$target_date[known]) != oil$date[rows])) {
    stop("Prediction target rows no longer match the raw quoted-row calendar.")
  }
  available[known] <- oil$available_date[rows]
  absolute <- which(known & predictions$target_type == "absolute_daily_change")
  if (length(absolute)) {
    rows <- predictions$target_row[absolute]
    previous <- rep(as.Date(NA), length(rows))
    previous[rows > 1L] <- oil$available_date[rows[rows > 1L] - 1L]
    # Missing publication evidence for either quote leaves the change immature.
    available[absolute] <- pmax(available[absolute], previous)
  }
  aggregate_rows <- which(predictions$target_type %in% c("monthly_average", "weekly_average"))
  if (length(aggregate_rows)) {
    starts <- as.Date(predictions$period_start[aggregate_rows])
    ends <- as.Date(predictions$period_end[aggregate_rows])
    keys <- paste(starts, ends)
    periods <- !duplicated(keys)
    period_available <- rep(as.Date(NA), sum(periods))
    for (j in seq_along(period_available)) {
      start <- starts[periods][j]
      end <- ends[periods][j]
      if (is.na(start) || is.na(end) || end > max(oil$date)) next
      period_rows <- oil$date >= start & oil$date <= end
      dates <- oil$available_date[period_rows]
      if (length(dates) && !anyNA(dates)) period_available[j] <- max(dates)
    }
    available[aggregate_rows] <- period_available[match(keys, keys[periods])]
  }
  validation <- predictions$phase == "validation"
  mature <- !is.na(available) & available <= VALIDATION_END
  predictions$label_available_date <- available
  predictions$selection_cutoff <- rep(as.Date(NA), nrow(predictions))
  predictions$selection_cutoff[validation] <- VALIDATION_END
  predictions$label_mature_at_selection <- ifelse(validation, mature, NA)
  in_window <- !is.na(predictions$target_date) &
    as.Date(predictions$target_date) >= VALIDATION_START &
    as.Date(predictions$target_date) <= VALIDATION_END
  in_window[aggregate_rows] <- in_window[aggregate_rows] &
    as.Date(predictions$period_start[aggregate_rows]) >= VALIDATION_START &
    as.Date(predictions$period_end[aggregate_rows]) <= VALIDATION_END
  immature <- validation & in_window & !mature
  predictions$actual[validation & !mature] <- NA_real_
  predictions$status[immature] <- "label_unavailable_at_selection"
  predictions$scored <- is.finite(predictions$actual) & is.finite(predictions$forecast)
  predictions
}

# Aggregates are admitted only after the whole calendar period has ended.
# The first partial month/week of the input is excluded, as in the original analysis.
period_start <- function(dates, kind) {
  if (kind == "monthly") return(as.Date(format(dates, "%Y-%m-01")))
  dates - ((as.POSIXlt(dates)$wday + 6L) %% 7L)
}
period_end <- function(starts, kind) {
  if (kind == "weekly") return(starts + 6L)
  as.Date(format(starts + 32L, "%Y-%m-01")) - 1L
}
aggregate_prices <- function(history, cutoff, kind) {
  starts <- period_start(history$date, kind)
  values <- aggregate(history$price, list(start = starts), mean)
  names(values)[2] <- "price"
  counts <- aggregate(history$price, list(start = starts), length)
  values$n_quotes <- counts$x[match(values$start, counts$start)]
  values$end <- period_end(values$start, kind)
  values <- values[values$start >= min(history$date) & values$end <= cutoff, ]
  values[order(values$start), ]
}

fit_stl <- function(history, cutoff, kind) {
  values <- aggregate_prices(history, cutoff, kind)
  frequency <- if (kind == "monthly") 12L else 52L
  if (nrow(values) <= 2L * frequency) stop("Insufficient complete periods for ", kind, " STL.")
  if (kind == "monthly") {
    expected <- seq(values$start[1], tail(values$start, 1), by = "month")
    start <- c(as.integer(format(values$start[1], "%Y")),
               as.integer(format(values$start[1], "%m")))
  } else {
    expected <- seq(values$start[1], tail(values$start, 1), by = "week")
    start <- c(1L, 1L)
  }
  if (!identical(as.numeric(expected), as.numeric(values$start))) {
    stop("Missing complete ", kind, " period; do not interpolate using future data.")
  }
  series <- ts(values$price, start = start, frequency = frequency)
  decomposition <- stl(series, s.window = "periodic", robust = TRUE)
  components <- decomposition$time.series
  template <- as.numeric(tapply(components[, "seasonal"], cycle(series), mean))
  values$trend <- as.numeric(components[, "trend"])
  values$seasonal <- as.numeric(components[, "seasonal"])
  values$remainder <- as.numeric(components[, "remainder"])
  denominator <- var(values$seasonal + values$remainder)
  strength <- if (denominator > 0) max(0, 1 - var(values$remainder) / denominator) else 0
  list(kind = kind, values = values, template = template,
       start = values$start[1], frequency = frequency, strength = strength)
}
season_at <- function(model, dates) {
  if (model$kind == "monthly") {
    index <- as.integer(format(dates, "%m"))
  } else {
    # Preserve the original 52-observation weekly cycle, rather than silently
    # replacing it with ISO week numbers (which can have 53 weeks).
    index <- (as.integer(period_start(dates, "weekly") - model$start) %/% 7L) %% 52L + 1L
  }
  model$template[index]
}

# Only these two calls fit anything. The validation fit and final fit have
# independently estimated, then frozen, IQR thresholds and STL templates.
fit_models <- function(cutoff) {
  last <- observable_prefix(cutoff)
  history <- oil[seq_len(last), ]
  training <- history[is.finite(history$diff_price), ]
  quartiles <- quantile(training$diff_price, c(0.25, 0.75), names = FALSE)
  limits <- quartiles + c(-3, 3) * diff(quartiles)
  extreme <- training$diff_price < limits[1] | training$diff_price > limits[2]
  filtered <- training[!extreme, ]
  list(cutoff = cutoff, effective_fit_end = tail(history$date, 1),
       history = history, training = training, filtered = filtered,
       availability_policy = AVAILABILITY_POLICY,
       limits = limits, extremes = training[extreme, ],
       mean_original = lm(diff_price ~ month + weekday, data = training),
       mean_3iqr = lm(diff_price ~ month + weekday, data = filtered),
       abs_original = lm(abs_change ~ month + weekday, data = training),
       abs_3iqr = lm(abs_change ~ month + weekday, data = filtered),
       weekday_original = lm(abs_change ~ weekday, data = training),
       weekday_3iqr = lm(abs_change ~ weekday, data = filtered),
       baseline_abs = mean(training$abs_change),
       monthly = fit_stl(history, tail(history$date, 1), "monthly"),
       weekly = fit_stl(history, tail(history$date, 1), "weekly"))
}

# Calendar hypothesis tests retain the original specifications and lag-10
# Newey-West covariance. HAC changes inference, not the fitted coefficients.
export_diagnostics <- function(fit, phase) {
  result <- list()
  coefficients <- list()
  groups <- list()
  for (variant in c("original", "3iqr")) {
    training <- if (variant == "original") fit$training else fit$filtered
    add_test <- function(name, p) {
      result[[length(result) + 1L]] <<- data.frame(
        phase = phase, fit_end = fit$cutoff, variant = variant,
        test = name, p_value = as.numeric(p))
    }
    for (group in c("month", "weekday")) {
      model <- lm(reformulate(group, response = "diff_price"), data = training)
      statistic <- summary(model)$fstatistic
      add_test(paste(group, "mean_OLS"),
               pf(statistic[1], statistic[2], statistic[3], lower.tail = FALSE))
      levene <- car::leveneTest(reformulate(group, response = "diff_price"),
                                data = training, center = median)
      add_test(paste(group, "Levene"), levene[1, "Pr(>F)"])
      summary_group <- aggregate(training$diff_price, list(level = training[[group]]),
                                 function(x) c(mean = mean(x), sd = sd(x), n = length(x)))
      groups[[length(groups) + 1L]] <- data.frame(
        phase = phase, variant = variant, calendar = group,
        level = as.character(summary_group$level), summary_group$x)
    }
    for (response in c("mean", "abs")) {
      model <- fit[[paste(response, variant, sep = "_")]]
      covariance <- sandwich::NeweyWest(model, lag = 10, prewhite = FALSE)
      for (group in c("month", "weekday")) {
        hypothesis <- car::linearHypothesis(
          model, grep(paste0("^", group), names(coef(model)), value = TRUE),
          vcov. = covariance, test = "F")
        add_test(paste(group, response, "HAC"), hypothesis[2, "Pr(>F)"])
      }
      table <- lmtest::coeftest(model, vcov. = covariance)
      coefficients[[length(coefficients) + 1L]] <- data.frame(
        phase = phase, fit_end = fit$cutoff, variant = variant, response = response,
        term = rownames(table), estimate = table[, 1], hac_se = table[, 2],
        statistic = table[, 3], p_value = table[, 4], row.names = NULL)
    }
  }
  export(do.call(rbind, result), paste0(phase, "_calendar_tests.csv"))
  export(do.call(rbind, coefficients), paste0(phase, "_hac_coefficients.csv"))
  export(do.call(rbind, groups), paste0(phase, "_calendar_sensitivity.csv"))
  export(fit$extremes, paste0(phase, "_extreme_observations.csv"))
  export(data.frame(phase = phase, fit_end = fit$cutoff,
                    lower = fit$limits[1], upper = fit$limits[2]),
         paste0(phase, "_iqr_thresholds.csv"))
  for (kind in c("monthly", "weekly")) {
    export(fit[[kind]]$values, paste0(phase, "_", kind, "_stl_components.csv"))
  }
  export(data.frame(kind = c("monthly", "weekly"),
                    strength = c(fit$monthly$strength, fit$weekly$strength)),
         paste0(phase, "_stl_strength.csv"))
  export(data.frame(protocol_id = PROTOCOL_ID, phase = phase,
                    availability_policy = AVAILABILITY_POLICY, fit_end = fit$cutoff,
                    effective_fit_end = fit$effective_fit_end,
                    n_quote_labels = nrow(fit$history), n_changes = nrow(fit$training),
                    n_verified_labels = sum(fit$history$availability_evidence == "matching_dated_archive"),
                    n_assumed_labels = sum(fit$history$availability_evidence == "assumed_weekly_release"),
                    monthly_complete_periods = nrow(fit$monthly$values),
                    weekly_complete_periods = nrow(fit$weekly$values)),
         paste0(phase, "_fit_provenance.csv"))
}

# Empirical residual quantiles use the fitting sample only. They are frozen
# uncertainty estimates, not October-calibrated confidence/coverage claims.
residual_quantiles <- function(errors) {
  quantile(errors[is.finite(errors)], c(0.05, 0.5, 0.95), names = FALSE)
}
price_error_quantiles <- function(fit, method, horizon) {
  history <- fit$history
  n <- nrow(history)
  starts <- seq_len(n - horizon)
  ends <- starts + horizon
  prediction <- history$price[starts]
  if (method %in% c("calendar_mean_original", "calendar_mean_3iqr")) {
    variant <- sub("calendar_mean_", "", method)
    changes <- as.numeric(predict(fit[[paste0("mean_", variant)]], newdata = history))
    cumulative <- c(0, cumsum(changes))
    prediction <- prediction + cumulative[ends + 1L] - cumulative[starts + 1L]
  } else if (grepl("^stl_", method)) {
    kind <- if (method == "stl_monthly_daily_bridge") "monthly" else "weekly"
    prediction <- prediction - season_at(fit[[kind]], history$date[starts]) +
      season_at(fit[[kind]], history$date[ends])
  }
  residual_quantiles(history$price[ends] - prediction)
}
aggregate_error_quantiles <- function(model, steps, seasonal) {
  values <- model$values
  if (steps >= nrow(values)) stop("Aggregate horizon exceeds fitting history.")
  starts <- seq_len(nrow(values) - steps)
  ends <- starts + steps
  prediction <- values$price[starts]
  if (seasonal) prediction <- prediction - values$seasonal[starts] + values$seasonal[ends]
  residual_quantiles(values$price[ends] - prediction)
}

# Unreached quoted-row dates stay NA. Numeric calendar forecasts use an
# explicitly assumed weekday scenario, never presented as future quote dates.
quote_dates <- oil$date
scenario_dates <- seq(max(quote_dates) + 1L, by = "day", length.out = 120L)
scenario_dates <- scenario_dates[as.POSIXlt(scenario_dates)$wday %in% 1:5]
forecast_calendar <- c(quote_dates, scenario_dates[seq_len(max(HORIZONS))])

forecast_phase <- function(fit, phase, window_start, window_end) {
  price_methods <- c("random_walk", "calendar_mean_original", "calendar_mean_3iqr",
                     "stl_monthly_daily_bridge", "stl_weekly_daily_bridge")
  abs_methods <- c("baseline_abs", "weekday_original", "weekday_3iqr",
                   "calendar_abs_original", "calendar_abs_3iqr")
  price_quantiles <- new.env(parent = emptyenv())
  frozen_price_quantiles <- function(method, gap) {
    key <- paste(method, gap, sep = "_")
    if (!exists(key, envir = price_quantiles, inherits = FALSE)) {
      assign(key, price_error_quantiles(fit, method, gap), envir = price_quantiles)
    }
    get(key, envir = price_quantiles, inherits = FALSE)
  }
  abs_quantiles <- lapply(abs_methods, function(method) {
    prediction <- if (method == "baseline_abs") rep(fit$baseline_abs, nrow(fit$training)) else {
      key <- sub("calendar_", "", method)
      as.numeric(predict(fit[[key]], newdata = fit$training))
    }
    residual_quantiles(fit$training$abs_change - prediction)
  })
  names(abs_quantiles) <- abs_methods
  rows <- list()
  origin_fixed <- max(which(oil$date <= fit$cutoff))
  target_indices <- which(oil$date >= window_start & oil$date <= window_end)
  # Origins are issue dates, not labels assumed observable on their own date.
  origins_daily <- unique(c(origin_fixed, target_indices))
  for (mode in MODES) {
    origins <- if (mode == "fixed_origin") origin_fixed else origins_daily
    for (origin in origins) {
      origin_date <- oil$date[origin]
      anchor_index <- observable_prefix(origin_date)
      anchor_date <- oil$date[anchor_index]
      anchor <- oil$price[anchor_index]
      aggregate_state <- lapply(c("monthly", "weekly"), function(kind) {
        values <- aggregate_prices(oil[seq_len(anchor_index), ], anchor_date, kind)
        values[nrow(values), ]
      })
      names(aggregate_state) <- c("monthly", "weekly")
      horizons <- if (mode == "fixed_origin") seq_len(max(HORIZONS)) else HORIZONS
      for (h in horizons) {
        target_index <- origin + h
        target_date <- if (target_index <= length(quote_dates)) quote_dates[target_index] else as.Date(NA)
        scenario_date <- forecast_calendar[target_index]
        known_calendar <- !is.na(target_date)
        in_window <- known_calendar && target_date >= window_start && target_date <= window_end
        # Window gating is followed by publication-maturity gating on the
        # completed frame, before any validation score or selection is computed.
        actual_price <- if (in_window && target_index <= nrow(oil)) oil$price[target_index] else NA_real_
        actual_abs <- if (in_window && target_index <= nrow(oil)) oil$abs_change[target_index] else NA_real_
        add <- function(method, target_type, prediction, actual, quantiles,
                        period_start_date = as.Date(NA), period_end_date = as.Date(NA)) {
          lower <- prediction + quantiles[1]
          median <- prediction + quantiles[2]
          upper <- prediction + quantiles[3]
          if (target_type == "absolute_daily_change") {
            prediction <- max(0, prediction)
            lower <- max(0, lower)
            median <- max(0, median)
            upper <- max(0, upper)
          }
          rows[[length(rows) + 1L]] <<- data.frame(
            protocol_id = PROTOCOL_ID, availability_policy = AVAILABILITY_POLICY,
            phase = phase, fit_end = fit$cutoff, effective_fit_end = fit$effective_fit_end,
            mode = mode, origin_date = origin_date, anchor_date = anchor_date,
            origin_price = anchor, anchor_available_date = oil$available_date[anchor_index],
            anchor_evidence = oil$availability_evidence[anchor_index],
            origin_row = origin, target_row = target_index,
            target_date = target_date, scenario_date = scenario_date, horizon = h,
            effective_steps_from_anchor = target_index - anchor_index,
            method = method, target_type = target_type, forecast = prediction,
            actual = actual, q05 = lower, q50 = median, q95 = upper,
            period_start = period_start_date, period_end = period_end_date,
            calendar_source = if (known_calendar) "observed_quote_calendar" else "assumed_future_weekdays",
            scored = is.finite(actual) && is.finite(prediction),
            status = if (!known_calendar) "unreached_assumed_calendar" else if (is.finite(actual)) {
              "realized_in_window"
            } else "unreached_or_outside_window")
        }
        for (method in price_methods) {
          prediction <- anchor
          if (method %in% c("calendar_mean_original", "calendar_mean_3iqr")) {
            dates <- forecast_calendar[seq.int(anchor_index + 1L, target_index)]
            model <- fit[[sub("calendar_", "", method)]]
            prediction <- anchor + sum(predict(model, newdata = calendar(dates)))
          } else if (grepl("^stl_", method)) {
            kind <- if (method == "stl_monthly_daily_bridge") "monthly" else "weekly"
            prediction <- anchor - season_at(fit[[kind]], anchor_date) +
              season_at(fit[[kind]], scenario_date)
          }
          add(method, "price", prediction, actual_price,
              frozen_price_quantiles(method, target_index - anchor_index))
        }
        for (method in abs_methods) {
          prediction <- if (method == "baseline_abs") fit$baseline_abs else {
            as.numeric(predict(fit[[sub("calendar_", "", method)]], newdata = calendar(scenario_date)))
          }
          add(method, "absolute_daily_change", prediction, actual_abs, abs_quantiles[[method]])
        }
        for (kind in c("monthly", "weekly")) {
          model <- fit[[kind]]
          # Only published, complete-period state changes; STL stays frozen.
          last <- aggregate_state[[kind]]
          target_start <- period_start(scenario_date, kind)
          target_end <- period_end(target_start, kind)
          steps <- if (kind == "weekly") {
            as.integer(target_start - last$start) %/% 7L
          } else {
            12L * (as.integer(format(target_start, "%Y")) - as.integer(format(last$start, "%Y"))) +
              as.integer(format(target_start, "%m")) - as.integer(format(last$start, "%m"))
          }
          actual <- NA_real_
          if (known_calendar && target_start >= window_start && target_end <= window_end &&
              target_end <= max(oil$date)) {
            period_rows <- oil$date >= target_start & oil$date <= target_end
            if (any(period_rows)) actual <- mean(oil$price[period_rows])
          }
          for (seasonal in c(FALSE, TRUE)) {
            prediction <- last$price
            quantiles <- aggregate_error_quantiles(model, max(1L, steps), seasonal)
            if (seasonal) prediction <- prediction - season_at(model, last$start) +
              season_at(model, target_start)
            add(paste(kind, if (seasonal) "stl_adjusted_rw" else "average_rw", sep = "_"),
                paste0(kind, "_average"), prediction, actual, quantiles, target_start, target_end)
          }
        }
      }
    }
  }
  apply_label_maturity(do.call(rbind, rows))
}

score_forecasts <- function(predictions) {
  keys <- c("phase", "mode", "target_type", "method", "horizon")
  groups <- split(predictions, interaction(predictions[keys], drop = TRUE))
  do.call(rbind, lapply(groups, function(group) {
    realized <- group[group$scored, ]
    n <- nrow(realized)
    cbind(group[1, keys, drop = FALSE], data.frame(
      protocol_id = PROTOCOL_ID, availability_policy = AVAILABILITY_POLICY,
      n_forecasts = nrow(group), n = n,
      n_distinct_targets = length(unique(realized$target_date)),
      n_distinct_periods = length(unique(realized$period_start[!is.na(realized$period_start)])),
      MAE = if (n) mean(abs(realized$actual - realized$forecast)) else NA_real_,
      RMSE = if (n) sqrt(mean((realized$actual - realized$forecast)^2)) else NA_real_,
      coverage90 = if (n) mean(realized$actual >= realized$q05 & realized$actual <= realized$q95) else NA_real_
    ))
  }))
}

select_methods <- function(validation_scores) {
  candidates <- validation_scores[validation_scores$mode == "daily_update" & validation_scores$n > 0, ]
  do.call(rbind, lapply(split(candidates, interaction(candidates$target_type, candidates$horizon,
                                                   drop = TRUE)), function(group) {
    group[order(group$RMSE, group$method), ][1, c("target_type", "horizon", "method", "n", "RMSE")]
  }))
}
mark_selected <- function(predictions, selection) {
  selection_key <- paste(selection$target_type, selection$horizon, selection$method)
  predictions$selected_on_validation <-
    paste(predictions$target_type, predictions$horizon, predictions$method) %in% selection_key
  predictions
}

if (rescore_only) {
  # No model fitting, prediction, diagnostics, or optional packages in replay.
  bundle <- readRDS(file.path(output_dir, "frozen_fits.rds"))
  stopifnot(identical(bundle$protocol$id, PROTOCOL_ID),
            identical(bundle$protocol$validation_end, VALIDATION_END),
            identical(bundle$protocol$train_end, TRAIN_END),
            identical(bundle$protocol$fit_end, FIT_END),
            identical(bundle$protocol$validation_start, VALIDATION_START),
            identical(bundle$protocol$test_start, TEST_START),
            identical(bundle$protocol$test_end, TEST_END))
  validation <- read.csv(file.path(output_dir, "validation_predictions.csv"),
                         stringsAsFactors = FALSE, na.strings = c("", "NA"))
  final <- read.csv(file.path(output_dir, "final_predictions.csv"),
                    stringsAsFactors = FALSE, na.strings = c("", "NA"))
  stopifnot(all(validation$phase == "validation"), all(final$phase == "final"),
            all(validation$protocol_id == PROTOCOL_ID), all(final$protocol_id == PROTOCOL_ID))
  validation <- apply_label_maturity(validation)
  final <- apply_label_maturity(final)
} else {
  for (package in c("car", "sandwich", "lmtest")) {
    if (!requireNamespace(package, quietly = TRUE)) {
      stop("Missing R package: ", package, ". Install car, sandwich and lmtest before running.")
    }
  }
  initial_fit <- fit_models(TRAIN_END)
  export_diagnostics(initial_fit, "validation")
  validation <- forecast_phase(initial_fit, "validation", VALIDATION_START, VALIDATION_END)
  final_fit <- fit_models(FIT_END)
  export_diagnostics(final_fit, "final")
  final <- forecast_phase(final_fit, "final", TEST_START, TEST_END)
  bundle <- list(protocol = list(id = PROTOCOL_ID, availability_policy = AVAILABILITY_POLICY,
                                train_end = TRAIN_END, validation_start = VALIDATION_START,
                                validation_end = VALIDATION_END, fit_end = FIT_END,
                                test_start = TEST_START, test_end = TEST_END,
                                horizons = HORIZONS, modes = MODES),
                 initial_fit = initial_fit, final_fit = final_fit)
}
validation_scores <- score_forecasts(validation)
selection <- select_methods(validation_scores)
final <- mark_selected(final, selection)
bundle$selection <- selection
bundle$protocol$selection_policy <- paste(
  "Validation labels published by", VALIDATION_END,
  "; absolute changes require both adjacent quotes; aggregates require all period quotes;",
  "final-window actuals are retrospective.")
export(validation, "validation_predictions.csv")
export(validation_scores, "validation_metrics.csv")
export(selection, "validation_selected_methods.csv")
export(final, "final_predictions.csv")
final_scores <- score_forecasts(final)
export(final_scores, "final_metrics.csv")
saveRDS(bundle, file.path(output_dir, "frozen_fits.rds"))
print(final_scores, row.names = FALSE)
cat(if (rescore_only) "Frozen forecasts rescored in" else "Frozen-fit seasonality outputs written to",
    normalizePath(output_dir), "\n")

# Diagnostic only: fixed VECM specification, historical quarterly refits.
# No October forecasts, tuning, model selection or production-file writes.
root <- normalizePath(getwd(), winslash = "/")
out <- file.path(root, "m4_reports/diagnostics/vecm_fusion")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
source(file.path(root, "m1_time_series/for_test_set.R"))
source(file.path(root, "m1_time_series/validation_selection.R"))
data <- m1_read_data(root, as.Date("2026-09-30"))
stopifnot(max(data$date) <= as.Date("2026-09-30"))
spec <- m1_candidate_grid()[["WTI_Brent_Gasoline_K3_R2"]]
cutoffs <- as.Date(c("2025-09-01", "2025-12-01", "2026-03-01", "2026-06-01", "2026-09-01"))
old_history <- data[seq_len(m1_observable_prefix(data, M1_PROTOCOL$train_end)), , drop = FALSE]
old_fit <- m1_fit_candidate(spec, old_history)
rows <- list(); audit <- list()
for (block in seq_along(cutoffs)) {
  cutoff <- cutoffs[block]
  stop_date <- if (block < length(cutoffs)) cutoffs[block + 1L] else as.Date("2026-10-01")
  history <- data[seq_len(m1_observable_prefix(data, cutoff)), , drop = FALSE]
  stopifnot(max(history$available_date) <= cutoff)
  fit <- m1_fit_candidate(spec, history)
  audit[[block]] <- data.frame(cutoff = cutoff, n_train = nrow(history),
    latest_quote = max(history$date), latest_release = max(history$available_date),
    specification = spec$name)
  origins <- which(data$date >= cutoff & data$date < stop_date)
  for (origin in origins) {
    visible <- m1_observable_prefix(data, data$date[origin])
    bridge <- origin - visible
    state <- data[seq_len(visible), spec$xvars, drop = FALSE]
    fresh <- m1_vecm_mean(fit, state, bridge + 5L)[, 1L]
    original <- m1_vecm_mean(old_fit, state, bridge + 5L)[, 1L]
    for (h in c(1L, 5L)) {
      target <- origin + h
      if (target > nrow(data)) next
      release <- max(data$available_date[origin:target])
      if (release > as.Date("2026-09-30")) next
      rows[[length(rows) + 1L]] <- data.frame(origin_date = data$date[origin],
        target_date = data$date[target], horizon = h, actual = data$target_wti[target],
        label_available_date = release, fit_cutoff = cutoff,
        fit_latest_release = max(history$available_date),
        state_latest_release = max(data$available_date[seq_len(visible)]),
        forecast_refit = fresh[bridge + h], forecast_original = original[bridge + h])
    }
  }
  message("Historical diagnostic fit complete: ", cutoff)
}
write.csv(do.call(rbind, rows), file.path(out, "historical_vecm_refits.csv"), row.names = FALSE)
write.csv(do.call(rbind, audit), file.path(out, "historical_vecm_refit_audit.csv"), row.names = FALSE)

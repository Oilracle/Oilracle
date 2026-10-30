# Standalone frozen-parameter evaluation; see doc/m1_time_series/README.md.
# Protocol dates mirror m3_ml/common/protocol.py; no Python runtime is needed.
M1_PROTOCOL <- list(
  id = "qf5210-frozen-october-2026",
  train_end = as.Date("2023-08-31"),
  validation_start = as.Date("2023-09-01"),
  validation_end = as.Date("2026-09-30"),
  fit_end = as.Date("2026-09-30"),
  test_start = as.Date("2026-10-01"),
  test_end = as.Date("2026-10-23"),
  horizons = c(1L, 5L, 20L, 60L),
  modes = c("fixed_origin", "daily_update")
)

m1_project_root <- function(script = NULL) {
  if (!is.null(script)) return(dirname(dirname(normalizePath(script, winslash = "/"))))
  candidates <- c(getwd(), dirname(getwd()))
  for (candidate in candidates) {
    if (file.exists(file.path(candidate, "m1_time_series", "for_test_set.R"))) {
      return(normalizePath(candidate, winslash = "/"))
    }
  }
  stop("Run from the project root/module directory, or pass --root=PATH")
}

m1_read_data <- function(root, as_of) {
  read_table <- function(path) {
    x <- read.csv(path, stringsAsFactors = FALSE)
    x$date <- as.Date(x$date)
    x <- x[order(x$date), , drop = FALSE]
    if (anyNA(x$date) || anyDuplicated(x$date)) stop("Invalid/duplicate dates: ", path)
    x[x$date <= as_of, , drop = FALSE]
  }
  daily <- read_table(file.path(root, "data", "processed", "wti_daily.csv"))
  spot <- read_table(file.path(root, "data", "time_series", "daily_spot.csv"))
  vars <- c("target_wti", "brent_spot", "ny_gasoline_spot", "ny_heating_oil_spot")
  idx <- match(daily$date, spot$date)
  if (anyNA(idx)) stop("daily_spot.csv must cover every quoted WTI date")
  if (!isTRUE(all.equal(daily$target_wti, spot$target_wti[idx], check.attributes = FALSE))) {
    stop("WTI targets disagree between daily and spot tables")
  }
  data <- daily[, c("date", "target_wti", "sp500_close", "oil_implied_volatility", "yahoo_dollar_index_close")]
  data$brent_spot <- spot$brent_spot[idx]
  data$ny_gasoline_spot <- spot$ny_gasoline_spot[idx]
  data$ny_heating_oil_spot <- spot$ny_heating_oil_spot[idx]
  # No scaling or backward filling: carry only previously observed external values.
  for (v in c("sp500_close", "oil_implied_volatility", "yahoo_dollar_index_close", vars[-1])) {
    for (i in seq_len(nrow(data))) {
      if (!is.finite(data[[v]][i]) && i > 1L) data[[v]][i] <- data[[v]][i - 1L]
    }
  }
  if (!all(is.finite(as.matrix(data[, c("target_wti", "sp500_close", "oil_implied_volatility", "yahoo_dollar_index_close", vars[-1])])))) {
    stop("Missing required model inputs after causal carry-forward")
  }
  publication <- read_table(file.path(root, "data", "processed", "wti_publication.csv"))
  idx <- match(data$date, publication$date)
  if (anyNA(idx) || !isTRUE(all.equal(data$target_wti, publication$target_wti[idx], check.attributes = FALSE))) {
    stop("Publication evidence must match every quoted target")
  }
  data$available_date <- as.Date(publication$available_date[idx])
  data$evidence <- publication$evidence[idx]
  if (anyNA(data$available_date)) stop("Missing declared publication schedule")
  data$origin_price <- m1_observable_price(data)
  if (!M1_PROTOCOL$fit_end %in% data$date) stop("Missing 2026-09-30 fit-origin quote")
  data
}

# Stop before the first unavailable quote. Later known quotes do not erase an
# interior gap or turn irregular observations into a compressed trading grid.
m1_observable_prefix <- function(data, origin_date) {
  known <- data$date <= origin_date & data$available_date <= origin_date
  first_missing <- match(FALSE, known)
  n <- if (is.na(first_missing)) nrow(data) else first_missing - 1L
  if (n < 20L) stop("Too few contiguous observable quotes at ", origin_date)
  n
}

# As-of publication join: select the newest observable QUOTE, not the most
# recently released record (an old late release must never rewind the anchor).
m1_observable_price <- function(data) {
  event_date <- pmax(data$date, data$available_date)
  events <- order(event_date, data$date)
  next_event <- 1L
  latest_quote <- 0L
  price <- rep(NA_real_, nrow(data))
  for (i in seq_len(nrow(data))) {
    while (next_event <= length(events) && event_date[events[next_event]] <= data$date[i]) {
      latest_quote <- max(latest_quote, events[next_event])
      next_event <- next_event + 1L
    }
    if (latest_quote > 0L) price[i] <- data$target_wti[latest_quote]
  }
  price
}

m1_fit_frozen <- function(data, selection) {
  history <- data[seq_len(m1_observable_prefix(data, M1_PROTOCOL$fit_end)),
                  setdiff(names(data), "origin_price"), drop = FALSE]
  chosen <- selection$chosen
  rw <- list(name = "RW", family = "RW", order = c(0, 1, 0), xvars = character(), drift = FALSE)
  arima_specs <- c(list(RW = rw), setNames(list(chosen$ARIMA), chosen$ARIMA$name),
                    setNames(list(chosen$ARIMAX), chosen$ARIMAX$name))
  arima <- lapply(arima_specs, m1_fit_candidate, history = history)
  garch <- m1_fit_candidate(chosen$GARCH, history)
  vecm <- m1_fit_candidate(chosen$VECM, history)
  specifications <- vapply(c(arima_specs, setNames(list(chosen$GARCH), chosen$GARCH$name),
                              setNames(list(chosen$VECM), chosen$VECM$name)),
                           m1_specification, character(1))
  list(protocol = M1_PROTOCOL, history = history, arima = arima, arima_specs = arima_specs,
       garch = garch, garch_spec = chosen$GARCH, vecm = vecm,
       vecm_spec = chosen$VECM, vecm_vars = chosen$VECM$xvars,
       selection_signature = selection$signature, specifications = specifications)
}

m1_arima_path <- function(fit, history, H, xreg = NULL, future_x = NULL) {
  # Arima(model=...) filters the new history with supplied coefficients; no optimization.
  filtered <- forecast::Arima(history, model = fit, xreg = xreg)
  # forecast::Arima recomputes residual variance even with model=; freeze that too.
  filtered$sigma2 <- fit$sigma2
  if (!identical(stats::coef(filtered), stats::coef(fit))) stop("ARIMA coefficients changed")
  args <- list(object = filtered, h = H, level = c(90, 95))
  if (!is.null(xreg)) {
    if (is.null(future_x)) future_x <- xreg[nrow(xreg), ]
    args$xreg <- matrix(rep(future_x, each = H), nrow = H,
                        dimnames = list(NULL, colnames(xreg)))
  }
  fc <- do.call(forecast::forecast, args)
  cbind(forecast = as.numeric(fc$mean), q025 = as.numeric(fc$lower[, "95%"]),
        q05 = as.numeric(fc$lower[, "90%"]), q50 = as.numeric(fc$mean),
        q95 = as.numeric(fc$upper[, "90%"]), q975 = as.numeric(fc$upper[, "95%"]))
}

m1_garch_state <- function(bundle, history) {
  fit <- bundle$garch
  cf <- fit@fit$coef
  phi <- unname(cf[grepl("^ar[0-9]+$", names(cf))])
  theta <- unname(cf[grepl("^ma[0-9]+$", names(cf))])
  p <- length(phi)
  q <- length(theta)
  n_fit <- nrow(bundle$history)
  changes <- diff(history)
  eps <- as.numeric(utils::tail(fit@residuals, 1L))
  variance <- as.numeric(utils::tail(fit@sigma.t, 1L))^2
  r_state <- if (p) utils::tail(changes[seq_len(n_fit - 1L)], p) else numeric()
  e_state <- if (q) utils::tail(fit@residuals, q) else numeric()
  if (length(history) > n_fit) {
    for (i in seq.int(n_fit, length(changes))) {
      variance <- unname(cf["omega"] + cf["alpha1"] * eps^2 + cf["beta1"] * variance)
      mu <- sum(phi * rev(r_state)) + sum(theta * rev(e_state))
      eps <- changes[i] - mu
      if (p) r_state <- utils::tail(c(r_state, changes[i]), p)
      if (q) e_state <- utils::tail(c(e_state, eps), q)
    }
  }
  if (!is.finite(eps) || !is.finite(variance) || variance <= 0) stop("Invalid filtered GARCH state")
  list(eps = eps, variance = variance, price = utils::tail(history, 1L),
       r = r_state, e = e_state, phi = phi, theta = theta)
}

m1_garch_mean <- function(state, H) {
  r <- state$r
  e <- state$e
  p <- length(state$phi)
  q <- length(state$theta)
  changes <- numeric(H)
  for (h in seq_len(H)) {
    changes[h] <- sum(state$phi * rev(r)) + sum(state$theta * rev(e))
    if (p) r <- utils::tail(c(r, changes[h]), p)
    if (q) e <- utils::tail(c(e, 0), q)
  }
  state$price + cumsum(changes)
}

m1_garch_path <- function(bundle, history, H, simulations, seed) {
  state <- m1_garch_state(bundle, history)
  cf <- bundle$garch@fit$coef
  p <- length(state$phi)
  q <- length(state$theta)
  omega <- unname(cf["omega"])
  alpha <- unname(cf["alpha1"])
  beta <- unname(cf["beta1"])
  nu <- unname(cf["shape"])
  set.seed(seed)
  eps <- rep(state$eps, simulations)
  variance <- rep(state$variance, simulations)
  price <- rep(state$price, simulations)
  r <- matrix(rep(state$r, each = simulations), nrow = simulations, ncol = p)
  e <- matrix(rep(state$e, each = simulations), nrow = simulations, ncol = q)
  quantiles <- matrix(NA_real_, H, 5L)
  for (h in seq_len(H)) {
    variance <- omega + alpha * eps^2 + beta * variance
    innovation <- sqrt(variance) * stats::rt(simulations, df = nu) * sqrt((nu - 2) / nu)
    mu <- rep(0, simulations)
    if (p) mu <- mu + as.numeric(r[, p:1L, drop = FALSE] %*% state$phi)
    if (q) mu <- mu + as.numeric(e[, q:1L, drop = FALSE] %*% state$theta)
    change <- mu + innovation
    price <- price + change
    if (p) r <- cbind(r[, -1L, drop = FALSE], change)
    if (q) e <- cbind(e[, -1L, drop = FALSE], innovation)
    eps <- innovation
    quantiles[h, ] <- stats::quantile(price, c(0.025, 0.05, 0.5, 0.95, 0.975), names = FALSE)
  }
  cbind(forecast = m1_garch_mean(state, H),
        q025 = quantiles[, 1], q05 = quantiles[, 2], q50 = quantiles[, 3],
        q95 = quantiles[, 4], q975 = quantiles[, 5])
}

m1_vecm_path <- function(bundle, history, H) {
  fit <- bundle$vecm
  state <- as.matrix(history[, bundle$vecm_vars, drop = FALSE])
  k <- ncol(state)
  p <- length(fit$A)
  path <- m1_vecm_mean(fit, state, H)
  # Frozen residual covariance and VAR impulse responses give marginal level variances.
  residuals <- as.matrix(fit$resid)
  Sigma <- crossprod(residuals) / nrow(residuals)
  psi <- vector("list", H)
  psi[[1L]] <- diag(k)
  covariance <- matrix(0, k, k)
  variance <- numeric(H)
  for (h in seq_len(H)) {
    if (h > 1L) {
      psi[[h]] <- matrix(0, k, k)
      for (lag in seq_len(min(p, h - 1L))) psi[[h]] <- psi[[h]] + fit$A[[lag]] %*% psi[[h - lag]]
    }
    covariance <- covariance + psi[[h]] %*% Sigma %*% t(psi[[h]])
    variance[h] <- covariance[1L, 1L]
  }
  sd <- sqrt(pmax(variance, 0))
  mean <- path[, 1L]
  cbind(forecast = mean, q025 = mean + stats::qnorm(0.025) * sd,
        q05 = mean + stats::qnorm(0.05) * sd, q50 = mean,
        q95 = mean + stats::qnorm(0.95) * sd, q975 = mean + stats::qnorm(0.975) * sd)
}

m1_score <- function(predictions, horizons = M1_PROTOCOL$horizons) {
  groups <- expand.grid(mode = M1_PROTOCOL$modes, method = unique(predictions$method),
                        horizon = horizons, stringsAsFactors = FALSE)
  rows <- lapply(seq_len(nrow(groups)), function(i) {
    group <- groups[i, ]
    x <- predictions[predictions$mode == group$mode & predictions$method == group$method &
                       predictions$horizon == group$horizon & predictions$evaluated, , drop = FALSE]
    n <- nrow(x)
    err <- x$forecast - x$actual
    n_pct <- sum(x$actual != 0)
    data.frame(group, protocol_id = M1_PROTOCOL$id, n = n, n_mape = n_pct,
               RMSE = if (n) sqrt(mean(err^2)) else NA_real_,
               MAE = if (n) mean(abs(err)) else NA_real_,
               MAPE = if (n_pct) 100 * mean(abs(err[x$actual != 0] / x$actual[x$actual != 0])) else NA_real_,
               coverage95 = if (n) mean(x$actual >= x$q025 & x$actual <= x$q975) else NA_real_,
               width95 = if (n) mean(x$q975 - x$q025) else NA_real_,
               interval_score95 = if (n) mean(x$q975 - x$q025 + 40 * pmax(x$q025 - x$actual, 0) +
                                               40 * pmax(x$actual - x$q975, 0)) else NA_real_)
  })
  do.call(rbind, rows)
}

m1_run <- function(root, output = file.path(root, "m1_time_series", "outputs", "frozen_october_2026"),
                   as_of = M1_PROTOCOL$test_end, simulations = 10000L, seed = 5210L,
                   bundle_path = file.path(output, "frozen_fit_selected_2026-09-30.rds"),
                   selection_only = FALSE) {
  packages <- c("forecast", "fGarch", "vars", "urca")
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) stop("Install required R packages: ", paste(missing, collapse = ", "))
  as_of <- as.Date(as_of)
  if (is.na(as_of) || as_of < M1_PROTOCOL$fit_end) stop("--as-of must be on/after 2026-09-30")
  if (length(simulations) != 1L || !is.finite(simulations) || simulations < 1L) stop("--simulations must be positive")
  data <- m1_read_data(root, as_of)
  sys.source(file.path(root, "m1_time_series", "validation_selection.R"), envir = environment(m1_fit_frozen))
  pretest <- data[seq_len(m1_observable_prefix(data, M1_PROTOCOL$fit_end)),
                  setdiff(names(data), "origin_price"), drop = FALSE]
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  selection <- m1_select_models(data, root, output)
  if (selection_only) {
    print(selection$scores, row.names = FALSE)
    return(invisible(selection))
  }
  if (file.exists(bundle_path)) {
    bundle <- readRDS(bundle_path)
    # The anchor is an as-of model input, not an estimated coefficient. Compare
    # only fitted-history fields, including when reusing an earlier saved fit.
    if (!identical(bundle$protocol, M1_PROTOCOL) ||
        !identical(bundle$selection_signature, selection$signature) ||
        !identical(bundle$history[, names(pretest), drop = FALSE], pretest)) {
      stop("Frozen bundle protocol/history differs; use a separate output directory for a deliberate pretest refit")
    }
  } else {
    bundle <- m1_fit_frozen(data, selection)
    dir.create(dirname(bundle_path), recursive = TRUE, showWarnings = FALSE)
    saveRDS(bundle, bundle_path)
  }
  n_fit <- nrow(bundle$history)
  fit_origin <- match(M1_PROTOCOL$fit_end, data$date)
  origins <- seq.int(fit_origin, nrow(data))
  H <- max(M1_PROTOCOL$horizons)
  records <- list()
  for (mode in M1_PROTOCOL$modes) {
    mode_origins <- if (mode == "fixed_origin") fit_origin else origins
    for (origin in mode_origins) {
      # Treat publication dates as declared evidence/assumptions, not vintage proof.
      n_visible <- m1_observable_prefix(data, data$date[origin])
      history <- data[seq_len(n_visible), , drop = FALSE]
      bridge <- origin - n_visible
      forecast_H <- bridge + H
      keep <- bridge + seq_len(H)
      paths <- lapply(names(bundle$arima), function(method) {
        spec <- bundle$arima_specs[[method]]
        X <- if (length(spec$xvars)) as.matrix(history[, spec$xvars, drop = FALSE]) else NULL
        future_x <- if (length(spec$xvars)) as.numeric(unlist(data[origin, spec$xvars, drop = FALSE], use.names = FALSE)) else NULL
        m1_arima_path(bundle$arima[[method]], history$target_wti, forecast_H, X,
                      future_x = future_x)[keep, , drop = FALSE]
      })
      names(paths) <- names(bundle$arima)
      # Canonical anchor includes declared/assumed publication rows, not stale
      # wti_daily feature values or hidden same-origin WTI.
      rw_shift <- data$origin_price[origin] - paths$RW[1L, "forecast"]
      paths$RW <- paths$RW + rw_shift
      paths[[bundle$garch_spec$name]] <- m1_garch_path(bundle, history$target_wti, forecast_H,
                                                     simulations, seed + origin)[keep, , drop = FALSE]
      paths[[bundle$vecm_spec$name]] <- m1_vecm_path(bundle, history, forecast_H)[keep, , drop = FALSE]
      target_rows <- origin + seq_len(H)
      reached <- target_rows <= nrow(data)
      dates <- as.Date(rep(NA_character_, H))
      dates[reached] <- data$date[target_rows[reached]]
      actual <- rep(NA_real_, H)
      actual[reached] <- data$target_wti[target_rows[reached]]
      evaluated <- reached & !is.na(dates) & dates >= M1_PROTOCOL$test_start & dates <= M1_PROTOCOL$test_end
      for (method in names(paths)) {
        if (!all(is.finite(paths[[method]]))) stop("Nonfinite forecast: ", method, " at ", data$date[origin])
        records[[length(records) + 1L]] <- data.frame(
          mode = mode, origin_date = data$date[origin], target_date = dates,
          horizon = seq_len(H), method = method, paths[[method]], actual = actual,
          evaluated = evaluated, origin_row = origin - 1L, target_row = target_rows - 1L,
          target_date_status = ifelse(reached, "observed_quoted_row", "unreached_quoted_row"),
          exogenous_scenario = if (method == bundle$arima_specs[[3L]]$name) "Selected exogenous variables fixed at origin" else "endogenous/no exogenous",
          fit_end = M1_PROTOCOL$fit_end, as_of = as_of, protocol_id = M1_PROTOCOL$id,
          origin_price = data$origin_price[origin],
          origin_price_policy = "Newest quote with declared available_date <= origin; includes assumed_weekly_release",
          state_last_quote = history$date[n_visible], state_last_price = history$target_wti[n_visible],
          n_observable_prefix = n_visible, bridge_rows = bridge,
          available_date_max = max(history$available_date),
          n_assumed_publications = sum(history$evidence == "assumed_weekly_release"),
          availability_policy = "WTI dated archive else assumed weekly release; other spots share WTI schedule assumption",
          stringsAsFactors = FALSE)
      }
    }
  }
  predictions <- do.call(rbind, records)
  scores <- m1_score(predictions)
  parameters <- do.call(rbind, lapply(names(bundle$arima), function(method) {
    fit <- bundle$arima[[method]]
    values <- c(stats::coef(fit), sigma2 = fit$sigma2)
    data.frame(method = method, parameter = names(values), value = unname(values))
  }))
  garch_cf <- bundle$garch@fit$coef
  parameters <- rbind(parameters, data.frame(method = bundle$garch_spec$name, parameter = names(garch_cf), value = unname(garch_cf)))
  for (lag in seq_along(bundle$vecm$A)) {
    A <- bundle$vecm$A[[lag]]
    names_A <- outer(bundle$vecm_vars, bundle$vecm_vars, function(y, x) paste0(y, "<-", x, ":lag", lag))
    parameters <- rbind(parameters, data.frame(method = bundle$vecm_spec$name, parameter = as.vector(names_A), value = as.vector(A)))
  }
  parameters <- rbind(parameters, data.frame(method = bundle$vecm_spec$name,
                                            parameter = paste0(bundle$vecm_vars, ":intercept"),
                                            value = as.numeric(bundle$vecm$deterministic[, 1L])))
  manifest <- data.frame(protocol_id = M1_PROTOCOL$id, train_end = M1_PROTOCOL$train_end, validation_start = M1_PROTOCOL$validation_start,
                         validation_end = M1_PROTOCOL$validation_end, fit_end = M1_PROTOCOL$fit_end,
                         test_start = M1_PROTOCOL$test_start, test_end = M1_PROTOCOL$test_end,
                         as_of = as_of, n_initial_train = m1_observable_prefix(data, M1_PROTOCOL$train_end),
                         n_validation = sum(data$date >= M1_PROTOCOL$validation_start & data$date <= M1_PROTOCOL$validation_end),
                         n_final_fit = n_fit, fit_last_quote = utils::tail(bundle$history$date, 1L),
                         n_fit_assumed_publications = sum(bundle$history$evidence == "assumed_weekly_release"),
                         availability_policy = "Dated WPSR match; unmatched assumed week Friday+5 USFederal days; other spots share WTI schedule assumption",
                         vintage_limit = "Retrospective current-vintage analysis; dates do not prove historical data vintages",
                         n_test_observed = sum(data$date >= M1_PROTOCOL$test_start & data$date <= M1_PROTOCOL$test_end),
                         simulations = simulations, seed = seed,
                         garch_algorithm = bundle$garch@fit$params$control$algorithm[1L],
                         garch_convergence = bundle$garch@fit$convergence,
                         garch_negative_loglikelihood = unname(bundle$garch@fit$llh),
                         garch_input_scale = bundle$garch@fit$series$scale,
                         garch_tol1 = bundle$garch@fit$params$control$tol1,
                         garch_tol2 = bundle$garch@fit$params$control$tol2,
                         specification_source = "Reproducible mature-label initial-train historical validation selection",
                         selection_criterion = M1_SELECTION_CRITERION,
                         selection_input_md5 = selection$provenance$input_md5,
                         selection_signature = selection$signature,
                         preprocessing = "No feature scaling; GARCH pretest internal SD scaling mapped back to USD; causal external carry-forward",
                         bundle = normalizePath(bundle_path, winslash = "/"))
  stamp <- paste0("2026-10-01_2026-10-23_asof_", as_of)
  write.csv(predictions, file.path(output, paste0("forecasts_", stamp, ".csv")), row.names = FALSE, na = "")
  write.csv(scores, file.path(output, paste0("metrics_", stamp, ".csv")), row.names = FALSE, na = "")
  write.csv(m1_score(predictions, seq_len(H)), file.path(output, paste0("metrics_all_horizons_", stamp, ".csv")), row.names = FALSE, na = "")
  write.csv(parameters, file.path(output, "frozen_parameters_2026-09-30.csv"), row.names = FALSE)
  write.csv(manifest, file.path(output, paste0("manifest_", stamp, ".csv")), row.names = FALSE)
  print(scores, row.names = FALSE)
  invisible(list(predictions = predictions, metrics = scores, bundle = bundle, manifest = manifest, selection = selection))
}

m1_cli <- function(args = commandArgs(trailingOnly = TRUE)) {
  options <- list()
  for (arg in args) {
    if (!grepl("^--[^=]+=.+$", arg)) stop("Options use --name=value: root, output, as-of, simulations, seed, bundle, selection-only")
    name <- sub("^--([^=]+)=.*$", "\\1", arg)
    if (!name %in% c("root", "output", "as-of", "simulations", "seed", "bundle", "selection-only")) stop("Unknown option: ", name)
    options[[name]] <- sub("^--[^=]+=", "", arg)
  }
  if (!is.null(options[["selection-only"]]) && !options[["selection-only"]] %in% c("true", "false")) {
    stop("--selection-only must be true or false")
  }
  script_arg <- grep("^--file=", commandArgs(), value = TRUE)
  script <- if (length(script_arg)) sub("^--file=", "", script_arg[1L]) else NULL
  root <- if (!is.null(options$root)) normalizePath(options$root, winslash = "/", mustWork = TRUE) else m1_project_root(script)
  output <- if (!is.null(options$output)) options$output else file.path(root, "m1_time_series", "outputs", "frozen_october_2026")
  m1_run(root = root, output = output,
         as_of = if (!is.null(options[["as-of"]])) options[["as-of"]] else M1_PROTOCOL$test_end,
         simulations = if (!is.null(options$simulations)) as.integer(options$simulations) else 10000L,
         seed = if (!is.null(options$seed)) as.integer(options$seed) else 5210L,
         bundle_path = if (!is.null(options$bundle)) options$bundle else file.path(output, "frozen_fit_selected_2026-09-30.rds"),
         selection_only = identical(options[["selection-only"]], "true"))
}

if (sys.nframe() == 0L) m1_cli()

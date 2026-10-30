# Non-plotting historical selection for the standalone final evaluator.
# This file is loaded by for_test_set.R; it never reads October selection targets.
M1_SELECTION_CRITERION <- "Mean h=1,5,20,60 validation RMSE/RW_RMSE; complete common samples; then complexity/name"

m1_candidate_grid <- function() {
  candidates <- list()
  add <- function(name, family, order = NULL, xvars = character(), drift = FALSE,
                  K = NA_integer_, rank = NA_integer_) {
    complexity <- if (family == "VECM") length(xvars)^2 * K + length(xvars) * rank else sum(order[c(1L, 3L)]) + length(xvars) + as.integer(drift)
    candidates[[name]] <<- list(name = name, family = family, order = order,
                                xvars = xvars, drift = drift, K = K, rank = rank,
                                complexity = complexity)
  }
  orders <- list(ARIMA011 = c(0, 1, 1), ARIMA111 = c(1, 1, 1),
                 ARIMA012 = c(0, 1, 2), ARIMA211 = c(2, 1, 1))
  add("RW_drift", "ARIMA", c(0, 1, 0), drift = TRUE)
  for (name in names(orders)) add(name, "ARIMA", orders[[name]])
  x_sets <- list(SP500 = "sp500_close",
                 SP500_OVX = c("sp500_close", "oil_implied_volatility"),
                 SP500_Dollar = c("sp500_close", "yahoo_dollar_index_close"),
                 SP500_OVX_Dollar = c("sp500_close", "oil_implied_volatility", "yahoo_dollar_index_close"))
  x_orders <- c(orders, list(ARIMA311 = c(3, 1, 1)))
  for (set in names(x_sets)) {
    for (name in names(x_orders)) add(paste(set, name, sep = "_"), "ARIMAX", x_orders[[name]], x_sets[[set]])
  }
  for (name in names(orders)) add(paste0("GARCH_", name), "GARCH", orders[[name]])
  vecm_sets <- list(
    WTI_Brent = list(vars = c("target_wti", "brent_spot"), K = c(4L, 5L), rank = 1L),
    WTI_Brent_Gasoline = list(vars = c("target_wti", "brent_spot", "ny_gasoline_spot"), K = c(3L, 4L), rank = 2L),
    WTI_Brent_Heating = list(vars = c("target_wti", "brent_spot", "ny_heating_oil_spot"), K = c(4L, 5L), rank = 2L),
    WTI_All4 = list(vars = c("target_wti", "brent_spot", "ny_gasoline_spot", "ny_heating_oil_spot"), K = c(3L, 4L), rank = 3L)
  )
  for (set in names(vecm_sets)) {
    spec <- vecm_sets[[set]]
    for (K in spec$K) add(paste0(set, "_K", K, "_R", spec$rank), "VECM", xvars = spec$vars, K = K, rank = spec$rank)
  }
  candidates
}

m1_specification <- function(spec) {
  if (spec$family == "VECM") {
    return(paste0("VECM ", paste(spec$xvars, collapse = "/"), "; K=", spec$K,
                  "; rank=", spec$rank, "; restricted constant"))
  }
  paste0(spec$family, "(", paste(spec$order, collapse = ","), ")",
         if (spec$drift) "; drift" else "; no drift",
         if (length(spec$xvars)) paste0("; x=", paste(spec$xvars, collapse = "/"), " fixed at origin") else "",
         if (spec$family == "GARCH") "; GARCH(1,1), standardized Student-t differences" else "")
}

m1_fit_candidate <- function(spec, history) {
  if (spec$family %in% c("ARIMA", "ARIMAX", "RW")) {
    X <- if (length(spec$xvars)) as.matrix(history[, spec$xvars, drop = FALSE]) else NULL
    return(forecast::Arima(history$target_wti, order = spec$order,
                            xreg = X, include.drift = spec$drift))
  }
  if (spec$family == "VECM") {
    johansen <- urca::ca.jo(as.matrix(history[, spec$xvars, drop = FALSE]),
                           type = "trace", ecdet = "const", K = spec$K, spec = "transitory")
    return(vars::vec2var(johansen, r = spec$rank))
  }
  # Supported same-likelihood hybrid solver; no acceptance of failed fits.
  formula <- stats::as.formula(sprintf("~ arma(%d,%d) + garch(1,1)", spec$order[1L], spec$order[3L]))
  fit <- fGarch::garchFit(formula, data = diff(history$target_wti), include.mean = FALSE,
                         cond.dist = "std", trace = FALSE, algorithm = "nlminb+nm",
                         control = list(xscale = TRUE, tol1 = 1e4, tol2 = 1))
  code <- fit@fit$convergence
  message <- if (is.null(fit@fit$message)) "" else paste(fit@fit$message, collapse = "; ")
  if (length(code) != 1L || is.na(code) || code != 0L ||
      length(fit@fit$llh) != 1L || !is.finite(fit@fit$llh)) {
    stop(spec$name, " likelihood optimization failed (code ", paste(code, collapse = ","), "): ", message)
  }
  cf <- fit@fit$coef
  if (!all(c("omega", "alpha1", "beta1", "shape") %in% names(cf)) || any(!is.finite(cf)) ||
      cf["omega"] <= 0 || cf["alpha1"] < 0 || cf["beta1"] < 0 ||
      cf["alpha1"] + cf["beta1"] >= 1 || cf["shape"] <= 2) {
    stop(spec$name, " has invalid/nonstationary Student-t GARCH parameters")
  }
  fit
}

m1_arima_mean <- function(fit, history, H, xreg = NULL, future_x = NULL) {
  filtered <- forecast::Arima(history, model = fit, xreg = xreg)
  filtered$sigma2 <- fit$sigma2
  if (!identical(stats::coef(filtered), stats::coef(fit))) stop("ARIMA coefficients changed while filtering")
  args <- list(object = filtered, h = H)
  if (!is.null(xreg)) {
    args$xreg <- matrix(rep(future_x, each = H), nrow = H,
                        dimnames = list(NULL, colnames(xreg)))
  }
  as.numeric(do.call(forecast::forecast, args)$mean)
}

m1_vecm_mean <- function(fit, state, H) {
  state <- as.matrix(state)
  k <- ncol(state)
  p <- length(fit$A)
  intercept <- as.numeric(fit$deterministic[, 1L])
  path <- matrix(NA_real_, H, k)
  for (h in seq_len(H)) {
    next_y <- intercept
    for (lag in seq_len(p)) next_y <- next_y + as.numeric(fit$A[[lag]] %*% state[nrow(state) - lag + 1L, ])
    path[h, ] <- next_y
    state <- rbind(state, next_y)
  }
  path
}

m1_input_digest <- function(object) {
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path), add = TRUE)
  saveRDS(object, path, compress = FALSE, version = 2)
  unname(tools::md5sum(path))
}

m1_select_models <- function(data, root, output) {
  data <- data[data$date <= M1_PROTOCOL$fit_end, , drop = FALSE]
  candidates <- m1_candidate_grid()
  packages <- c("forecast", "fGarch", "urca", "vars")
  provenance <- list(protocol_id = M1_PROTOCOL$id, criterion = M1_SELECTION_CRITERION,
                      training = "One initial fit through available TRAIN_END prefix; no validation parameter refits",
                      validation = "Every quoted origin from TRAIN_END; key horizons; only targets mature by FIT_END",
                      input_md5 = m1_input_digest(data), candidates = candidates,
                      source_md5 = tools::md5sum(file.path(root, "m1_time_series", c("for_test_set.R", "validation_selection.R"))),
                      package_versions = setNames(vapply(packages, function(x) as.character(utils::packageVersion(x)), character(1)), packages))
  signature <- m1_input_digest(provenance)
  cache <- file.path(output, "validation_selection_2023-09-01_2026-09-30.rds")
  if (file.exists(cache)) {
    selection <- readRDS(cache)
    if (!identical(selection$signature, signature)) {
      stop("Selection input/source/package digest changed; use a clean output directory for a deliberate pretest reselection")
    }
    return(selection)
  }
  n_train <- m1_observable_prefix(data, M1_PROTOCOL$train_end)
  history <- data[seq_len(n_train), , drop = FALSE]
  origins <- which(data$date >= M1_PROTOCOL$train_end & data$date <= M1_PROTOCOL$validation_end)
  plan <- list()
  for (origin in origins) {
    target <- origin + M1_PROTOCOL$horizons
    eligible <- target <= nrow(data)
    eligible[eligible] <- data$date[target[eligible]] >= M1_PROTOCOL$validation_start &
      data$date[target[eligible]] <= M1_PROTOCOL$validation_end & data$available_date[target[eligible]] <= M1_PROTOCOL$fit_end
    if (!any(eligible)) next
    visible <- m1_observable_prefix(data, data$date[origin])
    plan[[length(plan) + 1L]] <- data.frame(origin = origin, visible = visible,
                                           horizon = M1_PROTOCOL$horizons[eligible], target = target[eligible],
                                           actual = data$target_wti[target[eligible]], RW = data$origin_price[origin])
  }
  plan <- do.call(rbind, plan)
  if (!nrow(plan) || anyNA(plan$RW)) stop("No complete causal validation target plan")
  by_origin <- split(seq_len(nrow(plan)), plan$origin)
  baseline <- vapply(M1_PROTOCOL$horizons, function(h) {
    rows <- plan$horizon == h
    if (!any(rows)) stop("No mature validation observations at horizon ", h)
    sqrt(mean((plan$actual[rows] - plan$RW[rows])^2))
  }, numeric(1))
  by_horizon <- list()
  rankings <- list()
  for (name in names(candidates)) {
    spec <- candidates[[name]]
    message("Historical selection: ", name)
    prediction <- rep(NA_real_, nrow(plan))
    error <- NULL
    tryCatch({
      fit <- m1_fit_candidate(spec, history)
      garch_bundle <- if (spec$family == "GARCH") list(garch = fit, history = history) else NULL
      for (rows in by_origin) {
        origin <- plan$origin[rows[1L]]
        visible <- plan$visible[rows[1L]]
        bridge <- origin - visible
        H <- bridge + max(plan$horizon[rows])
        observed <- data[seq_len(visible), , drop = FALSE]
        if (spec$family %in% c("ARIMA", "ARIMAX")) {
          X <- if (length(spec$xvars)) as.matrix(observed[, spec$xvars, drop = FALSE]) else NULL
          future_x <- if (length(spec$xvars)) as.numeric(unlist(data[origin, spec$xvars, drop = FALSE], use.names = FALSE)) else NULL
          path <- m1_arima_mean(fit, observed$target_wti, H, X, future_x)
        } else if (spec$family == "GARCH") {
          state <- m1_garch_state(garch_bundle, observed$target_wti)
          path <- m1_garch_mean(state, H)
        } else {
          path <- m1_vecm_mean(fit, observed[, spec$xvars, drop = FALSE], H)[, 1L]
        }
        prediction[rows] <- path[bridge + plan$horizon[rows]]
      }
      if (any(!is.finite(prediction))) stop("Nonfinite/incomplete candidate validation path")
    }, error = function(e) { error <<- conditionMessage(e) })
    status <- if (is.null(error)) "ok" else "failed"
    metrics <- lapply(seq_along(M1_PROTOCOL$horizons), function(i) {
      h <- M1_PROTOCOL$horizons[i]
      rows <- plan$horizon == h
      ok <- rows & is.finite(prediction)
      n <- sum(ok)
      rmse <- if (n) sqrt(mean((prediction[ok] - plan$actual[ok])^2)) else NA_real_
      relative <- if (status != "ok" || n != sum(rows)) Inf else if (baseline[i] > 0) rmse / baseline[i] else if (rmse == 0) 1 else Inf
      data.frame(protocol_id = M1_PROTOCOL$id, family = spec$family, method = name, horizon = h,
                 n = n, n_expected = sum(rows), RMSE = rmse, RW_RMSE = baseline[i], relative_RMSE = relative,
                 status = status, error = if (is.null(error)) "" else error, stringsAsFactors = FALSE)
    })
    metrics <- do.call(rbind, metrics)
    by_horizon[[name]] <- metrics
    rankings[[name]] <- data.frame(protocol_id = M1_PROTOCOL$id, family = spec$family, method = name,
                                   score = mean(metrics$relative_RMSE), complexity = spec$complexity,
                                   status = status, error = if (is.null(error)) "" else error,
                                   input_md5 = provenance$input_md5, selection_signature = signature,
                                   criterion = M1_SELECTION_CRITERION, stringsAsFactors = FALSE)
  }
  scores <- do.call(rbind, rankings)
  scores <- scores[order(scores$family, scores$score, scores$complexity, scores$method), , drop = FALSE]
  chosen <- list()
  for (family in c("ARIMA", "ARIMAX", "GARCH", "VECM")) {
    eligible <- scores[scores$family == family & scores$status == "ok" & is.finite(scores$score), , drop = FALSE]
    if (!nrow(eligible)) {
      write.csv(scores, file.path(output, "validation_selection_scores.csv"), row.names = FALSE)
      write.csv(do.call(rbind, by_horizon), file.path(output, "validation_selection_by_horizon.csv"), row.names = FALSE)
      stop("No valid historical selection candidate for ", family, "; inspect selection scores")
    }
    chosen[[family]] <- candidates[[eligible$method[1L]]]
  }
  scores$selected <- scores$method %in% vapply(chosen, function(x) x$name, character(1))
  selection <- list(signature = signature, provenance = provenance, chosen = chosen,
                     scores = scores, by_horizon = do.call(rbind, by_horizon),
                     n_initial_fit = n_train, initial_fit_last_quote = utils::tail(history$date, 1L))
  selected_config <- do.call(rbind, lapply(chosen, function(spec) {
    data.frame(protocol_id = M1_PROTOCOL$id, family = spec$family, method = spec$name,
               order = paste(spec$order, collapse = ","), xvars = paste(spec$xvars, collapse = ";"),
               include_drift = spec$drift, K = spec$K, rank = spec$rank,
               specification = m1_specification(spec), input_md5 = provenance$input_md5,
               selection_signature = signature, criterion = M1_SELECTION_CRITERION)
  }))
  write.csv(scores, file.path(output, "validation_selection_scores.csv"), row.names = FALSE)
  write.csv(selection$by_horizon, file.path(output, "validation_selection_by_horizon.csv"), row.names = FALSE)
  write.csv(selected_config, file.path(output, "selected_config.csv"), row.names = FALSE)
  saveRDS(selection, cache)
  selection
}

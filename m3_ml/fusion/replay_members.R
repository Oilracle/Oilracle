# Reuse actual member specifications/functions and frozen M2 initial objects.
root <- normalizePath(getwd(), winslash="/")
out <- file.path(root,"m3_ml/fusion/outputs/frozen_october_2026")
spec <- jsonlite::fromJSON(file.path(out,"specification.json"))
source(file.path(root,"m1_time_series/for_test_set.R"))
source(file.path(root,"m1_time_series/validation_selection.R"))
data <- m1_read_data(root,M1_PROTOCOL$fit_end)
chosen <- m1_candidate_grid()[[spec$m1_method]]
history <- data[seq_len(m1_observable_prefix(data,M1_PROTOCOL$train_end)),,drop=FALSE]
fit <- m1_fit_candidate(chosen,history)
bundle <- list(garch=fit,history=history)
hs <- c(1:17,20,60)
origins <- which(data$date>=M1_PROTOCOL$validation_start & data$date<=M1_PROTOCOL$fit_end)
rows <- list()
for(origin in origins) {
  visible <- m1_observable_prefix(data,data$date[origin]); bridge <- origin-visible
  observed <- data[seq_len(visible),,drop=FALSE]; H <- bridge+max(hs)
  if(chosen$family %in% c("ARIMA","ARIMAX")) {
    X <- if(length(chosen$xvars)) as.matrix(observed[,chosen$xvars,drop=FALSE]) else NULL
    fx <- if(length(chosen$xvars)) as.numeric(unlist(data[origin,chosen$xvars,drop=FALSE])) else NULL
    path <- m1_arima_mean(fit,observed$target_wti,H,X,fx)
  } else if(chosen$family=="GARCH") path <- m1_garch_mean(m1_garch_state(bundle,observed$target_wti),H)
  else path <- m1_vecm_mean(fit,observed[,chosen$xvars,drop=FALSE],H)[,1]
  rows[[length(rows)+1L]] <- data.frame(origin_date=data$date[origin],horizon=hs,forecast=path[bridge+hs],method=chosen$name)
}
m1 <- do.call(rbind,rows); stopifnot(all(is.finite(m1$forecast)))
write.csv(m1,file.path(out,"m1_validation.csv"),row.names=FALSE)

# Import function definitions, without executing the member's top-level run.
needed <- c("calendar","period_start","season_at","observable_prefix")
for(e in parse(file.path(root,"m2_seasonality/seasonality.R"))) {
  if(is.call(e) && identical(e[[1]],as.name("<-")) && as.character(e[[2]])[1] %in% needed) eval(e)
}
oil <- data.frame(date=data$date,price=data$target_wti,available_date=data$available_date)
frozen <- readRDS(file.path(root,"m2_seasonality/outputs/frozen_october_2026/frozen_fits.rds"))
fit2 <- frozen$initial_fit; method <- spec$m2_method; rows <- list()
for(origin in origins) {
  visible <- observable_prefix(oil$date[origin]); target <- origin+hs
  keep <- target<=nrow(oil); target <- target[keep]; horizons <- hs[keep]
  if(!length(target)) next
  if(method %in% c("calendar_mean_original","calendar_mean_3iqr")) {
    dates <- oil$date[seq.int(visible+1L,max(target))]
    cumulative <- cumsum(predict(fit2[[sub("calendar_","",method)]],newdata=calendar(dates)))
    pred <- oil$price[visible]+cumulative[target-visible]
  } else {
    kind <- if(method=="stl_monthly_daily_bridge") "monthly" else "weekly"
    pred <- oil$price[visible]-season_at(fit2[[kind]],oil$date[visible])+season_at(fit2[[kind]],oil$date[target])
  }
  rows[[length(rows)+1L]] <- data.frame(origin_date=oil$date[origin],horizon=horizons,forecast=pred,method=method)
}
m2 <- do.call(rbind,rows)
old <- read.csv(file.path(root,"m2_seasonality/outputs/frozen_october_2026/validation_predictions.csv"))
old <- old[old$mode=="daily_update" & old$target_type=="price" & old$method==method & old$horizon %in% c(1,5,20,60),]
matched <- merge(m2,old[,c("origin_date","horizon","forecast")],by=c("origin_date","horizon"))
stopifnot(nrow(matched)>2000,max(abs(matched$forecast.x-matched$forecast.y))<1e-8)
write.csv(m2,file.path(out,"m2_validation.csv"),row.names=FALSE)
message("Member replay complete: M1=",nrow(m1),", M2=",nrow(m2),"; M2 matches original exports")

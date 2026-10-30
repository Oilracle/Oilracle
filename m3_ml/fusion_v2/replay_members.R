# Fit actual M1/M2 specifications on the same 63-origin schedule as ML.
root <- normalizePath(getwd(),winslash="/")
out <- file.path(root,"m3_ml/fusion_v2/outputs/frozen_october_2026")
spec <- jsonlite::fromJSON(file.path(out,"specification.json"))
folds <- jsonlite::fromJSON(file.path(out,"base_folds.json"))
source(file.path(root,"m1_time_series/for_test_set.R"))
source(file.path(root,"m1_time_series/validation_selection.R"))
data <- m1_read_data(root,M1_PROTOCOL$fit_end)
chosen <- m1_candidate_grid()[[spec$m1_method]]
stopifnot(chosen$family=="VECM",spec$m2_method=="stl_monthly_daily_bridge")
needed <- c("period_start","period_end","aggregate_prices","fit_stl","season_at")
for(e in parse(file.path(root,"m2_seasonality/seasonality.R"))) {
  if(is.call(e) && identical(e[[1]],as.name("<-")) && as.character(e[[2]])[1] %in% needed) eval(e)
}
hs <- c(1:17,20,60); rows1 <- list(); rows2 <- list(); audit <- list()
for(j in seq_len(nrow(folds))) {
  cutoff <- as.Date(folds$cutoff[j]); visible <- m1_observable_prefix(data,cutoff)
  history <- data[seq_len(visible),,drop=FALSE]
  fit1 <- m1_fit_candidate(chosen,history)
  hist2 <- data.frame(date=history$date,price=history$target_wti)
  fit2 <- fit_stl(hist2,tail(hist2$date,1),"monthly")
  audit[[j]] <- data.frame(cutoff=cutoff,latest_label_release=max(history$available_date),
                            effective_quote=tail(history$date,1),n_train=visible)
  stopifnot(max(history$available_date)<=cutoff)
  for(origin in seq.int(folds$start_row[j]+1L,folds$end_row[j]+1L)) {
    known <- m1_observable_prefix(data,data$date[origin]); bridge <- origin-known
    observed <- data[seq_len(known),,drop=FALSE]
    path <- m1_vecm_mean(fit1,observed[,chosen$xvars,drop=FALSE],bridge+max(hs))[,1]
    rows1[[length(rows1)+1L]] <- data.frame(origin_date=data$date[origin],horizon=hs,
                       forecast=path[bridge+hs],method=chosen$name,base_fit_cutoff=cutoff)
    valid <- origin+hs<=nrow(data); horizons <- hs[valid]; target <- origin+horizons
    if(length(target)) {
      pred <- data$target_wti[known]-season_at(fit2,data$date[known])+season_at(fit2,data$date[target])
      rows2[[length(rows2)+1L]] <- data.frame(origin_date=data$date[origin],horizon=horizons,
                         forecast=pred,method=spec$m2_method,base_fit_cutoff=cutoff)
    }
  }
  message("M1/M2 refit ",j,"/",nrow(folds)," cutoff=",cutoff)
}
for(name in c("m1","m2")) {
  result <- do.call(rbind,if(name=="m1") rows1 else rows2)
  stopifnot(all(is.finite(result$forecast)),all(result$forecast>0))
  write.csv(result,file.path(out,paste0(name,"_validation.csv")),row.names=FALSE)
}
write.csv(do.call(rbind,audit),file.path(out,"r_refit_audit.csv"),row.names=FALSE)

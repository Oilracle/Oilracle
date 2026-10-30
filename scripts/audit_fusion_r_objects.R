# Inspect native frozen member objects; do not rewrite any member artifact.
root <- normalizePath(getwd(),winslash="/")
out <- file.path(root,"m3_ml/fusion/outputs/frozen_october_2026/diagnostics")
source(file.path(root,"m1_time_series/for_test_set.R"))
source(file.path(root,"m1_time_series/validation_selection.R"))
spec <- jsonlite::fromJSON(file.path(root,"m3_ml/fusion/outputs/frozen_october_2026/specification.json"))
files <- file.path(root,c("m1_time_series/outputs/frozen_october_2026/frozen_fit_selected_2026-09-30.rds",
                         "m2_seasonality/outputs/frozen_october_2026/frozen_fits.rds"))
before <- tools::md5sum(files)
data <- m1_read_data(root,as.Date("2026-10-23"))
bundle <- readRDS(files[1])
stopifnot(bundle$vecm_name==spec$m1_method || is.null(bundle$vecm_name))
origin <- which(data$date==M1_PROTOCOL$fit_end)
visible <- m1_observable_prefix(data,M1_PROTOCOL$fit_end)
bridge <- origin-visible
predict_path <- function(d) m1_vecm_mean(bundle$vecm,d[seq_len(visible),bundle$vecm_vars,drop=FALSE],bridge+60)[bridge+1:60,1]
prediction <- predict_path(data)
native <- read.csv(file.path(root,"m1_time_series/outputs/frozen_october_2026/forecasts_2026-10-01_2026-10-23_asof_2026-10-23.csv"))
native <- native[native$mode=="fixed_origin" & native$method==spec$m1_method,]
stopifnot(max(abs(prediction-native$forecast[match(1:60,native$horizon)]))<1e-8)
mutated <- data
hidden <- data$available_date>M1_PROTOCOL$fit_end
mutated[hidden,bundle$vecm_vars] <- mutated[hidden,bundle$vecm_vars]+1000
stopifnot(identical(prediction,predict_path(mutated)))
needed <- c("period_start","season_at","observable_prefix")
for(e in parse(file.path(root,"m2_seasonality/seasonality.R"))) {
  if(is.call(e)&&identical(e[[1]],as.name("<-"))&&as.character(e[[2]])[1]%in%needed) eval(e)
}
oil <- data.frame(date=data$date,price=data$target_wti,available_date=data$available_date)
fit <- readRDS(files[2])$final_fit
stopifnot(spec$m2_method=="stl_monthly_daily_bridge")
visible2 <- observable_prefix(M1_PROTOCOL$fit_end)
targets <- origin+1:17
prediction2 <- oil$price[visible2]-season_at(fit$monthly,oil$date[visible2])+season_at(fit$monthly,oil$date[targets])
native2 <- read.csv(file.path(root,"m2_seasonality/outputs/frozen_october_2026/final_predictions.csv"))
native2 <- native2[native2$mode=="fixed_origin"&native2$method==spec$m2_method&native2$target_type=="price",]
stopifnot(max(abs(prediction2-native2$forecast[match(1:17,native2$horizon)]))<1e-8)
stopifnot(identical(before,tools::md5sum(files)))
result <- list(m1_native_frozen_vecm_reproduces_60_steps=TRUE,m1_hidden_observation_mutation_invariant=TRUE,
               m2_native_final_stl_reproduces_17_steps=TRUE,frozen_objects_unchanged=TRUE,
               effective_quote_date=as.character(data$date[visible]),fixed_anchor=data$origin_price[origin])
jsonlite::write_json(result,file.path(out,"native_r_audit.json"),auto_unbox=TRUE,pretty=TRUE)
print(result)

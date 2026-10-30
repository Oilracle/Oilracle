# Historical exploratory and model-selection analysis.
# --final-evaluation runs the standalone frozen-fit protocol without plots/EDA.
.m1_script_arg <- grep("^--file=", commandArgs(), value = TRUE)
.m1_source_file <- if (length(.m1_script_arg)) sub("^--file=", "", .m1_script_arg[1L]) else NULL
.m1_candidates <- if (!is.null(.m1_source_file)) dirname(dirname(normalizePath(.m1_source_file))) else c(getwd(), dirname(getwd()))
.m1_root <- .m1_candidates[vapply(.m1_candidates, function(x) file.exists(file.path(x, "m1_time_series", "for_test_set.R")), logical(1))][1L]
if (is.na(.m1_root)) stop("Run from the project root or m1_time_series directory")
source(file.path(.m1_root, "m1_time_series", "for_test_set.R"))
if (sys.nframe() == 0L && "--final-evaluation" %in% commandArgs(trailingOnly = TRUE)) {
  m1_cli(setdiff(commandArgs(trailingOnly = TRUE), "--final-evaluation"))
  quit(save = "no")
}
.m1_publication <- read.csv(file.path(.m1_root, "data", "processed", "wti_publication.csv"))
.m1_publication$date <- as.Date(.m1_publication$date)
.m1_publication$available_date <- as.Date(.m1_publication$available_date)

library(ggplot2)
library(dplyr)
library(fBasics)
library(tseries)
library(forecast)
library(fGarch)
library(lmtest)
library(FinTS)
library(moments)
library(tidyr)
library(MTS)


df <- read.csv(file.path(.m1_root, "data", "processed", "wti_daily.csv"))
df <- df %>% mutate(date = as.Date(date)) %>% 
  arrange(date) %>%
  filter(date <= M1_PROTOCOL$validation_end) # Never load October into model selection.
df$available_date <- .m1_publication$available_date[match(df$date, .m1_publication$date)]
if (anyNA(df$available_date)) stop("Publication evidence must cover the analysis grid")
df$origin_price <- m1_observable_price(df)

# Initial training/validation dates are fixed, not a percentage split.
split_date <- M1_PROTOCOL$validation_start
train <- df[seq_len(m1_observable_prefix(df, M1_PROTOCOL$train_end)), , drop = FALSE]
valid <- df %>% filter(date >= split_date, date <= M1_PROTOCOL$validation_end)
sapply(list(train = train, valid = valid), nrow)


# spot price plot: Descriptive EDA, using all historical data, only want to understand overall trend
# Model selection related test: using training set only

ggplot(df, aes(date, target_wti)) +
  geom_line(linewidth = 0.4, color = "#2563A6") +
  geom_vline(xintercept = as.Date(split_date), linetype = "dashed",
    color = "#D97706", linewidth = 0.6) +
  annotate("text", x = as.Date(split_date) + 30, y = Inf,
    label = paste0("Validation\n", as.Date(split_date)), hjust = 0, vjust = 1.5, 
    color = "#D97706", size = 3.5, fontface = "bold") +
  labs(title = "WTI Cushing Daily Spot Price",
    subtitle = "Historical prices with training and validation splits",
    x = "Date", y = "USD/barrel"
  ) +
  scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "#1F2937"),
    plot.subtitle = element_text(size = 11, color = "#64748B"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    plot.title.position = "plot"
  )


# since we have a negative value, use first difference rather than log first
df <- df %>% mutate(diff_wti = target_wti - lag(target_wti))
ggplot(df, aes(date, diff_wti)) +
  geom_line(linewidth = 0.4, color = "#2563A6") +
  geom_vline(xintercept = as.Date(split_date), linetype = "dashed",
             color = "#D97706", linewidth = 0.6) +
  annotate("text", x = as.Date(split_date) + 30, y = Inf,
           label = paste0("Validation\n", as.Date(split_date)), hjust = 0, vjust = 1.5, 
           color = "#D97706", size = 3.5, fontface = "bold") +
  labs(title = "WTI Cushing Daily Spot Price Difference",
       subtitle = "Historical price difference with training and validation splits",
       x = "Date", y = "USD/barrel"
  ) +
  scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "#1F2937"),
    plot.subtitle = element_text(size = 11, color = "#64748B"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    plot.title.position = "plot"
  )


y_train <- ts(train$target_wti, frequency = 1)
dy_train <- diff(y_train)


Box.test(y_train, lag = 20, type = "Ljung-Box")    
# lag = 20: 20 trading days approximately one trading month.


adf.test(y_train) # H0:  non-stationary, cannot reject H0
adf.test(dy_train) # H0 is rejected


ArchTest(as.numeric(dy_train), lags = 10)
# lag = 10: since it captures relatively short-run volatility dependence


par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
acf(y_train, main = "ACF - Price")
pacf(y_train, main = "PACF - Price")
acf(dy_train, main = "ACF - First Difference")
pacf(dy_train, main = "PACF - First Difference")
par(mfrow = c(1, 1))


summary(dy_train) # mean very close to 0
skewness(dy_train) # not symmetric 
kurtosis(dy_train) # excess kurtosis, very heavy tail
jarque.bera.test(dy_train) # H0: follow normal dist, reject H0
qqnorm(dy_train, main = "Normal QQ Plot of WTI First Differences")
qqline(dy_train, col = "red", lwd = 2)


dy <- as.numeric(dy_train)
train_sensitivity <- train %>% mutate(diff_wti = target_wti - lag(target_wti)) %>%
  filter(!date %in% as.Date(c("2020-04-20", "2020-04-21")))
dy_excl <- na.omit(train_sensitivity$diff_wti)
data.frame(
  Sample = c("Original", "Excluding extreme days"),
  Mean = c(mean(dy), mean(dy_excl)),
  SD = c(sd(dy), sd(dy_excl)),
  Skewness = c(skewness(dy), skewness(dy_excl)),
  Excess_Kurtosis = c(
    kurtosis(dy),
    kurtosis(dy_excl)
  )
)
jarque.bera.test(dy)
jarque.bera.test(dy_excl)
qqnorm(dy_excl, main = "Normal QQ Plot of WTI First Differences exclude negative value")
qqline(dy_excl, col = "red", lwd = 2)



orders <- list(
  RW = c(0, 1, 0),     # as benchmark
  ARIMA011 = c(0, 1, 1),
  ARIMA110 = c(1, 1, 0),
  ARIMA111 = c(1, 1, 1),
  ARIMA012 = c(0, 1, 2),
  ARIMA211 = c(2, 1, 1)
)

fits_arima <- lapply(orders, function(ord) {
  Arima(y_train, order = ord, include.drift = FALSE)
})
fits_drift_arima <- lapply(orders, function(ord) {
  Arima(y_train, order = ord, include.drift = TRUE)
})
lb_pvalue <- function(fit, lag = 20) {
  Box.test(residuals(fit), lag = 20, type = "Ljung-Box", 
           fitdf = sum(arimaorder(fit)[c("p", "q")]))$p.value
}
results_arima <- data.frame(
  Model = names(fits_arima),
  AIC = sapply(fits_arima, AIC),
  AIC_drift = sapply(fits_drift_arima, AIC),
  BIC = sapply(fits_arima, BIC),
  BIC_drift = sapply(fits_drift_arima, BIC),
  LB_pvalue = round(sapply(fits_arima, lb_pvalue), 4),
  LB_pvalue_drift = round(sapply(fits_drift_arima, lb_pvalue), 4)
)
results_arima
par(mfrow = c(2, 3))
for (name in names(fits_arima)) {
  fit <- fits_arima[[name]]
  acf(residuals(fit), lag.max = 60, main = name)
}
for (name in names(fits_drift_arima)) {
  fit <- fits_drift_arima[[name]]
  acf(residuals(fit), lag.max = 60, main = name)
}
par(mfrow = c(1, 1))

m1=Arima(y_train, order=c(0,1,1), include.drift=FALSE)
for (m in c(5,10,15,20)){
  p = Box.test(m1$residuals, lag = m, type = "Ljung-Box", fitdf=1)$p.value
  cat("Lag =", m, " p-value =", p, "\n")
}      # failed for larger lag > 15

train_early <- train$target_wti[train$date < as.Date("2020-01-01")]
train_late <- train$target_wti[train$date >= as.Date("2020-01-01")]
fit_early <- Arima(ts(train_early), order = c(0, 1, 1), include.drift = FALSE)
fit_late <- Arima(ts(train_late), order = c(0, 1, 1), include.drift = FALSE)
par(mfrow = c(1, 2))
acf(residuals(fit_early), lag.max = 60, main = "Before 2020")
acf(residuals(fit_late), lag.max = 60, main = "2020 onward")
par(mfrow = c(1, 1))
Box.test(residuals(fit_early), lag = 20, type = "Ljung-Box", fitdf = 1)
Box.test(residuals(fit_late), lag = 20, type = "Ljung-Box", fitdf = 1)
coef(fit_early)
coef(fit_late)




data_tv <- df
y <- data_tv$target_wti
n_train <- sum(df$date <= M1_PROTOCOL$train_end)
n_valid <- nrow(valid)
H <- max(M1_PROTOCOL$horizons)
origins <- n_train:(n_train + n_valid - H) # Common complete 60-row validation paths.
length(origins)
orders <- list(
  RW = c(0, 1, 0),
  ARIMA011 = c(0, 1, 1),
  ARIMA111 = c(1, 1, 1),
  ARIMA012 = c(0, 1, 2),
  ARIMA211 = c(2, 1, 1)
)
models <- c(
  "RW",
  "RW_drift",      # see drift performance
  "ARIMA011",
  "ARIMA111",
  "ARIMA012",
  "ARIMA211"
)
forecast_arima <- function(y_train, model, h = max(M1_PROTOCOL$horizons), bridge = 0L, anchor = NULL) {
  drift <- (model == "RW_drift")
  order <- if (drift) orders$RW else orders[[model]]
  fit <- Arima(ts(y_train), order = order, include.drift = drift)
  pred <- as.numeric(forecast(fit, h = h + bridge)$mean)[bridge + seq_len(h)]
  if (model == "RW" && !is.null(anchor)) pred[] <- anchor
  pred
}
actual <- t(sapply(origins, function(t) {y[(t + 1):(t + H)]}))
colnames(actual) <- paste0("h", 1:H)
rownames(actual) <- as.character(data_tv$date[origins])
predictions_arima <- list()
for (model in models) {
  cat("Running:", model, "\n")
  pred_mat <- matrix(NA_real_, nrow = length(origins), ncol = H)
  for (i in seq_along(origins)) {
    t <- origins[i]
    visible <- m1_observable_prefix(df, df$date[t])
    y_history <- y[seq_len(visible)]
    pred_mat[i, ] <- forecast_arima(y_history, model, H, t - visible, df$origin_price[t])
  }
  colnames(pred_mat) <- paste0("h", 1:H)
  predictions_arima[[model]] <- pred_mat
}
accuracy_results_arima <- data.frame()
for (model in models) {
  errors <- actual - predictions_arima[[model]]
  rmse <- sqrt(colMeans(errors^2))
  mape <- colMeans(abs(errors / actual)) * 100
  mae <- colMeans(abs(errors))
  accuracy_results_arima <- rbind(
    accuracy_results_arima,
    data.frame(
      Model = model,
      Horizon = 1:H,
      RMSE = rmse,
      MAPE = mape,
      MAE = mae
    )
  )
}
ggplot(accuracy_results_arima, aes(Horizon, RMSE, color = Model)) +
  geom_line(linewidth = 0.7) +
  labs(title = "Expanding-Window ARIMA on Validation", subtitle = "WTI Spot Price: 1–60 Steps Forecast",
    x = "Forecast Horizon", y = "Out-of-Sample RMSE (USD/barrel)", color = "Model") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "#1F2937"),
    plot.subtitle = element_text(size = 11, color = "#64748B"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    plot.title.position = "plot"
  )
ggplot(accuracy_results_arima, aes(Horizon, MAPE, color = Model)) +
  geom_line(linewidth = 0.7) +
  labs(title = "Expanding-Window ARIMA on Validation", subtitle = "WTI Spot Price: 1–60 Steps Forecast",
       x = "Forecast Horizon", y = "Out-of-Sample MAPE %", color = "Model") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "#1F2937"),
    plot.subtitle = element_text(size = 11, color = "#64748B"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    plot.title.position = "plot"
  )
ggplot(accuracy_results_arima, aes(Horizon, MAE, color = Model)) +
  geom_line(linewidth = 0.7) +
  labs(title = "Expanding-Window ARIMA on Validation", subtitle = "WTI Spot Price: 1–60 Steps Forecast",
       x = "Forecast Horizon", y = "Out-of-Sample MAE (USD/barrel)", color = "Model") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "#1F2937"),
    plot.subtitle = element_text(size = 11, color = "#64748B"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    plot.title.position = "plot"
  )
rw_rmse_arima <- accuracy_results_arima %>% 
  filter(Model == "RW") %>% 
  select(Horizon, RW_RMSE = RMSE, RW_MAPE = MAPE, RW_MAE = MAE)
relative_results <- accuracy_results_arima %>% 
  left_join(rw_rmse_arima, by = "Horizon") %>%
  mutate(Relative_RMSE = RMSE / RW_RMSE, Relative_MAPE = MAPE / RW_MAPE, Relative_MAE = MAE / RW_MAE) 
ggplot(relative_results, aes(Horizon, y = Relative_RMSE, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_line(linewidth = 0.7) +
  labs(title = "Relative RMSE Compared with Random Walk",
    x = "Forecast Horizon", y = "RMSE / Random Walk RMSE") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "#1F2937"),
    plot.subtitle = element_text(size = 11, color = "#64748B"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    plot.title.position = "plot"
  )
ggplot(relative_results, aes(Horizon, y = Relative_MAPE, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_line(linewidth = 0.7) +
  labs(title = "Relative MAPE Compared with Random Walk",
       x = "Forecast Horizon", y = "MAPE / Random Walk MAPE") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "#1F2937"),
    plot.subtitle = element_text(size = 11, color = "#64748B"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    plot.title.position = "plot"
  )
ggplot(relative_results, aes(Horizon, y = Relative_MAE, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_line(linewidth = 0.7) +
  labs(title = "Relative MAE Compared with Random Walk",
       x = "Forecast Horizon", y = "MAE / Random Walk MAE") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "#1F2937"),
    plot.subtitle = element_text(size = 11, color = "#64748B"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    plot.title.position = "plot"
  )
group_results_arima <- accuracy_results_arima %>%
  mutate(Group = case_when( Horizon <= 5 ~ "Short (1-5)", Horizon <= 10 ~ "Medium (6-10)", TRUE ~ "Long (11-60)")) %>%
  group_by(Model, Group) %>%
  summarise(Mean_MAPE = mean(MAPE), Mean_RMSE = mean(RMSE), Mean_MAE = mean(MAE), .groups = "drop") %>%
  arrange(Group, Mean_MAPE, Mean_RMSE, Mean_MAE)
group_results_arima %>% as.data.frame() %>%
  mutate(across(c(Mean_MAPE, Mean_RMSE, Mean_MAE), ~ sprintf("%.4f", .x))) %>%
  print(row.names = FALSE, right = FALSE)
data_tv <- df[seq_len(m1_observable_prefix(df, M1_PROTOCOL$fit_end)), , drop = FALSE]
y_tv <- ts(data_tv$target_wti, frequency = 1)
final_fit_arima <- Arima(y_tv, order = c(0, 1, 1), include.drift = FALSE)
summary(final_fit_arima)
Box.test(residuals(final_fit_arima), lag = 20, type = "Ljung-Box", fitdf = 1)
checkresiduals(final_fit_arima)


windows <- c(500, 1000)
predictions_sliding <- list()
for (W in windows) {
  for (model in models) {
    model_name <- paste0(model, "_W", W)
    cat("Running:", model_name, "\n")
    pred_mat <- matrix(NA_real_, nrow = length(origins), ncol = H)
    for (i in seq_along(origins)) {
      t <- origins[i]
      visible <- m1_observable_prefix(df, df$date[t])
      y_history <- tail(y[seq_len(visible)], W)
      pred_mat[i, ] <- forecast_arima(y_history, model, H, t - visible, df$origin_price[t])}
    colnames(pred_mat) <- paste0("h", 1:H)
    rownames(pred_mat) <- as.character(data_tv$date[origins])
    predictions_sliding[[model_name]] <- pred_mat
  }
}
accuracy_sliding <- data.frame()
for (name in names(predictions_sliding)) {
  errors <- actual - predictions_sliding[[name]]
  rmse <- sqrt(colMeans(errors^2))
  mape <- colMeans(abs(errors / actual)) * 100
  accuracy_sliding <- rbind(accuracy_sliding,
    data.frame(
      Model = name,
      Horizon = 1:H,
      RMSE = rmse,
      MAPE = mape
    )
  )
}
accuracy_sliding <- accuracy_sliding %>%
  mutate(Window = ifelse(grepl("W500", Model), "W = 500", "W = 1000"),
         Base_Model = sub("_W[0-9]+$", "", Model))
ggplot(accuracy_sliding, aes(x = Horizon, y = RMSE, color = Base_Model)) +
  geom_line(linewidth = 1) +
  facet_wrap(~ Window) +
  labs(title = "Sliding-Window ARIMA Validation",
       x = "Forecast Horizon (Trading Days)", y = "RMSE (USD/barrel)") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "#1F2937"),
    plot.subtitle = element_text(size = 11, color = "#64748B"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    plot.title.position = "plot",
    plot.margin = margin(15, 20, 15, 15)
  )
ggplot(accuracy_sliding, aes(x = Horizon, y = MAPE, color = Base_Model)) +
  geom_line(linewidth = 1) +
  facet_wrap(~ Window) +
  labs(title = "Sliding-Window ARIMA Validation",
       x = "Forecast Horizon (Trading Days)", y = "MAPE %") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "#1F2937"),
    plot.subtitle = element_text(size = 11, color = "#64748B"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    plot.title.position = "plot",
    plot.margin = margin(15, 20, 15, 15)
  )
rw_rmse_sliding <- accuracy_sliding %>% filter(Model == "RW_W500") %>%  #same for both windows
  select(Horizon, RW_RMSE = RMSE, RW_MAPE = MAPE)
relative_sliding <- accuracy_sliding %>% left_join(rw_rmse_sliding, by = "Horizon") %>%
  mutate(Relative_RMSE = RMSE / RW_RMSE, Relative_MAPE = MAPE / RW_MAPE)
ggplot(relative_sliding, aes(Horizon, Relative_RMSE, color = Base_Model)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_line(linewidth = 1) +
  facet_wrap(~ Window) +
  labs(title = "Relative RMSE Compared with Random Walk", 
       x = "Forecast Horizon", y = "RMSE / RW RMSE") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "#1F2937"),
    plot.subtitle = element_text(size = 11, color = "#64748B"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    plot.title.position = "plot",
    plot.margin = margin(15, 20, 15, 15)
  )
ggplot(relative_sliding, aes(Horizon, Relative_MAPE, color = Base_Model)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_line(linewidth = 1) +
  facet_wrap(~ Window) +
  labs(title = "Relative MAPE Compared with Random Walk", 
       x = "Forecast Horizon", y = "MAPE / RW MAPE") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "#1F2937"),
    plot.subtitle = element_text(size = 11, color = "#64748B"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    plot.title.position = "plot",
    plot.margin = margin(15, 20, 15, 15)
  )


par(mfrow = c(2, 3))
for (name in names(fits_arima)) {
  fit <- fits_arima[[name]]
  acf(residuals(fit)**2, main = name)  # arch effects
}
par(mfrow = c(1, 1)) 
arch_results <- data.frame(
  Model = names(fits_arima),
  ARCH_pvalue = sapply(fits_arima, function(fit) {ArchTest(na.omit(as.numeric(residuals(fit))), lags = 10)$p.value})
)
arch_results

orders <- list(
  ARIMA011 = c(0, 1),
  ARIMA012 = c(0, 2),
  ARIMA111 = c(1, 1),
  ARIMA211 = c(2, 1)
)
dy_train <- diff(train$target_wti)
fits_garch <- lapply(orders, function(pq) {
  formula <- as.formula(sprintf("~ arma(%d,%d) + garch(1,1)", pq[1], pq[2]))
  garchFit(formula = formula, data = dy_train, include.mean = FALSE, cond.dist = "std", trace = FALSE)
})
names(fits_garch) <- names(orders)
lb_pvalue_garch <- function(fit, lag = 20) {
  z <- na.omit(as.numeric(residuals(fit))/as.numeric(volatility(fit)))
  cf <- coef(fit)
  p <- sum(grepl("^ar[0-9]+$", names(cf)))
  q <- sum(grepl("^ma[0-9]+$", names(cf)))
  Box.test(z, lag = lag, type = "Ljung-Box", fitdf = p + q)$p.value
}
lb2_pvalue_garch <- function(fit, lag = 20) {
  z <- as.numeric(residuals(fit))/as.numeric(volatility(fit))
  z <- na.omit(z)
  Box.test(z**2, lag = lag, type = "Ljung-Box")$p.value
}
results_garch <- data.frame(
  Model = paste0(names(fits_garch), "-GARCH(1,1)"),
  AIC = sapply(fits_garch, function(fit) {unname(fit@fit$ics["AIC"]) * length(residuals(fit))}),
  BIC = sapply(fits_garch, function(fit) {unname(fit@fit$ics["BIC"]) * length(residuals(fit))}),
  LB_pvalue = round(sapply(fits_garch, lb_pvalue_garch), 4),
  left_Unexplained = round(sapply(fits_garch, lb2_pvalue_garch), 4)
)
results_garch


set.seed(123)
B <- 1000              
H <- max(M1_PROTOCOL$horizons)
actual <- t(vapply(origins, function(t) {y[(t + 1):(t + H)]}, numeric(H)))
stopifnot(all(is.finite(y)), n_valid >= H)
forecast_arima <- function(history, p, q, H) {
  fit <- Arima( ts(history), order = c(p, 1, q), include.drift = FALSE)
  fc <- forecast(fit, h = H, level = 95)
  list(point = as.numeric(fc$mean), lower = as.numeric(fc$lower[, 1]), upper = as.numeric(fc$upper[, 1]))
}
arma_mean_forecast <- function(dy, eps, phi, theta, H) {
  p <- length(phi)
  q <- length(theta)
  r_state <- as.numeric(dy)
  e_state <- as.numeric(eps)
  pred <- numeric(H)
  for (j in seq_len(H)) {
    mu <- 0
    if (p > 0) {
      mu <- mu + sum(phi * rev(tail(r_state, p)))
    }
    if (q > 0) {
      mu <- mu + sum(theta * rev(tail(e_state, q)))
    }
    pred[j] <- mu
    r_state <- c(r_state, mu)
    e_state <- c(e_state, 0)
  }
  pred
}
forecast_garch <- function(history, p, q, H, B) {
  dy <- diff(as.numeric(history))
  form <- as.formula(sprintf("~ arma(%d,%d) + garch(1,1)", p, q))
  fit <- garchFit(formula = form, data = dy, include.mean = FALSE, cond.dist = "std", trace = FALSE)
  cf <- coef(fit)
  required <- c("omega", "alpha1", "beta1", "shape")
  stopifnot(all(required %in% names(cf)))
  stopifnot(all(is.finite(cf)))
  omega <- unname(cf["omega"])
  alpha <- unname(cf["alpha1"])
  beta  <- unname(cf["beta1"])
  nu    <- unname(cf["shape"])
  if (omega <= 0 || alpha < 0 || beta < 0 ||alpha + beta >= 1 || nu <= 2) {
    stop("Invalid or nonstationary GARCH parameters")
  }
  phi <- if (p > 0) {as.numeric(cf[paste0("ar", 1:p)])} else numeric(0)
  theta <- if (q > 0) {as.numeric(cf[paste0("ma", 1:q)])} else numeric(0)
  stopifnot(all(is.finite(c(phi, theta))))
  eps <- as.numeric(residuals(fit))
  sig <- as.numeric(volatility(fit))
  stopifnot(length(eps) == length(dy),length(sig) == length(dy),all(is.finite(eps)), all(is.finite(sig)))
  n <- length(dy)
  last_price <- tail(history, 1)
  changes <- matrix(NA_real_, H, B)
  for (b in seq_len(B)) {
    r_state <- dy
    e_state <- eps
    variance_prev <- tail(sig, 1)^2
    for (j in seq_len(H)) {
      variance_new <- omega +alpha * tail(e_state, 1)^2 +beta * variance_prev
      if (!is.finite(variance_new) || variance_new <= 0) {stop("Invalid simulated variance")}
      z <- rt(1, df = nu) * sqrt((nu - 2) / nu)
      e_new <- sqrt(variance_new) * z
      mu <- 0
      if (p > 0) {mu <- mu + sum(phi * rev(tail(r_state, p)))}
      if (q > 0) {mu <- mu + sum(theta * rev(tail(e_state, q)))}
      r_new <- mu + e_new
      changes[j, b] <- r_new
      r_state <- c(r_state, r_new)
      e_state <- c(e_state, e_new)
      variance_prev <- variance_new
    }
  }
  price_paths <- apply(changes, 2, cumsum) + last_price
  price_paths <- matrix(price_paths, nrow = H)
  mean_changes <- arma_mean_forecast(dy = dy, eps = eps, phi = phi, theta = theta, H = H)
  
  point <- last_price + cumsum(mean_changes)
  lower <- apply(price_paths, 1, quantile, 0.025)
  upper <- apply(price_paths, 1, quantile, 0.975)
  list(point = as.numeric(point), lower = as.numeric(lower), upper = as.numeric(upper))
}
model_names <- c(paste0("ARIMA_", names(orders)), paste0("GARCH_", names(orders)))
predictions_garch <- list()
for (name in model_names) {
  cat("Running:", name, "\n")
  point_mat <- matrix(NA_real_, length(origins), H)
  lower_mat <- matrix(NA_real_, length(origins), H)
  upper_mat <- matrix(NA_real_, length(origins), H)
  base_name <- sub("^(ARIMA|GARCH)_", "", name)
  pq <- orders[[base_name]]
  for (i in seq_along(origins)) {
    t <- origins[i]
    visible <- m1_observable_prefix(df, df$date[t])
    bridge <- t - visible
    history <- y[seq_len(visible)]
    fc <- tryCatch({
      if (grepl("^ARIMA_", name)) {forecast_arima(history, pq[1], pq[2], H + bridge)}
      else {set.seed(123 + i)
        forecast_garch(history, pq[1], pq[2], H + bridge, B)}},
      error = function(e) {message(name, " origin ", i, ": ",conditionMessage(e))
        NULL
      })
    if (is.null(fc)) next
    point_mat[i, ] <- fc$point[bridge + seq_len(H)]
    lower_mat[i, ] <- fc$lower[bridge + seq_len(H)]
    upper_mat[i, ] <- fc$upper[bridge + seq_len(H)]
  }
  predictions_garch[[name]] <- list(point = point_mat, lower = lower_mat, upper = upper_mat)
}
fit_status <- data.frame(
  Model = names(predictions_garch),
  Failed = sapply(predictions_garch, function(x) {sum(!complete.cases(x$point, x$lower, x$upper))})
)
fit_status
common_ok <- Reduce(`&`,lapply(predictions_garch, function(x) {complete.cases(x$point, x$lower, x$upper)}))
cat("Common valid origins:", sum(common_ok),"/", length(origins), "\n")
if (!any(common_ok)) {stop("No common valid origins")}
actual_garch <- actual[common_ok, , drop = FALSE]
point_results_garch <- bind_rows(lapply(names(predictions_garch), function(name) {
  pred <- predictions_garch[[name]]$point[common_ok, , drop = FALSE]
  err <- actual_garch - pred
  data.frame(Model = name, Horizon = 1:H, RMSE = sqrt(colMeans(err^2)), MAPE = colMeans(abs(err / actual_garch)) * 100, MAE = colMeans(abs(err)))}))
ggplot(point_results_garch, aes(x = Horizon, y = RMSE, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
  geom_line(linewidth = 1) +
  labs(title = "ARIMA vs ARIMA-GARCH: RMSE", subtitle = "Expanding-Window Validation",
       x = "Forecast Horizon", y = "RMSE", color = "Model") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "#1F2937"),
    plot.subtitle = element_text(size = 11, color = "#64748B"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    plot.title.position = "plot"
  )
ggplot(point_results_garch, aes(x = Horizon, y = MAPE, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
  geom_line(linewidth = 1) +
  labs(title = "ARIMA vs ARIMA-GARCH: MAPE", subtitle = "Expanding-Window Validation",
       x = "Forecast Horizon", y = "MAPE", color = "Model") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "#1F2937"),
    plot.subtitle = element_text(size = 11, color = "#64748B"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    plot.title.position = "plot"
  )
ggplot(point_results_garch, aes(x = Horizon, y = MAE, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
  geom_line(linewidth = 1) +
  labs(title = "ARIMA vs ARIMA-GARCH: MAE", subtitle = "Expanding-Window Validation",
       x = "Forecast Horizon", y = "MAE", color = "Model") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "#1F2937"),
    plot.subtitle = element_text(size = 11, color = "#64748B"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    plot.title.position = "plot"
  )
rw_pred_grach <- matrix(df$origin_price[origins[common_ok]], nrow = sum(common_ok), ncol = H)
rw_rmse_garch <- sqrt(colMeans((actual_garch - rw_pred_grach)^2))
rw_mape_garch <- colMeans(abs((actual_garch - rw_pred_grach) / actual_garch))*100
rw_mae_garch <- colMeans(abs(actual_garch - rw_pred_grach))
point_results_garch <- point_results_garch %>% mutate(Relative_RMSE = RMSE / rw_rmse_garch[Horizon],
                                                      Relative_MAPE = MAPE / rw_mape_garch[Horizon],
                                                      Relative_MAE = MAE / rw_mae_garch[Horizon])
ggplot(point_results_garch, aes(Horizon, Relative_RMSE, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_line(linewidth = 1) +
  labs(title = "ARIMA vs ARIMA-GARCH: Relative RMSE", x = "Forecast Horizon", y = "RMSE / RW RMSE") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "#1F2937"),
    plot.subtitle = element_text(size = 11, color = "#64748B"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    plot.title.position = "plot"
  )
ggplot(point_results_garch, aes(Horizon, Relative_MAPE, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_line(linewidth = 1) +
  labs(title = "ARIMA vs ARIMA-GARCH: Relative MAPE", x = "Forecast Horizon", y = "MAPE / RW MAPE") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "#1F2937"),
    plot.subtitle = element_text(size = 11, color = "#64748B"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    plot.title.position = "plot"
  )
ggplot(point_results_garch, aes(Horizon, Relative_MAE, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_line(linewidth = 1) +
  labs(title = "ARIMA vs ARIMA-GARCH: Relative MAE", x = "Forecast Horizon", y = "MAE / RW MAE") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "#1F2937"),
    plot.subtitle = element_text(size = 11, color = "#64748B"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    plot.title.position = "plot"
  )
interval_results <- bind_rows(lapply(names(predictions_garch), function(name) {
  lower <- predictions_garch[[name]]$lower[common_ok, , drop = FALSE]
  upper <- predictions_garch[[name]]$upper[common_ok, , drop = FALSE]
  alpha_level <- 0.05
  covered <- actual_garch >= lower & actual_garch <= upper
  width <- upper - lower
  score <- width + (2 / alpha_level) * pmax(lower - actual_garch, 0) + (2 / alpha_level) * pmax(actual_garch - upper, 0)
  data.frame(Model = name, Horizon = 1:H, Coverage = colMeans(covered),Width = colMeans(width),Interval_Score = colMeans(score))
}))
ggplot(interval_results, aes(Horizon, Coverage, color = Model)) +
  geom_hline(yintercept = 0.95, linetype = "dashed") +
  geom_line(linewidth = 1) +
  labs(title = "95% Prediction Interval Coverage",
       x = "Forecast Horizon", y = "Coverage") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "#1F2937"),
    plot.subtitle = element_text(size = 11, color = "#64748B"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    plot.title.position = "plot",
    plot.margin = margin(15, 20, 15, 15)
  )
ggplot(interval_results, aes(Horizon, Width, color = Model)) +
  geom_line(linewidth = 1) +
  labs(title = "95% Prediction Interval Width", x = "Forecast Horizon", y = "Average Width (USD/barrel)") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "#1F2937"),
    plot.subtitle = element_text(size = 11, color = "#64748B"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    plot.title.position = "plot",
    plot.margin = margin(15, 20, 15, 15)
  )
ggplot(interval_results, aes(Horizon, Interval_Score, color = Model)) +
  geom_line(linewidth = 1) +
  labs(title = "95% Interval Score (Lower Is Better)",
       x = "Forecast Horizon", y = "Interval Score") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", color = "#1F2937"),
    plot.subtitle = element_text(size = 11, color = "#64748B"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10, color = "#475569"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#E5E7EB", linewidth = 0.4),
    plot.title.position = "plot",
    plot.margin = margin(15, 20, 15, 15)
  )
group_results_garch <- point_results_garch %>%
  mutate(Group = case_when( Horizon <= 5 ~ "Short (1-5)", Horizon <= 10 ~ "Medium (6-10)", TRUE ~ "Long (11-60)")) %>%
  group_by(Model, Group) %>%
  summarise(Mean_MAPE = mean(MAPE), Mean_RMSE = mean(RMSE), Mean_MAE = mean(MAE), .groups = "drop") %>%
  arrange(Group, Mean_MAPE, Mean_RMSE, Mean_MAE)
print(group_results_garch, n=Inf) 

interval_summary <- interval_results %>% 
  mutate(Group = case_when( Horizon <= 5 ~ "Short (1-5)", Horizon <= 10 ~ "Medium (6-10)", TRUE ~ "Long (11-60)")) %>%
  group_by(Model, Group) %>%
  summarise(Mean_Coverage = mean(Coverage), Mean_Width = mean(Width), 
            Mean_Interval_Score = mean(Interval_Score),.groups = "drop") %>%
  arrange(Group, Mean_Interval_Score, Mean_Coverage, Mean_Width)
summary_table_garch <- left_join(group_results_garch, interval_summary, by = c("Model", "Group")) %>% arrange(Group, Mean_MAPE, Mean_RMSE)
summary_table_garch %>% as.data.frame() %>%
  mutate(across(c(Mean_MAPE, Mean_RMSE, Mean_MAE), ~ sprintf("%.4f", .x))) %>%
  print(row.names = FALSE, right = FALSE)
data_tv <- df[seq_len(m1_observable_prefix(df, M1_PROTOCOL$fit_end)), , drop = FALSE]
dy_tv <- diff(data_tv$target_wti)
final_fit_garch <- garchFit(formula = ~ arma(0,1) + garch(1,1), data = dy_tv, 
                            include.mean = FALSE, cond.dist = "std", trace = FALSE)
summary(final_fit_garch)


features <- read.csv(file.path(.m1_root, "data", "processed", "features.csv"))
vars <- features %>% filter(group=="market") %>% pull(column) 
cor_results <- data.frame( Variable = vars,
                           Correlation = sapply(vars, function(v) {
                             cor(diff(train$target_wti), diff(train[[v]]))}))
cor_results %>% arrange(abs(Correlation))
par(mfrow = c(2, 4))
for (v in vars) {
  dy <- diff(train$target_wti)
  dx <- diff(train[[v]])
  ccf(dy, dx, lag.max = 20, main = paste("CCF: WTI vs", v), xlab = "Lag (Trading Days)")
}
par(mfrow = c(1, 1))
x_vars <- c(
  "yahoo_dollar_index_close",
  "oil_implied_volatility",
  "sp500_close",
  "breakeven_inflation_10y"
)
features %>% filter(column %in% x_vars) %>% select(column, group, frequency, available)
adf.test(train$yahoo_dollar_index_close)
adf.test(train$oil_implied_volatility)
adf.test(train$sp500_close)
adf.test(train$breakeven_inflation_10y)
kpss.test(train$oil_implied_volatility, null = "Level")  # confliting results
acf(train$oil_implied_volatility)
acf(diff(train$oil_implied_volatility))

dy <- diff(train$target_wti)
X_diff <- as.data.frame(lapply(train[, x_vars, drop = FALSE], diff))
reg_data <- data.frame(d_wti = dy, X_diff)
reg_data <- na.omit(reg_data)
MLR <- lm(d_wti ~ . - 1, data = reg_data)
summary(MLR)     # only sp500_close is siginificant
par(mfrow = c(1, 2))
acf(residuals(MLR), main = "ACF of Regression Residuals")
pacf(residuals(MLR), main = "PACF of Regression Residuals")
MLR2 <- lm(d_wti ~ sp500_close - 1, data = reg_data)
acf(residuals(MLR2), main = "ACF of Regression Residuals")
pacf(residuals(MLR2), main = "PACF of Regression Residuals")
MLR3 <- lm(d_wti ~ sp500_close + oil_implied_volatility  - 1, data = reg_data)
acf(residuals(MLR3), main = "ACF of Regression Residuals")
pacf(residuals(MLR3), main = "PACF of Regression Residuals")
MLR4 <- lm(d_wti ~ sp500_close + yahoo_dollar_index_close  - 1, data = reg_data)
acf(residuals(MLR4), main = "ACF of Regression Residuals")
pacf(residuals(MLR4), main = "PACF of Regression Residuals")
par(mfrow = c(1, 1))
x_sets <- list(
  SP500 = c("sp500_close"),
  SP500_OVX = c("sp500_close", "oil_implied_volatility"),
  SP500_Dollar = c("sp500_close", "yahoo_dollar_index_close"),
  SP500_OVX_Dollar = c("sp500_close", "oil_implied_volatility", "yahoo_dollar_index_close"))
orders <- list(
  ARIMA011 = c(0, 1, 1),
  ARIMA012 = c(0, 1, 2),
  ARIMA111 = c(1, 1, 1),
  ARIMA211 = c(2, 1, 1),
  ARIMA311 = c(3, 1, 1)
)
fits_arimax <- list()
for (s in names(x_sets)) {
  for (o in names(orders)) {
    name <- paste(s, o, sep = "_")
    fits_arimax[[name]] <- Arima(train$target_wti, order = orders[[o]], 
                                 xreg = as.matrix(train[, x_sets[[s]], drop = FALSE]), include.drift = FALSE)
  }
}
lb_pvalue <- function(fit, lag = 20) {
  Box.test(residuals(fit), lag = 20, type = "Ljung-Box", 
           fitdf = sum(arimaorder(fit)[c("p", "q")]))$p.value
}
results <- data.frame(
  Model = names(fits_arimax),
  AIC = sapply(fits_arimax, AIC),
  BIC = sapply(fits_arimax, BIC),
  LB_pvalue = round(sapply(fits_arimax, lb_pvalue), 4),
  row.names = NULL
) 
results
check_model <- c("SP500_OVX_ARIMA011", "SP500_OVX_ARIMA111", "SP500_ARIMA011")
par(mfrow = c(1, 3))
for (name in check_model) {
  acf(residuals(fits_arimax[[name]]), lag.max = 40, main = name)}
par(mfrow = c(1, 1))
train_late <- train %>% filter(date >= as.Date("2020-01-01"))
fit_late_arimax <- Arima(train_late$target_wti, order = c(0, 1, 1),
                         xreg = as.matrix(train_late[, c("sp500_close", "oil_implied_volatility"),drop = FALSE]),
                         include.drift = FALSE)
acf(residuals(fit_late_arimax), lag.max = 40)
Box.test(residuals(fit_late_arimax), lag = 20, type = "Ljung-Box", fitdf = 1)


# Explicit fixed-at-origin external-variable scenario throughout each 60-row path.
H <- max(M1_PROTOCOL$horizons)
data_val <- df
y <- data_val$target_wti
n_train <- sum(df$date <= M1_PROTOCOL$train_end)
n_valid <- nrow(valid)
origins <- n_train:(n_train + n_valid - H) 
length(origins)
actual <- t(sapply(origins, function(t) {y[(t + 1):(t + H)]}))
forecast_arimax <- function(t, xvars, order) {
  visible <- m1_observable_prefix(data_val, data_val$date[t])
  bridge <- t - visible
  y_train <- y[seq_len(visible)]
  X_train <- as.matrix(data_val[seq_len(visible), xvars, drop = FALSE])
  fit <- Arima(y_train, order = order, xreg = X_train, include.drift = FALSE)
  last_x <- as.numeric(unlist(data_val[t, xvars, drop = FALSE], use.names = FALSE))
  X_future <- matrix(rep(last_x, each = H + bridge), nrow = H + bridge, ncol = length(xvars))
  colnames(X_future) <- colnames(X_train)
  as.numeric(forecast(fit, h = H + bridge, xreg = X_future)$mean)[bridge + seq_len(H)]
}
predictions_arimax <- list()
predictions_arimax[["RW"]] <- matrix(data_val$origin_price[origins],nrow = length(origins),ncol = H)
predictions_arimax[["ARIMA011"]] <- t(sapply(origins, function(t) {
  visible <- m1_observable_prefix(data_val, data_val$date[t])
  bridge <- t - visible
  fit <- Arima(y[seq_len(visible)], order = c(0, 1, 1), include.drift = FALSE)
  as.numeric(forecast(fit, h = H + bridge)$mean)[bridge + seq_len(H)]
})
)
for (set_name in names(x_sets)) {
  for (order_name in names(orders)) {
    model_name <- paste(set_name, order_name, sep = "_")
    cat("Running:", model_name, "\n")
    predictions_arimax[[model_name]] <- t(sapply(origins, function(t) {
      forecast_arimax(t, xvars = x_sets[[set_name]], order = orders[[order_name]])})
    )
  }
}
point_results_arimax <- bind_rows(lapply(names(predictions_arimax), function(name) {
  err <- actual - predictions_arimax[[name]]
  data.frame( Model = name, Horizon = 1:H, RMSE = sqrt(colMeans(err^2)), MAPE = colMeans(abs(err / actual)) * 100, MAE = colMeans(abs(err)))
}))
rw_rmse_arimax <- point_results_arimax %>% filter(Model == "RW") %>% arrange(Horizon) %>% pull(RMSE)
rw_mape_arimax <- point_results_arimax %>% filter(Model == "RW") %>% arrange(Horizon) %>% pull(MAPE)
rw_mae_arimax <- point_results_arimax %>% filter(Model == "RW") %>% arrange(Horizon) %>% pull(MAE)
point_results_arimax <- point_results_arimax %>%
  mutate( Relative_RMSE = RMSE / rw_rmse_arimax[Horizon], Relative_MAPE = MAPE / rw_mape_arimax[Horizon],
          Relative_MAE = MAE / rw_mae_arimax[Horizon])
summary_results_arimax <- point_results_arimax %>%
  mutate(Horizon_Group = case_when(Horizon <= 5 ~ "Short (1-5)",
                                   Horizon <= 10 ~ "Medium (6-10)",
                                   TRUE ~ "Long (11-60)")) %>%
  group_by(Model, Horizon_Group) %>%
  summarise( Mean_MAPE = mean(MAPE), Mean_RMSE = mean(RMSE), Mean_MAE = mean(MAE), .groups = "drop") %>%
  arrange(Horizon_Group, Mean_MAPE, Mean_RMSE)
summary_results_arimax %>% as.data.frame() %>%
  mutate(across(c(Mean_MAPE, Mean_RMSE, Mean_MAE), ~ sprintf("%.4f", .x))) %>%
  print(row.names = FALSE, right = FALSE)
overall_results_arimax <- point_results_arimax %>%
  group_by(Model) %>%
  summarise(Mean_MAPE = mean(MAPE), Mean_RMSE = mean(RMSE), Mean_MAE = mean(MAE), .groups = "drop") %>%
  arrange(Mean_RMSE)
overall_results_arimax %>% as.data.frame() %>%
  mutate(across(c(Mean_MAPE, Mean_RMSE, Mean_MAE), ~ sprintf("%.4f", .x))) %>%
  print(row.names = FALSE, right = FALSE)
ggplot(point_results_arimax, aes(Horizon, RMSE, color = Model)) +
  geom_line(linewidth = 0.8) +
  labs(title = "ARIMAX: Expanding-Window Validation", 
       subtitle = "WTI Price Forecasting (1-60 Trading Days)",
       x = "Forecast Horizon", y = "RMSE (USD/barrel)") +
  theme_minimal()
ggplot(point_results_arimax, aes(Horizon, MAPE, color = Model)) +
  geom_line(linewidth = 0.8) +
  labs(title = "ARIMAX: Expanding-Window Validation", 
       subtitle = "WTI Price Forecasting (1-60 Trading Days)",
       x = "Forecast Horizon", y = "MAPE%") +
  theme_minimal()
ggplot(point_results_arimax, aes(Horizon, MAE, color = Model)) +
  geom_line(linewidth = 0.8) +
  labs(title = "ARIMAX: Expanding-Window Validation", 
       subtitle = "WTI Price Forecasting (1-60 Trading Days)",
       x = "Forecast Horizon", y = "MAE (USD/barrel)") +
  theme_minimal()
ggplot(point_results_arimax, aes(Horizon, Relative_RMSE, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_line(linewidth = 0.8) +
  labs(title = "ARIMAX: Relative RMSE vs Random Walk",
       x = "Forecast Horizon", y = "RMSE / Random Walk RMSE") +
  theme_minimal()
ggplot(point_results_arimax, aes(Horizon, Relative_MAPE, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_line(linewidth = 0.8) +
  labs(title = "ARIMAX: Relative MAPE vs Random Walk",
       x = "Forecast Horizon", y = "MAPE / Random Walk MAPE") +
  theme_minimal()
ggplot(point_results_arimax, aes(Horizon, Relative_MAE, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_line(linewidth = 0.8) +
  labs(title = "ARIMAX: Relative MAE vs Random Walk",
       x = "Forecast Horizon", y = "MAE / Random Walk MAE") +
  theme_minimal()
point_results_constant <- point_results_arimax
summary_results_constant <- summary_results_arimax
overall_results_constant <- overall_results_arimax
predictions_constant<- predictions_arimax


data_tv <- df[seq_len(m1_observable_prefix(df, M1_PROTOCOL$fit_end)), , drop = FALSE]
y_tv <- data_tv$target_wti
X_tv <- as.matrix(data_tv[, "sp500_close", drop = FALSE])
final_fit_arimax <- Arima(y_tv, order = c(0, 1, 1), xreg = X_tv,include.drift = FALSE)
summary(final_fit_arimax)
res_arimax<- residuals(final_fit_arimax)
Box.test(res_arimax, lag=20, type="Ljung-Box", fitdf = 1)
checkresiduals(final_fit_arimax)
acf(res_arimax, main = "ACF - ARIMAX Residuals")


price_changes <- data_val %>%
  mutate(date = as.Date(date)) %>%
  arrange(date) %>%
  mutate(
    Year = format(date, "%Y"),
    Diff_WTI = target_wti - lag(target_wti)) %>%
  filter(date >= as.Date("2023-09-01"), date <= as.Date("2026-09-30"))
volatility_by_year <- price_changes %>%
  group_by(Year) %>%
  summarise(
    N = sum(is.finite(Diff_WTI)),
    Mean_Diff = mean(Diff_WTI, na.rm = TRUE),
    SD_Diff = sd(Diff_WTI, na.rm = TRUE),
    Mean_Abs_Diff = mean(abs(Diff_WTI), na.rm = TRUE),
    Max_Abs_Diff = max(abs(Diff_WTI), na.rm = TRUE),
    Large_Change_5 = sum(abs(Diff_WTI) > 5, na.rm = TRUE),
    Large_Change_10 = sum(abs(Diff_WTI) > 10, na.rm = TRUE),
    .groups = "drop")
volatility_by_year
ggplot(volatility_by_year, aes(x = Year, y = SD_Diff)) +
  geom_col(fill = "#2563A6", width = 0.65) +
  geom_text(aes(label = round(SD_Diff, 2)), vjust = -0.5, size = 4) +
  labs(title = "Annual Volatility of WTI Price Changes", subtitle = "Sep 2023- Sep 2026",
       x = "Year", y = "SD of Daily Price Changes (USD/barrel)") +
  theme_minimal(base_size = 13)
ggplot(price_changes, aes(x = date, y = Diff_WTI)) +
  geom_line(color = "#2563A6", linewidth = 0.4) +
  facet_wrap(~ Year, scales = "free_x", ncol = 2) +
  labs(title = "Daily WTI Price Changes by Year", x = "Date", y = "Daily Price Change (USD/barrel)") +
  theme_minimal(base_size = 13)


models_yearly <- c("RW", "ARIMA011", "SP500_ARIMA011")
H <- ncol(actual)
yearly_evaluation <- bind_rows(
  lapply(models_yearly, function(model) {
    pred <- predictions_arimax[[model]]
    bind_rows(lapply(seq_along(origins), function(i) {
      t <- origins[i]
      data.frame(
        Model = model,
        Origin_Date = as.Date(data_val$date[t]),
        Target_Date = as.Date(data_val$date[(t + 1):(t + H)]),
        Horizon = 1:H,
        Actual = as.numeric(actual[i, ]),
        Predicted = as.numeric(pred[i, ])
      )}))})) %>%
  mutate(
    Year = format(Target_Date, "%Y"),
    Error = Actual - Predicted,
    SE = Error^2,
    AE = abs(Error),
    APE = ifelse(
      Actual != 0,
      abs(Error / Actual) * 100,
      NA_real_
    )) %>%
  filter(Year %in% c("2023", "2024", "2025", "2026"))
accuracy_yearly <- yearly_evaluation %>%
  group_by(Year, Model, Horizon) %>%
  summarise(
    N = n(),
    RMSE = sqrt(mean(SE)),
    MAE = mean(AE),
    MAPE = mean(APE, na.rm = TRUE),
    .groups = "drop"
  )
rw_yearly <- accuracy_yearly %>%
  filter(Model == "RW") %>%
  select(
    Year,
    Horizon,
    RW_RMSE = RMSE,
    RW_MAE = MAE,
    RW_MAPE = MAPE
  )
accuracy_yearly <- accuracy_yearly %>%
  left_join(rw_yearly, by = c("Year", "Horizon")) %>%
  mutate(
    Relative_RMSE = RMSE / RW_RMSE,
    Relative_MAE = MAE / RW_MAE,
    Relative_MAPE = MAPE / RW_MAPE
  )
summary_yearly <- accuracy_yearly %>%
  group_by(Year, Model) %>%
  summarise(
    Mean_RMSE = mean(RMSE),
    Mean_MAE = mean(MAE),
    Mean_MAPE = mean(MAPE),
    Mean_Relative_RMSE = mean(Relative_RMSE),
    Mean_Relative_MAE = mean(Relative_MAE),
    Mean_Relative_MAPE = mean(Relative_MAPE),
    .groups = "drop"
  ) %>%
  arrange(Year, Mean_RMSE)
summary_yearly
ggplot(accuracy_yearly, aes(Horizon, Relative_RMSE, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_line(linewidth = 0.9) +
  facet_wrap(~ Year, ncol = 2) +
  labs(title = "Annual Relative RMSE vs Random Walk", subtitle = "Relative RMSE below 1 indicates improvement",
       x = "Forecast Horizon (Trading Days)", y = "RMSE / RW RMSE") +
  theme_minimal(base_size = 13)
ggplot(accuracy_yearly, aes(Horizon, Relative_MAPE, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_line(linewidth = 0.9) +
  facet_wrap(~ Year, ncol = 2) +
  labs(title = "Annual Relative MAPE vs Random Walk", x = "Forecast Horizon (Trading Days)", y = "MAPE / RW MAPE") +
  theme_minimal(base_size = 13)

data_tv %>% arrange(date) %>%
  mutate(`Diff_WTI%` = 100*(target_wti - lag(target_wti))/lag(target_wti)) %>%
  filter(date >= as.Date("2026-01-01"), date <= as.Date("2026-09-30")) %>%
  arrange(desc(abs(`Diff_WTI%`))) %>%
  select(date, target_wti, `Diff_WTI%`) %>%
  filter(abs(`Diff_WTI%`) > 10)


y <- data_val$target_wti
n_train <- sum(df$date <= M1_PROTOCOL$train_end)
n_valid <- nrow(valid)
H <- max(M1_PROTOCOL$horizons)
origins <- n_train:(n_train + n_valid - H)
windows <- c(500, 1000)
x_sets <- list(SP500 = c("sp500_close"))
orders <- list(
  ARIMA011 = c(0, 1, 1),
  ARIMA211 = c(2, 1, 1)
)
all_xvars <- unique(unlist(x_sets))
actual <- t(sapply(origins, function(t) {y[(t + 1):(t + H)]}))
colnames(actual) <- paste0("h", 1:H)
rownames(actual) <- as.character(data_val$date[origins])
forecast_arimax <- function(t, xvars, order, W = NULL) {
  visible <- m1_observable_prefix(data_val, data_val$date[t])
  bridge <- t - visible
  idx <- if (is.null(W)) seq_len(visible) else seq.int(max(1L, visible - W + 1L), visible)
  y_history <- y[idx]
  X_history <- as.matrix(data_val[idx, xvars, drop = FALSE])
  fit <- Arima(y_history,order = order,xreg = X_history,include.drift = FALSE)
  last_x <- as.numeric(unlist(data_val[t, xvars, drop = FALSE], use.names = FALSE))
  X_future <- matrix(rep(last_x, each = H + bridge), nrow = H + bridge, ncol = length(xvars))
  colnames(X_future) <- colnames(X_history)
  as.numeric(forecast(fit, h = H + bridge, xreg = X_future)$mean)[bridge + seq_len(H)]}
forecast_arima <- function(t) {
  visible <- m1_observable_prefix(data_val, data_val$date[t])
  bridge <- t - visible
  fit <- Arima(y[seq_len(visible)], order = c(0, 1, 1), include.drift = FALSE)
  as.numeric(forecast(fit, h = H + bridge)$mean)[bridge + seq_len(H)]
}
predictions_sliding <- list()
predictions_sliding[["RW"]] <- matrix(data_val$origin_price[origins], nrow = length(origins), ncol = H)
predictions_sliding[["ARIMA011"]] <- matrix(NA_real_, nrow = length(origins), ncol = H)
for (set_name in names(x_sets)) {
  for (order_name in names(orders)) {
    base_name <- paste(set_name, order_name, sep = "_")
    
    predictions_sliding[[paste0(base_name, "_Expanding")]] <- matrix(NA_real_, length(origins), H)
    for (W in windows) {
      predictions_sliding[[paste0(base_name, "_W", W)]] <- matrix(NA_real_, length(origins), H)
    }
  }
}
for (i in seq_along(origins)){ 
  t <- origins[i]
  if (i %% 50 == 0) {
    cat("Running origin:", i, "/", length(origins), "\n")
  }
  predictions_sliding[["ARIMA011"]][i, ] <- forecast_arima(t)
  for (set_name in names(x_sets)) {
    xvars <- x_sets[[set_name]]
    for (order_name in names(orders)) {
      order <- orders[[order_name]]
      base_name <- paste(set_name, order_name, sep = "_")
      predictions_sliding[[paste0(base_name, "_Expanding")]][i, ] <- forecast_arimax(
        t = t, xvars = xvars, order = order, W = NULL )
      for (W in windows) {
        model_name <- paste0(base_name, "_W", W)
        predictions_sliding[[model_name]][i, ] <-forecast_arimax(
          t = t, xvars = xvars, order = order, W = W)}}}}
fit_status_sliding <- data.frame(
  Model = names(predictions_sliding),
  Failed = sapply(predictions_sliding, function(x) {
    sum(!complete.cases(x))
  })
)
fit_status_sliding
accuracy_sliding <- bind_rows(
  lapply(names(predictions_sliding), function(model) {
    pred <- predictions_sliding[[model]]
    err <- actual - pred
    data.frame(
      Model = model,
      Horizon = 1:H,
      RMSE = sqrt(colMeans(err^2)),
      MAE = colMeans(abs(err)),
      MAPE = colMeans(abs(err / actual)) * 100
    )
  })
)
rw_results <- accuracy_sliding %>% filter(Model == "RW") %>%
  select(
    Horizon,
    RW_RMSE = RMSE,
    RW_MAE = MAE,
    RW_MAPE = MAPE
  )
relative_sliding <- accuracy_sliding %>%
  left_join(rw_results, by = "Horizon") %>%
  mutate(
    Relative_RMSE = RMSE / RW_RMSE,
    Relative_MAE = MAE / RW_MAE,
    Relative_MAPE = MAPE / RW_MAPE
  )
group_results_sliding <- relative_sliding %>%
  mutate(Group = case_when( Horizon <= 5 ~ "Short (1-5)", Horizon <= 10 ~ "Medium (6-10)", TRUE ~ "Long (11-60)")) %>%
  group_by(Model, Group) %>%
  summarise(
    Mean_RMSE = mean(RMSE),
    Mean_MAE = mean(MAE),
    Mean_MAPE = mean(MAPE),
    Mean_Relative_RMSE = mean(Relative_RMSE),
    Mean_Relative_MAE = mean(Relative_MAE),
    Mean_Relative_MAPE = mean(Relative_MAPE),
    .groups = "drop"
  ) %>%
  arrange(Group, Mean_RMSE)
print(group_results_sliding, n = Inf, width = Inf)
overall_results_sliding <- relative_sliding %>%
  group_by(Model) %>%
  summarise(
    Mean_RMSE = mean(RMSE),
    Mean_MAE = mean(MAE),
    Mean_MAPE = mean(MAPE),
    Mean_Relative_RMSE = mean(Relative_RMSE),
    Mean_Relative_MAE = mean(Relative_MAE),
    Mean_Relative_MAPE = mean(Relative_MAPE),
    .groups = "drop"
  ) %>%
  arrange(Mean_RMSE)
ggplot(accuracy_sliding, aes(Horizon, MAPE, color = Model)) +
  geom_line(linewidth = 0.8) +
  labs( title = "ARIMAX: Expanding vs Sliding Window", subtitle = "WTI Validation (2023-Sep 2026)",
        x = "Forecast Horizon (Trading Days)", y = "RMSE (USD/barrel)") +
  theme_minimal(base_size = 13)
ggplot(relative_sliding,aes(Horizon, Relative_RMSE, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_line(linewidth = 0.8) +
  labs(title = "ARIMAX: Relative RMSE vs Random Walk", subtitle = "Expanding vs Sliding Window",
       x = "Forecast Horizon", y = "RMSE / RW RMSE") +
  theme_minimal(base_size = 13)
ggplot(relative_sliding, aes(Horizon, Relative_MAPE, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_line(linewidth = 0.8) +
  labs(title = "ARIMAX: Relative MAPE vs Random Walk", subtitle = "Expanding vs Sliding Window",
       x = "Forecast Horizon", y = "MAPE / RW MAPE") +
  theme_minimal(base_size = 13)

library(vars)
library(urca)

daily_spot <- read.csv(file.path(.m1_root, "data", "time_series", "daily_spot.csv")) %>%
  dplyr::select(date, target_wti, brent_spot, ny_gasoline_spot, ny_heating_oil_spot) %>%
  mutate(date=as.Date(date)) %>%
  arrange(date) %>%
  filter(date <= M1_PROTOCOL$validation_end)
# Common WTI release schedule for other spots is an explicit assumption, not evidence.
daily_spot$available_date <- .m1_publication$available_date[match(daily_spot$date, .m1_publication$date)]
daily_spot$origin_price <- df$origin_price[match(daily_spot$date, df$date)]
features <- read.csv(file.path(.m1_root, "data", "processed", "features.csv"))


split_date <- M1_PROTOCOL$validation_start
train_daily <- daily_spot[seq_len(m1_observable_prefix(daily_spot, M1_PROTOCOL$train_end)), , drop = FALSE]
valid_daily <- daily_spot %>% filter(date >= split_date, date <= M1_PROTOCOL$validation_end)


Y_train <- train_daily %>% dplyr::select(target_wti, brent_spot, ny_gasoline_spot, ny_heating_oil_spot) %>%
  as.matrix()
x_vars_var <- c(
  "brent_spot",
  "ny_gasoline_spot",
  "ny_heating_oil_spot"
)
MTSplot(Y_train)  
MTSplot(diff(Y_train))
mean(diff(train_daily$brent_spot))
mean(diff(train_daily$ny_gasoline_spot))
mean(diff(train_daily$ny_heating_oil_spot))
stationarity_results <- bind_rows(
  lapply(colnames(Y_train), function(v) {
    x <- Y_train[,v]
    data.frame(Variable = v, ADF_Level = suppressWarnings(adf.test(x)$p.value),
               KPSS_Level = suppressWarnings(kpss.test(x, null = "Level")$p.value),
               ADF_Diff = suppressWarnings(adf.test(diff(x))$p.value),
               KPSS_Diff = suppressWarnings(kpss.test(diff(x), null = "Level")$p.value)) }))
stationarity_results 
par(mfrow = c(1, 3))
for (v in x_vars_var) {
  dy <- diff(train_daily$target_wti)
  dx <- diff(train_daily[[v]])
  ccf(dy, dx, main = paste("CCF: d_WTI vs d_", v), xlab = "Lag (Trading Days)")
}
par(mfrow = c(1, 1))
lag_selection <- VARselect(Y_train, lag.max = 10, type = "const")
lag_selection$selection
p <- as.integer(lag_selection$selection["SC(n)"])
johansen <- ca.jo(Y_train, type = "trace", ecdet = "const", K = max(2, p), spec = "transitory")
summary(johansen)


vecm_sets <- list(
  WTI_Brent = c(
    "target_wti",
    "brent_spot"
  ),
  WTI_Brent_Gasoline = c(
    "target_wti",
    "brent_spot",
    "ny_gasoline_spot"
  ),
  WTI_Brent_Heating = c(
    "target_wti",
    "brent_spot",
    "ny_heating_oil_spot"
  ),
  WTI_All4 = c(
    "target_wti",
    "brent_spot",
    "ny_gasoline_spot",
    "ny_heating_oil_spot"
  )
)
lag_results_vecm <- list()
johansen_results_vecm <- list()
fits_vecm <- list()
fits_vec2var <- list()
results_vecm <- data.frame()
for (name in names(vecm_sets)) {
  Y <- as.matrix(train_daily[, vecm_sets[[name]], drop = FALSE])
  k <- ncol(Y)
  lag_selection <- VARselect(Y, lag.max = 10, type = "const")
  lag_results_vecm[[name]] <- lag_selection
  K <- as.integer(lag_selection$selection["SC(n)"])
  K <- max(2, K)
  johansen <- ca.jo(Y, type = "trace", ecdet = "const", K = K, spec = "transitory")
  johansen_results_vecm[[name]] <- johansen
  teststat <- rev(johansen@teststat)
  critical_5 <- rev(johansen@cval[, "5pct"])
  r <- 0
  for (i in seq_along(teststat)) {
    if (teststat[i] > critical_5[i]) {r <- r + 1} else {break}
  }
  if (r > 0 && r < k) {
    vecm_fit <- cajorls(johansen, r = r)
    fits_vecm[[name]] <- vecm_fit
    var_fit <- vec2var(johansen, r = r)
    fits_vec2var[[name]] <- var_fit
    serial_test <- serial.test(var_fit, lags.pt = 20, type = "PT.adjusted")
    serial_p <- serial_test$serial$p.value
    AIC_value <- NA_real_
    BIC_value <- NA_real_
  } else {
    fits_vecm[[name]] <- NULL
    fits_vec2var[[name]] <- NULL
    serial_p <- NA_real_
  }
  results_vecm <- rbind(
    results_vecm,
    data.frame(
      Model = name,
      Variables = k,
      K = K,      # model order
      Rank = r,
      Serial_pvalue = serial_p
    )
  )
}
results_vecm


vecm_candidates <- list(
  WTI_Brent_K4_R1 = list(
    vars = c("target_wti", "brent_spot"),
    K = 4, r = 1
  ),
  WTI_Brent_K5_R1 = list(
    vars = c("target_wti", "brent_spot"),
    K = 5, r = 1
  ),
  WTI_Brent_Gasoline_K3_R2 = list(
    vars = c("target_wti", "brent_spot", "ny_gasoline_spot"),
    K = 3, r = 2
  ),
  WTI_Brent_Gasoline_K4_R2 = list(
    vars = c("target_wti", "brent_spot", "ny_gasoline_spot"),
    K = 4, r = 2
  ),
  WTI_Brent_Heating_K4_R2 = list(
    vars = c("target_wti", "brent_spot", "ny_heating_oil_spot"),
    K = 4, r = 2
  ),
  WTI_Brent_Heating_K5_R2 = list(
    vars = c("target_wti", "brent_spot", "ny_heating_oil_spot"),
    K = 5, r = 2
  ),
  WTI_All4_K3_R3 = list(
    vars = c("target_wti", "brent_spot",
             "ny_gasoline_spot", "ny_heating_oil_spot"),
    K = 3, r = 3
  ),
  WTI_All4_K4_R3 = list(
    vars = c("target_wti", "brent_spot",
             "ny_gasoline_spot", "ny_heating_oil_spot"),
    K = 4, r = 3
  )
)
data_val <- daily_spot
y <- data_val$target_wti
n_train <- sum(daily_spot$date <= M1_PROTOCOL$train_end)
n_valid <- nrow(valid_daily)
H <- max(M1_PROTOCOL$horizons)
origins <- n_train:(n_train + n_valid - H)
actual <- t(vapply(origins, function(t) y[(t + 1):(t + H)], numeric(H)))
forecast_vecm <- function(t, vars, K, r, H) {
  visible <- m1_observable_prefix(data_val, data_val$date[t])
  bridge <- t - visible
  Y_history <- as.matrix(data_val[seq_len(visible), vars, drop = FALSE])
  johansen <- ca.jo(Y_history, type = "trace", ecdet = "const", K = K, spec = "transitory")
  fit <- vec2var(johansen, r = r)
  fc <- predict(fit, n.ahead = H + bridge, ci = 0.95)
  as.numeric(fc$fcst[["target_wti"]][, "fcst"])[bridge + seq_len(H)]
}
predictions_vecm <- list()
predictions_vecm[["RW"]] <- matrix(data_val$origin_price[origins], nrow = length(origins), ncol = H)
for (name in names(vecm_candidates)) {
  predictions_vecm[[name]] <- matrix(NA_real_,nrow = length(origins),ncol = H)
}
for (name in names(vecm_candidates)) {
  cat("\nRunning:", name, "\n")
  spec <- vecm_candidates[[name]]
  for (i in seq_along(origins)) {
    t <- origins[i]
    if (i %% 50 == 0) {
      cat("Origin:", i, "/", length(origins), "\n")
    }
    fc <- tryCatch(forecast_vecm(t = t,vars = spec$vars,K = spec$K,r = spec$r,H = H),
                   error = function(e) {message(name, " origin ", i, ": ",conditionMessage(e)) 
                     NULL}
    )
    if (is.null(fc)) next
    predictions_vecm[[name]][i, ] <- fc
  }
}
fit_status_vecm <- data.frame(
  Model = names(predictions_vecm),
  Failed = sapply(predictions_vecm, function(x) {sum(!complete.cases(x))})
)
fit_status_vecm
common_ok <- Reduce(`&`,lapply(predictions_vecm, complete.cases))
cat("Common valid origins:", sum(common_ok), "/", length(origins), "\n")
if (!any(common_ok)) {stop("No common valid forecast origins")}
actual_valid <- actual[common_ok, , drop = FALSE]
point_results_vecm <- bind_rows(lapply(names(predictions_vecm), function(name) {
  pred <- predictions_vecm[[name]][common_ok, , drop = FALSE]
  err <- actual_valid - pred
  data.frame(
    Model = name,
    Horizon = 1:H,
    RMSE = sqrt(colMeans(err^2)),
    MAE = colMeans(abs(err)),
    MAPE = colMeans(abs(err / actual_valid)) * 100
  )
})
)
rw_rmse_vecm <- point_results_vecm %>% filter(Model == "RW") %>% arrange(Horizon) %>% pull(RMSE)
rw_mape_vecm <- point_results_vecm %>% filter(Model == "RW") %>% arrange(Horizon) %>% pull(MAPE)
rw_mae_vecm <- point_results_vecm %>% filter(Model == "RW") %>% arrange(Horizon) %>% pull(MAE)
point_results_vecm <- point_results_vecm %>%
  mutate(Relative_RMSE = RMSE / rw_rmse_vecm[Horizon],
         Relative_MAPE = MAPE / rw_mape_vecm[Horizon],
         Relative_MAE = MAE / rw_mae_vecm[Horizon])
group_results_vecm <- point_results_vecm %>%
  mutate(Group = case_when(Horizon <= 5 ~ "Short (1-5)",Horizon <= 10 ~ "Medium (6-10)",TRUE ~ "Long (11-60)")) %>%
  group_by(Model, Group) %>%
  summarise(
    Mean_MAPE = mean(MAPE),
    Mean_RMSE = mean(RMSE),
    Mean_MAE = mean(MAE),
    .groups = "drop"
  ) %>%
  arrange(Group, Mean_MAPE, Mean_RMSE, Mean_MAE)
group_results_vecm
overall_results_vecm <- point_results_vecm %>%
  group_by(Model) %>%
  summarise(
    Mean_MAPE = mean(MAPE),
    Mean_RMSE = mean(RMSE),
    Mean_MAE = mean(MAE),
    .groups = "drop"
  ) %>%
  arrange(Mean_MAPE, Mean_RMSE, Mean_MAE)
overall_results_vecm
ggplot(point_results_vecm, aes(Horizon, RMSE, color = Model)) +
  geom_line(linewidth = 0.8) +
  labs(title = "VECM: Expanding-Window Validation", subtitle = "WTI Price Forecasting (1-60 Trading Days)",
       x = "Forecast Horizon", y = "RMSE (USD/barrel)") +
  theme_minimal(base_size = 13)
ggplot(point_results_vecm, aes(Horizon, MAPE, color = Model)) +
  geom_line(linewidth = 0.8) +
  labs(title = "VECM: Expanding-Window Validation", subtitle = "WTI Price Forecasting (1-60 Trading Days)",
       x = "Forecast Horizon", y = "MAPE %") +
  theme_minimal(base_size = 13)
ggplot(point_results_vecm, aes(Horizon, MAE, color = Model)) +
  geom_line(linewidth = 0.8) +
  labs(title = "VECM: Expanding-Window Validation", subtitle = "WTI Price Forecasting (1-60 Trading Days)",
       x = "Forecast Horizon", y = "MAE (USD/barrel)") +
  theme_minimal(base_size = 13)
ggplot(point_results_vecm, aes(Horizon, Relative_RMSE, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_line(linewidth = 0.8) +
  labs(title = "VECM: Relative RMSE vs RW", x = "Forecast Horizon", y = "RMSE / RW RMSE") +
  theme_minimal(base_size = 13)
ggplot(point_results_vecm, aes(Horizon, Relative_MAPE, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_line(linewidth = 0.8) +
  labs(title = "VECM: Relative MAPE vs RW", x = "Forecast Horizon", y = "MAPE / RW MAPE") +
  theme_minimal(base_size = 13)
ggplot(point_results_vecm, aes(Horizon, Relative_MAE, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_line(linewidth = 0.8) +
  labs(title = "VECM: Relative MAE vs RW", x = "Forecast Horizon", y = "MAE / RW MAE") +
  theme_minimal(base_size = 13)


vecm_vars <- c("target_wti", "brent_spot", "ny_gasoline_spot")
K_final <- 3
r_final <- 2
H <- max(M1_PROTOCOL$horizons)
data_tv <- daily_spot[seq_len(m1_observable_prefix(daily_spot, M1_PROTOCOL$fit_end)), , drop = FALSE]
Y_tv <- as.matrix(data_tv[, vecm_vars, drop = FALSE])
final_johansen_vecm <- ca.jo(Y_tv, type = "trace", ecdet = "const", K = K_final, spec = "transitory")
summary(final_johansen_vecm)
final_fit_vecm <- cajorls(final_johansen_vecm, r = r_final)
summary(final_fit_vecm$rlm) 
print(round(coef(final_fit_vecm$rlm), 4)) # fitted coefficients
final_fit_vecm$beta
print(round(final_fit_vecm$beta, 4))


final_fit_vec2var <- vec2var(final_johansen_vecm, r = r_final)
res_vecm <- residuals(final_fit_vec2var)
serial.test(final_fit_vec2var, lags.pt = 20, type = "PT.adjusted") 
arch.test(final_fit_vec2var, lags.multi = 10, multivariate.only = TRUE)
acf(res_vecm[, "resids of target_wti"], main ="ACF: target_wti")


names(predictions_arima)
names(predictions_garch)
names(predictions_arimax)
names(predictions_vecm)


H <- max(M1_PROTOCOL$horizons)
selected_predictions <- list(
  RW = predictions_arima[["RW"]],
  ARIMA = predictions_arima[["ARIMA011"]],
  ARIMA_GARCH = predictions_garch[["GARCH_ARIMA011"]]$point,
  ARIMAX = predictions_arimax[["SP500_ARIMA011"]],
  VECM = predictions_vecm[["WTI_Brent_Gasoline_K3_R2"]]
)
data_compare <- df
data_vecm_compare <- data_val
n_train <- sum(df$date <= M1_PROTOCOL$train_end)
origins_univariate <- seq(from = n_train, to = nrow(data_compare) - H, by = 1)
n_train_vecm <- sum(daily_spot$date <= M1_PROTOCOL$train_end)
origins_vecm <- seq(from = n_train_vecm, to = nrow(data_vecm_compare) - H, by = 1)
dates_univariate <- as.Date(data_compare$date[origins_univariate])
dates_vecm <- as.Date(data_vecm_compare$date[origins_vecm])
common_dates <- dates_univariate[dates_univariate %in% dates_vecm]
idx_uni <- match(common_dates, dates_univariate)
idx_vecm <- match(common_dates, dates_vecm)
selected_predictions <- list(
  RW = selected_predictions$RW[idx_uni, , drop = FALSE],
  ARIMA = selected_predictions$ARIMA[idx_uni, , drop = FALSE],
  ARIMA_GARCH = selected_predictions$ARIMA_GARCH[idx_uni, , drop = FALSE],
  ARIMAX = selected_predictions$ARIMAX[idx_uni, , drop = FALSE],
  VECM = selected_predictions$VECM[idx_vecm, , drop = FALSE]
)
y_compare <- data_compare$target_wti
actual_compare <- t(vapply(origins_univariate[idx_uni],
  function(t) {
    y_compare[(t + 1):(t + H)]
  }, numeric(H)))
comparison_results <- bind_rows(
  lapply(names(selected_predictions), function(name) {
    pred <- selected_predictions[[name]]
    err <- actual_compare - pred
    data.frame(
      Model = name,
      Horizon = 1:H,
      N = colSums(is.finite(err)),
      RMSE = sqrt(colMeans(err^2)),
      MAPE = 100 * colMeans(abs(err / actual_compare))
    )
  })
)
rw_results <- comparison_results %>% filter(Model == "RW") %>%
  dplyr::select(
    Horizon,
    RW_RMSE = RMSE,
    RW_MAPE = MAPE)
comparison_results <- comparison_results %>% left_join(rw_results, by = "Horizon") %>%
  mutate(
    Relative_RMSE = RMSE / RW_RMSE,
    Relative_MAPE = MAPE / RW_MAPE
  )
comparison_results
model_colors <- c(
  RW = "black",
  ARIMA = "#E69F00",
  ARIMA_GARCH = "#CC79A7",
  ARIMAX = "#0072B2",
  VECM = "#009E73"
)
comparison_results$Model <- factor(comparison_results$Model, levels = names(model_colors))
p_rmse <- ggplot(comparison_results, aes(Horizon, Relative_RMSE, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_line(linewidth = 1.1) +
  scale_color_manual(values = model_colors) +
  scale_x_continuous(breaks = M1_PROTOCOL$horizons) +
  labs(title = "Selected Models: Relative RMSE", x = "Forecast Horizon (Trading Days)", y = "RMSE / RW RMSE") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom")
p_mape <- ggplot(comparison_results, aes(Horizon, Relative_MAPE, color = Model)) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  geom_line(linewidth = 1.1) +
  scale_color_manual(values = model_colors) +
  scale_x_continuous(breaks = M1_PROTOCOL$horizons) +
  labs(title = "Selected Models: Relative MAPE", x = "Forecast Horizon (Trading Days)", y = "MAPE / RW MAPE") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom")
print(p_rmse)
print(p_mape)

# Dated historical validation outputs (never final-test results).
validation_output <- file.path(.m1_root, "m1_time_series", "outputs", "frozen_october_2026", "validation_2023-09-01_2026-09-30")
dir.create(validation_output, recursive = TRUE, showWarnings = FALSE)
validation_forecasts <- bind_rows(lapply(names(selected_predictions), function(method) {
  bind_rows(lapply(seq_along(common_dates), function(i) {
    origin <- origins_univariate[idx_uni[i]]
    data.frame(mode = "historical_expanding_refit", origin_date = common_dates[i],
               target_date = data_compare$date[origin + seq_len(H)], horizon = seq_len(H),
               method = method, forecast = selected_predictions[[method]][i, ],
               actual = actual_compare[i, ])
  }))
}))
write.csv(validation_forecasts, file.path(validation_output, "forecasts.csv"), row.names = FALSE)
write.csv(comparison_results, file.path(validation_output, "metrics_all_horizons.csv"), row.names = FALSE)
write.csv(comparison_results %>% filter(Horizon %in% M1_PROTOCOL$horizons),
          file.path(validation_output, "metrics_key_horizons.csv"), row.names = FALSE)

# Rebuild the quoted-row spot table from either the project/module directory.
.m1_script <- grep("^--file=", commandArgs(), value = TRUE)
.m1_candidates <- if (length(.m1_script)) dirname(dirname(normalizePath(sub("^--file=", "", .m1_script[1L])))) else c(getwd(), dirname(getwd()))
.m1_root <- .m1_candidates[vapply(.m1_candidates, function(x) file.exists(file.path(x, "m1_time_series", "daily_spot_df.R")), logical(1))][1L]
if (is.na(.m1_root)) stop("Run from the project root or m1_time_series directory")

library(dplyr)
library(tidyr)

df <- read.csv(file.path(.m1_root, "data", "processed", "wti_daily.csv"))
df$date<- as.Date(df$date)
df <- df %>% arrange(date)
colSums(is.na(df))  # no NAs

brent <- read.csv(file.path(.m1_root, "data", "time_series", "raw", "brent_spot_daily.csv"))
colSums(is.na(brent)) # 1173 NAs -> fixed
brent_price <- brent %>% filter(as.Date(observation_date) >= as.Date("2011-08-04"),
                                as.Date(observation_date) <= as.Date("2026-10-23")) %>% 
  mutate(observation_date = as.Date(observation_date)) %>%
  arrange(observation_date) %>%
  fill(DCOILBRENTEU, .direction = "down")
colSums(is.na(brent_price))

ny_gasoline <- read.csv(file.path(.m1_root, "data", "time_series", "raw", "ny_conventional_gasoline_spot_daily.csv"))
colSums(is.na(ny_gasoline))   # no NAs
ny_gasoline_spot <- ny_gasoline %>% filter(as.Date(observation_date) >= as.Date("2011-08-04"),
                                           as.Date(observation_date) <= as.Date("2026-10-23")) %>% 
  select(observation_date, value) %>%
  mutate(value = value * 42) %>%
  mutate(observation_date = as.Date(observation_date)) %>%
  arrange(observation_date)

ny_heating_oil <- read.csv(file.path(.m1_root, "data", "time_series", "raw", "ny_heating_oil_spot_daily.csv"))
colSums(is.na(ny_heating_oil)) # no NAs
ny_heating_oil_spot <- ny_heating_oil %>% filter(as.Date(observation_date) >= as.Date("2011-08-04"),
                                           as.Date(observation_date) <= as.Date("2026-10-23")) %>%
  select(observation_date, value) %>%
  mutate(value = value * 42) %>%
  mutate(observation_date = as.Date(observation_date)) %>%
  arrange(observation_date)

daily_spot <- df %>% select(date, target_wti, breakeven_inflation_10y) %>%
  left_join(brent_price, by=c("date"="observation_date")) %>%
  left_join(ny_gasoline_spot, by=c("date"="observation_date")) %>%
  left_join(ny_heating_oil_spot, by=c("date"="observation_date"))
colnames(daily_spot) <- c("date", "target_wti", "breakeven_inflation_10y", "brent_spot",
                          "ny_gasoline_spot", "ny_heating_oil_spot")
colSums(is.na(daily_spot))   # 11 NAs for gasoline spot and heating oil spot
daily_spot <- daily_spot %>% 
  fill(brent_spot, ny_heating_oil_spot, ny_gasoline_spot, .direction = "down")
colSums(is.na(daily_spot)) 
write.csv(daily_spot, file.path(.m1_root, "data", "time_series", "daily_spot.csv"), row.names = FALSE)







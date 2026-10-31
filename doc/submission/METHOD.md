# Method and evaluation

## One daily forecast system

Oilracle combines the following modules into price, distribution and risk forecasts at 1, 5, 20 and 60 quoted-day horizons:

| Module | Contribution |
| --- | --- |
| Time-series regression | A three-price WTI/Brent/gasoline VECM with level-VAR lag order 3 and cointegration rank 2 |
| Seasonality | Robust monthly STL; the daily bridge adds the change in seasonal components to the observable spot anchor |
| LSTM | A single LSTM layer reads 60 quoted days. Its final 32-unit hidden state feeds a linear head with 60 anchor-relative price differences. Training uses 20% dropout and 15 epochs, and forecasts average three seeds. |
| ML point models | Feature-based residual forecasts using market, fundamental and member-model information |
| Distribution and risk models | Conditional uncertainty, calibrated quantiles, interval bounds and tail-risk measures |

Futures and basis information also enter the integration. Rules differ by horizon: a conservative residual combination at one day, a regularized static combination using 126 mature errors at five days, market-state-dependent reliability at 20 days, and a regularized combination using 252 errors at 60 days. There is no minimum component weight. Member signals can enter through intermediate ensembles and features; inclusion alone does not establish each module's independent accuracy contribution.

The final accepted adjustment recalibrates five-day tails using previously published forecast errors, preserves the point median and ordered quantiles, and caps interval-width expansion at 35%. Point forecasts are unchanged by this adjustment. Other horizon distributions are retained.

## Selection and evaluation

Initial training uses information through 31 August 2023. The development window runs from September 2023 through September 2026. Historical forward fits use only information and matured forecast errors available at each origin. The final fusion's reported outer point-validation origins begin 2 September 2025; target dates end 18 September 2026 because labels must be available by the September cutoff.

Model choices, combination rules and calibration are selected from pre-October history. Final fitting uses information available by 30 September 2026. October rolling forecasts update inputs while fitted parameters, scalers and calibration remain frozen; a conditional weighting map may produce different weights from new inputs without being refitted.

Evaluation covers target dates 1–23 October 2026. This interval contains 17 quoted days, giving 17 realized one-day and 13 five-day targets. No 20- or 60-quoted-day forecast issued from the final cutoff is realized within that window. Forecasts are evaluated by target date, not as a single fixed-origin path.

## Results

RMSE is in USD per barrel. The random-walk benchmark holds the latest spot quote published at the forecast origin.

| Sample | Horizon | n | Oilracle RMSE | Published-spot RW RMSE |
| --- | ---: | ---: | ---: | ---: |
| Historical development | 1 | 261 | 5.1196 | 7.9918 |
| Historical development | 5 | 257 | 7.8418 | 9.7375 |
| Historical development | 20 | 242 | 13.9357 | 14.8045 |
| Historical development | 60 | 202 | 23.4549 | 25.1294 |
| October evaluation | 1 | 17 | 6.6272 | 9.1101 |
| October evaluation | 5 | 13 | 10.0658 | 11.2532 |

Historical gains over the stronger futures-price-hold baseline are 1.14% and 0.26% at one and five days. Twenty-day RMSE is 2.49% worse than futures hold; 60-day RMSE is 3.50% better. Spot-RW improvements do not imply dominance over every benchmark or standalone module.

The standalone monthly STL charts use a longer first-stage validation window, September 2023 through 18 September 2026, with the August-2023 fit held fixed. Against the same published-spot RW anchor, RMSE falls by 0.26%, 0.14%, 2.21% and 6.71% at 1, 5, 20 and 60 quoted days, on 758, 754, 739 and 699 targets. All pointwise 95% paired moving-block bootstrap intervals include zero. The calculation uses 5,000 draws with blocks of 20, 20, 20 and 60 chronological origins and does not account for method-selection uncertainty. These are standalone development-validation results, distinct from the common fusion sample above.

Risk calibration uses a separate matched June–September sample. Five-day nominal 90% interval coverage improves from 65.28% to 79.17% across 72 targets, with 35% wider intervals and 0.87% lower CRPS. One-day coverage remains 93.42% across 76 targets. October coverage is 94.12% at one day and 100% at five days, based on only 17 and 13 targets respectively.

## Output interpretation

`forecasts.csv` contains 68 rows: 17 daily origins times four horizons. Key fields are:

- `origin_date`, `target_date`, `horizon`: issue date, outcome date and quoted-day horizon. Unobserved future target dates or outcomes may be blank.
- `origin_price`, `rw_price`: the observable spot anchor and hold benchmark.
- `forecast`, `actual`, `lower90`, `upper90`: prices in USD per barrel.
- `center`, `q_*`, `t_*`: log returns relative to `origin_price`; convert a quantile to price with `origin_price * exp(quantile)`.
- `prob_up`, `prob_drop_5pct`, `prob_rise_5pct`: probabilities on a 0–1 scale.
- `var_loss_log_*`, `es_loss_log_*`: value-at-risk and expected-shortfall measures in log-loss units, not dollars.

RMSE and MAE assess point errors. CRPS assesses the predictive distribution, with lower values preferred. Coverage must be read together with interval width; wider ranges can cover more outcomes without improving point accuracy.

Historical results are development validation, not an untouched test. October was viewed during earlier research and is retrospective evaluation, although it did not select the final rules. Its short, overlapping target windows do not establish future coverage, statistical dominance or trading profitability. Publication assumptions and incomplete historical vintages further limit real-time claims.

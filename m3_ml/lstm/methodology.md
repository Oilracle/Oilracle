> **SUPERSEDED HISTORICAL METHODOLOGY.** The experiment described below is retained
> for provenance, not as the current protocol. See `START_HERE.md` for initial
> training through 2023-08-31, historical validation through 2026-09-30, a genuine
> final frozen fit and separate fixed-origin/daily-update October 1–23 forecasts.
> New outputs belong to `outputs/frozen_october_2026/`. The current implementation
> purges all 60 labels using publication availability, not a fixed ten-row buffer.
> Direct multi-output paths need no future actual exogenous inputs. Historical
> gains and counts below must never be presented as the new experiment's results.

# Technical methodology and code guide

## 1. Scope and relationship to the project brief

The relevant assignment is the oil forecasting alternative on PDF pages 8-9 (printed slide numbers 32-33). It asks for daily crude oil price and/or return forecasts for 1-60 days, out-of-sample comparisons, multiple accuracy measures, robustness, data documentation and automatic forecast updates. Economic-value analysis is encouraged, rather than required. The presentation instructions on PDF page 10 specify eight minutes plus two minutes of Q&A per group, participation by every member, and a zipped submission containing editable slides, data and code by 31 October 2026. Check Canvas for any changes or additional instructions.

This contribution covers the deep-learning component independently. The question is: **does a sequence model of already published energy and financial variables improve WTI forecasts over an available-information persistence benchmark?** Other teammates can compare their ARIMA/VAR/tree models on the exported date pairs, or use these LSTM predictions in an ensemble with weights chosen on validation data. No teammate outputs feed this LSTM.

## 2. Data audit and provenance

The source is the supplied `wti_daily.csv` plus its supplied `features.csv` dictionary. The data has 3,790 rows, from 2011-08-04 through 2026-10-06, 32 predictors, one date and one target. There are no missing cells or duplicate dates. Dates are already increasing and remain in that order. All predictors must be finite; the script raises an error rather than silently backfilling or deleting rows. The original input CSVs are copied unchanged under `data/`.

The target is nominal WTI Cushing spot price, USD/barrel, identified in the dictionary as FRED DCOILWTICO (underlying EIA source). Predictors include Brent and refined-product prices, inventories, imports/exports, production and demand proxies, plus rates, equity prices, implied volatility and the dollar index. The full variable-level units, source IDs and release assumptions remain in `data/processed/features.csv`; that file is part of the submission.

The input SHA-256 is `8431640f4cf19f67aed661dce306103cc69f5d08d7a32143ad5f43152e4913eb`. This identifies the exact CSV used, rather than proving correctness. The GitHub repository URL, commit, raw downloads, release timestamps and preprocessing code were not supplied. Availability claims are inherited from the teammate's dictionary and cannot be independently certified from these two files. FRED states data are subject to revision. This is a chronological backtest conditional on the processed alignment, not a fully audited real-time vintage study.

One observation is nonpositive; the minimum is -36.98. Keep it. We do not log-transform prices, calculate log returns across zero, or replace the observation. Prices and changes remain meaningful in dollars even when negative. MAPE is supplementary and explicitly restricted to actual prices greater than USD 1; its eligible sample count is exported. MAE/RMSE over all eligible observations are the headline metrics. The current test sample is positive, but future data and training need not be.

## 3. Information set and the forecast clock

The dictionary defines row t as the trading day being forecast, with every feature public before 00:00 New York time that day. Weekly EIA observations have already been aligned using a report-release-plus-one-federal-business-day rule. Financial variables use two federal business days under the teammate's assumptions. Repeating a weekly value across daily rows represents the information last available, not fresh daily measurements.

The implementation trusts this alignment and does not shift the predictors again. It excludes `target_wti` completely from input features, including lagged versions. In particular, it does not assume y[t-1] is published just because its economic date precedes t. The anchor a[t] is the predictor `wti_spot_last_published`, which is already available on row t. It can be stale.

For horizon h in 1,...,60, the target index is **t+h-1**. A 60-row input window runs from t-59 through t. Horizon 1 is the upcoming price on the origin date, viewed before that trading day begins; horizon 60 is 59 subsequent observations later. The horizons count observed trading rows, not calendar days. Missing holidays are not invented, and Friday's h=2 means the next observed date. This convention needs explicit agreement across the group. An after-close origin convention would require shifting both origins and labels consistently.

For example, the last supplied row is 2026-10-06 with an available published WTI anchor of 85.23 and an unavailable-at-origin target of 96.24. The model may read 85.23 and the other predictors, but not 96.24. Its horizon-1 forecast is 84.37, and that error remains in the evaluation.

## 4. Chronological split and label purging

There is no random train/test split. Nominal training ends 31 December 2021; validation origins start 3 January 2022; test origins start 2 January 2024. A conservative ten-observation release buffer is withheld before both split boundaries. This is a stated implementation assumption, not a verified release schedule for every historical observation.

During model selection, the last eligible training target date is 2021-12-16. Since each sample has 60 labels, the last training origin is 2021-09-22. There are 2,474 common training origins across all candidates. Validation origins run from 2022-01-03 through 2023-09-19, 430 full paths, with last labels on 2023-12-14. A validation origin near December 2023 cannot be used if its 60-day label path crosses into 2024. Purging prevents those test labels entering validation selection.

After configuration and training duration are fixed, the model is reinitialized and refit on all 2,973 pre-test complete-path origins. Final refit labels end 2023-12-14; final weights never consume a 2024-2026 target. This is a frozen-model rolling-origin backtest: each test date supplies a new observed feature window, while model weights/scalers remain fixed. It is not an expanding-window retraining experiment. Later test features are permitted because they would be observed when that later forecast is issued.

Supervised origins stop 59 rows before the label cutoff. Feature scalers are fitted only through the last eligible fit origin; label scalers use only eligible fit samples. This is conservative: some late pre-test features are unused in scaler estimation. Validation/test distributions never determine scaling.

## 5. Forecast target and preprocessing equations

Let X[t] be the 32-vector of available predictors, y[t] the WTI target, and a[t] the last published quote. First standardize each predictor using pre-test fitted means and standard deviations: z[t,j] = (X[t,j] - mean[j]) / sd[j]. A zero standard deviation is replaced by 1 to avoid division by zero. No whole-sample normalization, interpolation or backward filling is used.

The supervised target is the 60-vector d[t,h] = y[t+h-1] - a[t]. Each horizon has its own training mean m[h] and scale s[h]. The network learns q[t,h] = (d[t,h] - m[h]) / s[h]. The output price is reconstructed as yhat[t,h] = a[t] + m[h] + s[h] * qhat[t,h]. This residual formulation lets persistence correspond to a zero dollar delta while allowing negative prices. It is a dollar change from a stale available quote, not an ordinary close-to-close return.

The objective is the mean squared standardized residual error, averaged equally over samples and all 60 horizons. Horizon scaling prevents high-variance long horizons from dominating solely because their raw dollar errors are larger. The objective is not a trading-loss function, and optimizing it need not minimize every raw-dollar MAE or directional error.

## 6. LSTM architecture and intuition

The selected input tensor is [batch, 60 time steps, 32 features]. A one-layer, forward-only LSTM with 32 hidden units processes the sequence. Only the last hidden state enters a dropout layer (0.2) and a linear layer with 60 outputs. There are 10,428 trainable parameters. Hidden and cell states start at zero for every window; they are not carried across batches.

At each step, the forget gate f decides which old memory to retain, input gate i decides how much new information to write, and output gate o decides what memory to reveal. With z as the current standardized feature vector, r as the hidden state, c as cell memory and g as candidate memory:

f = sigmoid(Wf[z,r_prev] + bf); i = sigmoid(Wi[z,r_prev] + bi)

g = tanh(Wg[z,r_prev] + bg); o = sigmoid(Wo[z,r_prev] + bo)

c = f * c_prev + i * g; r = o * tanh(c)

Products are elementwise. These gates can represent nonlinear temporal relationships and persistent information without requiring a fixed AR coefficient per lag. That capability is a modeling motivation, not proof of predictive value. Weekly inputs repeated over a window may add less new information than 60 daily rows suggest.

All 60 horizons are predicted directly from the same known history. The network does not feed a predicted price into tomorrow's inputs or require future macro/inventory values. Direct outputs avoid recursive error propagation but do not impose a smooth or internally coherent future path; some horizon-to-horizon jaggedness is expected.

Dropout is applied after the LSTM. Setting the built-in PyTorch recurrent dropout on a single-layer LSTM would not regularize between recurrent layers; the explicit dropout in `head` avoids that issue. Evaluation uses `model.eval()` to disable dropout.

## 7. Model selection and final fitting

Three small candidates were predefined: lookback/hidden/dropout of (20,16,0.1), (60,16,0.1), and (60,32,0.2). All use learning rate 0.001, one layer, AdamW weight decay 0.001, batch size 128 and gradient clipping at norm 1. A maximum of 80 epochs and patience 12 control training. A validation improvement must exceed 0.00001 in standardized MSE. Common training and validation origins permit a fair comparison despite different lookbacks.

| candidate | lookback | hidden | dropout | lr | best_epoch | validation_mse |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 20 | 16 | 0.100 | 0.001 | 8 | 1.658 |
| 1 | 60 | 16 | 0.100 | 0.001 | 7 | 1.684 |
| 2 | 60 | 32 | 0.200 | 0.001 | 4 | 1.560 |

Candidate 2 was selected at epoch 4. The best validation checkpoint determines the duration; final fitting starts from new initializations and trains for exactly four epochs on combined pre-test data. Three seeds were fixed in advance: 42, 43 and 44. Their price forecasts are averaged equally. Seed-specific errors are retained in `outputs/seed_stability.csv`; the seed with best test performance is never selected. Mini-batch shuffling occurs only among already constructed training windows, after the chronological split. It does not randomize train/test allocation, and independent zero-initialized windows permit it.

The search is deliberately modest because only about 3,000 eligible supervised origins exist and neighboring windows/targets overlap heavily. This is not an exhaustive architecture search. An early selected epoch can indicate rapid overfitting or distribution mismatch; it does not prove that four epochs are universally optimal.

## 8. Benchmarks and ablation

Persistence forecasts a[t] at every horizon. It is a no-change forecast based on the available published quote. It is not the stronger conventional y[t-1] random walk unless yesterday's economic price is actually observable under the same clock. The limitation is explicit, and all models share the same information set.

HistoricalMeanDelta adds the fitted mean residual at each horizon to a[t]. This controls for a simple average correction to the anchor. Ridge uses all 32 standardized current-row predictors and a linear multi-output regression of the same standardized residuals, with an unpenalized intercept. Alpha is chosen among 1, 10, 100, 1,000 and 10,000 using validation standardized MSE; the chosen alpha is 1,000. It is refit on the same pre-test label paths. Ridge can represent linear information in current levels; it lacks the LSTM's full sequence and nonlinear gates.

The published-WTI-only LSTM uses the selected architecture and fixed duration with just the anchor sequence and seed 42. It tests whether complex predictors seem useful descriptively. It is not separately optimized, and its single-seed comparison with a three-seed main ensemble is not a controlled feature-importance experiment. Teammates' ARIMA/VAR/tree models are not reimplemented here; compare them by common exported origins/targets/horizons.

## 9. Evaluation equations and interpretation

MAE = mean(abs(yhat-y)); RMSE = sqrt(mean((yhat-y)^2)). Both use USD/barrel; RMSE penalizes large misses more strongly. Percentage improvement is 100*(1 - model_error/persistence_error). Negative values mean worse performance.

The supplementary MAPE = 100*mean(abs(yhat-y)/y) is restricted to y>1, with `mape_n` reported. This avoids undefined or misleading divisions around zero and negative prices but changes the eligible sample. Do not compare MAPE across models using different filtered rows.

Directional accuracy compares sign(yhat-a) and sign(y-a), excluding actual unchanged prices. Up precision is TP/(TP+FP), up recall is TP/(TP+FN), and F1 is their harmonic mean. When a class denominator is zero, the code reports zero. The neutral forecast share is also exported. Persistence's neutral forecasts are counted as wrong on nonzero actual moves; it should be described as abstaining, not as a 0%-skill direction classifier. This published-anchor measure is clearly distinguished from close-to-close direction.

For each horizon, all models share the same observed date pairs. The test contains 689 horizon-1 origins and 630 horizon-60 origins. Besides horizon-specific full samples, a common-origin table uses 630 complete paths. Predictions on neighboring days and horizons overlap; 630 forecasts are not 630 independent 60-day experiments.

Robustness is summarized by origin year and origin OVX above/below the fitted pre-test median. The bootstrap resamples paired model/benchmark errors in 60-origin blocks and recomputes RMSE improvement. Its ranges are approximate descriptive uncertainty under a dependence-preserving resampling choice. All four selected-horizon intervals include zero; no statistically secure superiority is claimed. No test-based architecture edits were made to improve these reported results.

## 10. Code map and how the pieces connect

`read_data` checks chronological dates and obtains permitted predictors from the dictionary. `boundary` imposes a release buffer. `origins_between` enforces whole-path label cutoffs. `windows` builds causal windows ending at the origin. `targets` builds the 60 dollar deltas. `fit_scaler` estimates only on the supplied fitting sample.

`OilLSTM` defines the network. `train_one` fits it with gradient clipping and optional validation early stopping. `predict` performs batch inference without gradients. `ridge_fit` and `ridge_predict` implement the regularized linear benchmark without an additional machine-learning dependency. `metric_values`, `evaluate` and `bootstrap_ci` generate the comparisons. `make_plots` exports the figures.

`run` orchestrates validation, final refitting, frozen testing and output generation. `forecast_from_checkpoint` loads saved models/scalers and needs only predictor rows. `main` exposes the command-line interface. Test cases cover feature exclusion, split purging, window causality, horizon alignment, negative-price retention, constant scaling and invariance of forecasts when every target is removed.

## 11. Automatic updates and operational boundaries

A new processed row triggers inference through `--forecast-only`. Training is not required for every update. The saved feature list/order, scalers, architecture and seeds are checked/loaded; the final lookback window yields 60 outputs. The current target can be blank or absent. A meaningful update needs genuinely newer features; rerunning unchanged data reproduces the same forecast. There is no persistent network request, recurring scheduler or upstream GitHub integration in this package because no repository/API endpoint was supplied.

The saved model was fitted with labels through 14 December 2023 and is intentionally frozen throughout the supplied historical test. A deployment refit can be added on a chosen cadence after reconstructing a new train/validation cutoff using actually released labels. Such a refit must be evaluated as its own walk-forward experiment. Do not overwrite the original test evidence and then describe it as untouched.

## 12. Limits and defensible next research

The data vintage and upstream alignment are not independently verifiable. Ten observations of boundary buffering are conservative but cannot certify all historical publication delays. The test includes large changes in 2026; this frozen model's gains are concentrated there and do not generalize uniformly to earlier years. The three-seed ensemble measures only initialization variation, not all model uncertainty. No calibrated future-price intervals are provided.

WTI spot quotes are not an executable portfolio. The supplied file has no matched instrument opening/execution prices, futures contract roll history, margin assumptions or cost schedule. Therefore this contribution does not fabricate a Sharpe ratio or P&L from forecast errors. If the group adds economic-value analysis, map time-stamped signals to an actual traded instrument using an agreed execution delay, returns, rolls and costs, then evaluate on a genuinely held-out period. A simple sign(yhat-a) signal alone is not a valid backtest.

Future work can test expanding/rolling refits, a larger but validation-only search, recent target lags when release-time evidence permits them, and ensembles with weights selected before test. An additional untouched period is necessary if these changes are motivated by the current results. Consistent improvements across years, errors and meaningful benchmarks would be stronger evidence than an isolated low RMSE.

## References

1. Supplied QF5210 Part II introduction PDF, oil forecasting brief on PDF pages 8-9 and presentation rules on page 10.
2. Supplied `data/processed/features.csv`: exact variable definitions, units and availability assumptions.
3. PyTorch LSTM documentation: https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.rnn.LSTM.html
4. U.S. EIA, WTI Cushing spot series via FRED DCOILWTICO: https://fred.stlouisfed.org/series/DCOILWTICO
5. Alquist, Kilian and Vigfusson (2011), Forecasting the Price of Oil, Federal Reserve IFDP 1022: https://www.federalreserve.gov/pubs/ifdp/2011/1022/ifdp1022.htm . This provides context for oil forecast benchmarks; its empirical results are not attributed to this project.

# LSTM results: frozen October protocol

Protocol: `qf5210-frozen-october-2026`. These numbers come from the completed
canonical-anchor training/inference run in `outputs/frozen_october_2026/`.
They supersede the old 2024-onward experiment, its h60 improvement claim and its
historical slides/notebook. The parent executed the genuine full training run;
this document transcribes its saved metrics, not a new verification run.

## Main finding

**The LSTM underperforms published-price persistence on the realized daily-update
h1 and h5 samples.** There are no realized October h20/h60 endpoints, so the
numerical long-horizon forecasts cannot support an accuracy claim. The lone
fixed-origin h5 endpoint is favorable to LSTM, but one observation does not
establish reliable superiority or justify selecting this model on October data.

## Accuracy in USD/barrel

| Mode | Method | Horizon | n | MAE | RMSE |
| --- | --- | --- | --- | --- | --- |
| fixed_origin | LSTM | 1 | 1 | 2.901948 | 2.901948 |
| fixed_origin | Persistence | 1 | 1 | 1.670000 | 1.670000 |
| fixed_origin | LSTM | 5 | 1 | 0.294303 | 0.294303 |
| fixed_origin | Persistence | 5 | 1 | 6.420000 | 6.420000 |
| fixed_origin | LSTM / Persistence | 20 | 0 | NA | NA |
| fixed_origin | LSTM / Persistence | 60 | 0 | NA | NA |
| daily_update | LSTM | 1 | 17 | 11.651355 | 12.939878 |
| daily_update | Persistence | 1 | 17 | 8.167059 | 9.110146 |
| daily_update | LSTM | 5 | 13 | 16.331814 | 17.273680 |
| daily_update | Persistence | 5 | 13 | 10.717692 | 11.253163 |
| daily_update | LSTM / Persistence | 20 | 0 | NA | NA |
| daily_update | LSTM / Persistence | 60 | 0 | NA | NA |

Source: `outputs/frozen_october_2026/test_metrics_key_horizons.csv`. Endpoint
counts are not the count of intermediate points on the fixed-origin full path.
All 60 numerical horizons remain in `test_predictions.csv`; unknown future
quotation dates and actuals remain NA. Never pool the two modes.

## Actual fitted model and information set

Historical validation selected candidate 2: a 60-row lookback, 32 hidden units,
0.2 dropout and AdamW learning rate 0.001. The final ensemble averages seeds
42, 43 and 44 trained for **15 fixed epochs**, with no October early stopping.
The saved audit records 2,897 initial-training origins, 699 historical-validation
origins and **3,659 final-fit origins**.

Initial fit uses information available by 2023-08-31; historical validation
covers target dates 2023-09-01 through 2026-09-30, subject to label maturity.
Final supervised paths require all 60 labels available by 2026-09-30. The actual
latest final-fit label is **2026-09-18**, and the latest final training origin is
2026-06-24; the 60-label path purge and publication delay explain this gap.
Weights, feature/label scalers, architecture, epochs and ensemble membership
remain fixed throughout the October 1–23 target window.

The observable WTI channel is reconstructed in memory by the common
publication-evidence as-of policy: use the newest scheduled available quote,
never rewind for an older quote released late. Both residual labels and LSTM
feature windows use this transformation. The original input CSV is unchanged.
Leading rows before the first available in-table quote retain their supplied
already-published quote. The canonical persistence scores now agree with the
other modules' shared benchmark, as reported by the parent's integrated run.

At origin row i the feature window ends at i and output h predicts row i+h.
The direct 60-output architecture requires no future actual exogenous path.
Fixed-origin uses September 30 once; daily-update moves only the causal observed
feature window and published anchor, never refitting coefficients/preprocessing.

## Limitations and reproduction

Publication evidence includes both matched historical WPSR dates and an explicit
retrospective weekly-release assumption for unmatched rows. Other covariate
publication conventions and current historical vintages remain limitations.
Earlier development inspected portions of October; this is a retrospective
protocol cutover, **not a newly untouched prospective experiment**. Seed min/max
is initialization sensitivity, not a calibrated prediction interval.

From the repository root:

```powershell
uv run python m3_ml/lstm/src/oil_lstm.py
uv run python m3_ml/lstm/src/oil_lstm.py --test-only
uv run python m3_ml/lstm/src/oil_lstm.py --forecast-only
uv run python -m unittest discover -s m3_ml/lstm/tests -v
uv run python m3_ml/lstm/build_handoff.py --execute
```

The handoff generator consumes actual approved outputs; it never invents missing
results. Binary slides remain historical until the project's new final deck is
built from the frozen-October metrics.

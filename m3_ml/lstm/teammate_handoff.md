> **SUPERSEDED HANDOFF.** Do not integrate the old `outputs/test_predictions.csv`
> or its old 2024-onward test claims. The canonical new input is
> `outputs/frozen_october_2026/test_predictions.csv`, with key
> `(mode,origin_date,target_date,target_row,horizon,method)` and fields `forecast`,
> `actual`, `origin_price`, `protocol_id`, `fit_end`, `train_end`,
> `validation_start`, `validation_end`. Unknown future dates/actuals remain NA;
> numerical h20/60 forecasts remain present but unscored. Filter h=1,5,20,60 for
> endpoint comparison and never pool modes. See `START_HERE.md` for training,
> publication assumptions and historical-artifact limitations.

# Teammate integration and submission

## Contract for model comparisons

The LSTM requires only the processed predictors, not another model's results. The test is frozen, with origins beginning 2024-01-02 and final observed target date 2026-10-06. Model selection uses data before 2024, with the stricter buffer and full-path purging described in `methodology.md`.

The prediction key is `(origin_date, target_date, horizon, model)`. At an origin before the row's trading day, h=1 targets that row; h=k targets row t+k-1. Horizons count observed rows. All raw results are in USD/barrel. The anchor is the last released WTI quote, not necessarily the previous observation's target. Both dictionary and data are included unchanged.

To compare with another model, select a common horizon and inner-join the exact keys except model. Check the actual target values agree. Compare on the shared rows and report counts. Do not merge by target date alone: multiple origins can predict a given target date. Do not align h=1 from an after-close model with h=1 from this before-midnight model without shifting the origin convention correctly.

```python
import pandas as pd
lstm = pd.read_csv('outputs/test_predictions.csv')
lstm = lstm[lstm.model.eq('LSTM')]
other = pd.read_csv('teammate_predictions.csv')
keys = ['origin_date', 'target_date', 'horizon']
paired = lstm.merge(other, on=keys, how='inner', suffixes=('_lstm', '_other'), validate='one_to_one')
assert (paired.actual_lstm - paired.actual_other).abs().max() < 1e-8
```

This assumes the other file contains one chosen model. Filter it first if it contains multiple models. The group's original train/test arrangement was not supplied; the dates used here are a documented default. If the group requires different dates, rerun all models consistently and regenerate the figures, report and slide values. Do not paste mismatched test results into one comparison table.

## Files to include in the final group ZIP

| File/folder | Purpose |
| --- | --- |
| `data/processed/wti_daily.csv`, `data/processed/features.csv` | Required daily observations and full variable definitions. Paths are from the repository root |
| `src/oil_lstm.py` | Full training, selection, evaluation and forecast code |
| `requirements.txt`, `environment_tested.txt` | Install instructions and tested versions |
| `tests/`, `outputs/verification.log` | Checks for timing and label leakage |
| `Oil_LSTM_Walkthrough.ipynb` | Explanatory notebook with cleared execution output |
| `outputs/test_predictions.csv` | Row-level model predictions for integration |
| `outputs/test_metrics_*.csv` | Tables at every horizon and common origins |
| `outputs/robustness.csv`, `outputs/bootstrap_improvement.csv` | Stability and uncertainty evidence |
| `outputs/models/` | Three checkpoints, fitted scalers and model metadata |
| `outputs/*.png` | Figures for report and discussion |
| `output/pdf/LSTM_technical_report.pdf` | Detailed methodology and results |
| `build_handoff.py` | Regenerates data-dependent documentation/notebook/PDF |

`environment_tested.txt` records relevant package versions for this run. The full machine environment is not bundled. Course presentation files are distributed separately. Model reproduction does not depend on presentation software.

## When results change

Keep the supplied output folder as the original experiment. New split dates, input revisions, altered features or hyperparameters require a new result folder and matching slides. `build_handoff.py` currently reads `outputs/`; update its source path or replace that folder only after preserving the old run. Its narrative is written for the supplied baseline experiment, so review all fixed dates, numbers and interpretations when rebuilding a different experiment. Automatically regenerated tables alone do not validate old prose.

If you want a live forecasting system, connect the teammate's original GitHub preprocessing to the forecast-only CLI and trigger it after each validated data refresh. Historical target vintages and release times need an upstream audit. Do not silently use newer revised data and describe the result as a fully real-time test.

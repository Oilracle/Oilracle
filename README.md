# Oilracle WTI Forecasting

Oilracle forecasts the daily WTI Cushing spot price in USD per barrel, together with predictive distributions and tail risk. It combines time-series regression, seasonality, LSTM, tabular machine learning, and calibrated risk forecasts.

Watch the [one-minute Oilracle pitch](https://youtu.be/HiKISuyWJno) for an introduction to the forecasting system.

This public source release contains the forecasting code, datasets, saved models and research documentation. Course submission slides, registration information and rehearsal materials are distributed separately and are intentionally omitted here.

## Repository layout

| Directory | Purpose |
| --- | --- |
| `data/` | Shared source snapshots, processed tables, long-history research data, and chart extracts |
| `m1_time_series/` | ARIMA, ARIMAX, VECM, and GARCH in R |
| `m2_seasonality/` | Calendar effects, HAC inference, and monthly STL |
| `m3_ml/` | Point forecasts, distributions, risk, LSTM, experimental Kronos, and fusion |
| `m4_reports/` | Final policy manifest, analysis, diagnostics and forecast charts |
| `scripts/` | Data acquisition, preparation, and independent audits |
| `doc/` | English methods, data, results, and reproduction documentation |

## Data and evaluation

The main table, `data/processed/wti_daily.csv` or its Parquet equivalent, contains 3,803 quoted days from 2011-08-04 through 2026-10-23. Source snapshots were refreshed on 2026-10-29. `target_wti` is the outcome and must never be used as a contemporaneous feature. The negative WTI price on 2020-04-20 is retained.

Initial training ends on 2023-08-31. Historical development validation covers 2023-09-01 through 2026-09-30, subject to label publication. Model selection precedes the final fit using information available by 2026-09-30. Parameters and preprocessing are then frozen. October target dates are 2026-10-01 through 2026-10-23.

The final product uses daily rolling forecasts (`daily_update`) at 1, 5, 20, and 60 quoted-day horizons. Inputs update as they become available; coefficients do not. October has 17, 13, 0, and 0 realized targets at these horizons. Fixed-origin paths are retained only as research provenance.

The sole official entry point is [final_strategy_manifest.json](m4_reports/final_strategy_manifest.json), which selects `short_bias_v6`. This internal identifier inherits the v5 point forecasts and upgrades five-day risk intervals. On common June–September historical origins, five-day coverage rises from 65.28% to 79.17%, CRPS improves by 0.87%, and interval width increases by 35%. October point RMSE/RW ratios are 0.7275 and 0.8945 at one and five days. See [the final model report](doc/m3_ml/FUSION_V6.md) for full results and limitations.

Historical validation was used for development. October had already been inspected during research and is a retrospective evaluation, not an untouched blind test. The project makes no trading-return or future-coverage guarantee.

## Reproduction

Start with the [setup and reproduction guide](doc/submission/README.md). It explains how to verify saved forecasts, regenerate metrics and figures, reapply the frozen final fusion, and rebuild the processed data. [Data acquisition](doc/submission/DATA.md#data-acquisition-and-refresh) covers source downloads, EIA API-key setup, output locations and validation. The [retraining guide](doc/submission/RETRAINING.md) documents the two-stage dates, individual module commands, full fusion dependency order, caches and troubleshooting.

> **WARNING — Full retraining is a long-running compute job.** Plan for extended CUDA/GPU training time for GPU-capable neural components. **CPU-only full retraining is strongly discouraged because it is extremely time-consuming.** Treat it as a separate code/resource adaptation and validation task, not a drop-in fallback. Several supplied modules, including the current standalone LSTM, are CPU-native; installing CUDA does not automatically move them to GPU. Read the retraining guide before launching training. Saved-result review, frozen final-fusion inference and data rebuilding do not require CUDA.

Install Git LFS before cloning, then run the following from the repository root to retrieve the three large research tables and install the locked Python environment:

```powershell
git lfs install --local
git lfs pull
uv sync --locked
```

R modules document their package requirements separately. Saved data and model outputs are included, so evaluation does not require API credentials or retraining. File bytes and original line endings are preserved because the frozen-model manifests verify SHA-256 hashes.

```powershell
uv run --locked python -B -m pytest -q m3_ml/tests/test_fusion_v6.py
uv run --locked python -B scripts/audit_submission_data.py
uv run --locked python -B -m m4_reports.plot_final_forecast
uv run --locked python -B -m m4_reports.plot_historical_forecast
```

Run expensive training in a separate clone or output directory to preserve the reference results. Use the saved snapshots to reproduce the reported scores; new provider downloads may revise history and require a deliberate new training run. [The documentation index](doc/README.md) links the data dictionary, publication assumptions, module methods and detailed workflows. Broader optional tests are listed in the reproduction guide.

## Public source release

The research data, model checkpoints and final forecast outputs are retained. Personal presentation materials, member rosters, speaker assignments, registration links and document-production assets are omitted. Notebook execution output has been cleared and two local manifest paths are relative to the repository. Model parameters, forecast values and evaluation metrics are unchanged.

See [Repository file policy](doc/REPOSITORY.md) and [the final forecast entry point](doc/m4_reports/README.md).

## AI assistance acknowledgement

AI agents were used for modelling discussions, code improvement and debugging, data-alignment checks, analysis of model results, and preparation of charts, documentation, slides and the video script. Microsoft text-to-speech was used to generate the video narration. The team takes responsibility for the methods, results, conclusions and accuracy of the submitted work.

# Project documentation

This directory describes the shared data, model implementations, evaluation protocol, and final deliverables. Source code remains in the corresponding module directories.

| Area | Documentation |
| --- | --- |
| Shared data | [Data files and processing](data/README.md), [long-history research data](data/long/README.md) |
| M1 | [Time-series regression](m1_time_series/README.md) |
| M2 | [Seasonality](m2_seasonality/README.md) |
| M3 | [ML and the evaluation protocol](m3_ml/README.md) |
| M3 comparisons | [Aligned comparison](m3_ml/compare/README.md) |
| M3 LSTM | [LSTM handoff](m3_ml/lstm/README.md) |
| M3 residuals | [Residual correction](m3_ml/residual/README.md) |
| Final model | [Official strategy](m3_ml/FINAL_STRATEGY.md), [final results](m3_ml/FUSION_V6.md) |
| M4 | [Final forecast outputs](m4_reports/README.md) |
| Submission checks | [Cleanup, data integrity and package verification](SUBMISSION_CHECKS.md) |
| Repository files | [Public source files and Git LFS](REPOSITORY.md) |
| Reproduction guides | [Setup, saved forecasts and data rebuilding](submission/README.md), [data acquisition](submission/DATA.md#data-acquisition-and-refresh), [retraining and compute warnings](submission/RETRAINING.md), [method](submission/METHOD.md) |

The [repository overview](../README.md) is the starting point. The final product is daily rolling forecasting with frozen September parameters. Earlier fixed-origin studies and model versions are explicitly historical research records, not competing final products.

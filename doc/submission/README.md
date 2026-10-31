# Oilracle: setup and reproduction guide

Oilracle produces daily rolling WTI price forecasts, predictive distributions and tail-risk estimates. The final parameters use information available by 30 September 2026; October outcomes are used for evaluation only.

This public repository contains data, source code, saved models and results. All paths and commands below are relative to the **repository root**, unless stated otherwise. Three large research tables use Git LFS; retrieve their full contents before running the code. See the [Git LFS instructions](../REPOSITORY.md#git-lfs-and-exact-bytes).

## Choose a reproduction task

| Task | Instructions | Fits models? |
| --- | --- | --- |
| Check data and reproduce reported tables and figures | Sections 1-3 below | No |
| Apply the saved final fusion to the saved member forecasts | Section 4 below | No |
| Rebuild processed data from the bundled source snapshots | Section 5 below | No |
| Download or refresh source data in a separate research copy | [Data acquisition instructions](DATA.md#data-acquisition-and-refresh) | No; changes data inputs |
| Rerun an individual module, the final calibration layer or the complete training chain | [Retraining guide](RETRAINING.md) | Yes |

The first route is sufficient to check the reported performance. Retraining is a separate research operation: it can take substantially longer and writes new models and selections. Use a separate clone for retraining so that the submitted reference results remain available for comparison.

> **WARNING — Full retraining is a long-running compute job.** Plan for extended CUDA/GPU training time for GPU-capable neural components. **CPU-only full retraining is strongly discouraged because it is extremely time-consuming.** Treat it as a separate adaptation task requiring code/resource configuration and validation, not a drop-in fallback. Some supplied modules are CPU-native; installing CUDA does not automatically move them to GPU. Read the resource warning in [RETRAINING.md](RETRAINING.md) before launching any training sequence. Saved-result review, frozen final-fusion inference and data rebuilding do not require CUDA.

## 1. Set up the environment

Install Python 3.12 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/). The packaged review commands were checked on Windows with Python 3.14.7. Install Git and the Git LFS executable, then clone the public repository and retrieve its large files:

```powershell
git clone https://github.com/Oilracle/Oilracle.git
cd Oilracle
git lfs install --local
git lfs pull
uv sync --locked
uv run --locked python --version
```

`pyproject.toml` lists direct dependencies and `uv.lock` records their resolved versions. `--locked` prevents an implicit lockfile update; see the [uv sync documentation](https://docs.astral.sh/uv/concepts/projects/sync/). Installation requires internet access. Once dependencies are installed, the review and snapshot-rebuild commands below use the bundled files and do not fetch replacement market data.

The Windows/Linux lock selects CUDA 12.8 PyTorch packages, but the review commands do not require a GPU. R is needed only for the R training/replay steps in the retraining guide. The repository does not contain a virtual environment: `uv sync` creates one locally.

## 2. Audit the inputs and model safeguards

```powershell
uv run --locked python -B scripts/audit_submission_data.py
uv run --locked python -B -m pytest -q m3_ml/tests/test_fusion_v6.py
```

The audit checks source hashes, table consistency, publication alignment and portable file dependencies. The targeted suite contains seven tests covering label availability, target-independent prediction, quantile ordering, width limits and sealed inputs. A successful run exits with code 0 and the test summary reports `7 passed`.

The broader optional suite is:

```powershell
uv run --locked python -B -m pytest -q m3_ml/tests m3_ml/lstm/tests
```

## 3. Recompute the reported metrics and figures

```powershell
uv run --locked python -B -m m4_reports.plot_final_forecast
uv run --locked python -B -m m4_reports.plot_historical_forecast
uv run --locked python -B m4_reports/prepare_seasonality_slide_evidence.py
```

The plot commands check the frozen model dependencies, recompute scores from saved predictions and regenerate October and historical figures. They do not select parameters or fit models. The seasonality command recalculates the standalone first-stage comparisons, paired bootstrap intervals and chart inputs, writing `.local/seasonality_slides/evidence.json`.

`m4_reports/final_strategy_manifest.json` identifies the sole final model and its integrity hash. Its delivered results are under:

```text
m3_ml/fusion_v6/outputs/frozen_october_2026/rolling/
```

| Output, relative to that directory | Contents |
| --- | --- |
| `forecasts.csv` | Daily rolling price forecasts, quantiles and risk estimates |
| `metrics.csv` | October scores and baseline comparisons |
| `historical_point_metrics.csv` | Historical point-forecast scores |
| `historical_risk_metrics.csv` | Historical interval and distribution scores |
| `delivery_scope.json`, `delivery_audit.json` | Evaluation scope and integrity checks |
| `figures/` | Regenerated October figures |
| `history_review/point_forecasts.csv` | Historical forecasts used in the review |
| `history_review/point_metrics.csv`, `history_review/risk_metrics.csv` | Recomputed historical scores |
| `history_review/` | Regenerated historical figures |

The following RMSE values, rounded to four decimals, are useful checks. Units are USD per barrel; RW holds the latest spot quote published at the forecast origin.

| Sample | Quoted-day horizon | Realized targets | Oilracle RMSE | Published-spot RW RMSE |
| --- | ---: | ---: | ---: | ---: |
| Historical point validation | 1 | 261 | 5.1196 | 7.9918 |
| Historical point validation | 5 | 257 | 7.8418 | 9.7375 |
| Historical point validation | 20 | 242 | 13.9357 | 14.8045 |
| Historical point validation | 60 | 202 | 23.4549 | 25.1294 |
| October evaluation | 1 | 17 | 6.6272 | 9.1101 |
| October evaluation | 5 | 13 | 10.0658 | 11.2532 |

There are 68 delivered rows: 17 daily origins times four horizons. October contains no matured 20- or 60-quoted-day targets, so their October scores are unavailable, not zero. Historical risk calibration uses a different matched sample from the point table: 76 one-day and 72 five-day targets. See [Method and evaluation](METHOD.md) for coverage results, benchmark definitions and limitations.

## 4. Reapply the saved final fusion, without training

Run these commands in order if you want to regenerate the final forecasts and delivery tables from the saved upstream forecasts and the saved final model:

```powershell
uv run --locked python -B -m m3_ml.fusion_v6.run test
uv run --locked python -B -m m4_reports.final_strategy
uv run --locked python -B -m m4_reports.short_strategy --audit
uv run --locked python -B -m m4_reports.plot_final_forecast
uv run --locked python -B -m m4_reports.plot_historical_forecast
```

The first command reads the saved preceding fusion forecasts and `models/correction.joblib`, checks the sealed input hashes, and rewrites `test_forecasts.csv`, `test_metrics.csv` and `test_audit.json` in the final fusion output directory. It does **not** regenerate all member-model forecasts. The delivery command rebuilds `rolling/` tables and updates `m4_reports/final_strategy_manifest.json`. The audit verifies the 68-row grid, unchanged point forecasts, ordered quantiles and the five-day interval-width cap. It should print `V6 delivery audit passed` (an internal code identifier).

These commands rewrite generated outputs even though they do not fit models. Keep an untouched checkout and record its commit SHA for comparison. A sealed-input mismatch must be investigated; do not edit hashes to make the audit pass.

## 5. Rebuild data from the included snapshots

Use a separate destination so that canonical inputs and their sealed hashes remain unchanged:

```powershell
uv run --locked python -B scripts/prepare_data.py --output-dir .local/data_check
uv run --locked python -B scripts/validate_prepared.py --output-dir .local/data_check --check-rerun
uv run --locked python -B scripts/audit_submission_data.py --rebuilt .local/data_check
```

The builder uses the included snapshots and rebuild inputs. The validator checks the prepared tables and repeats the build to test deterministic output. The last audit compares the rebuilt core tables with the bundled data and checks the validation report. None of these commands replaces `data/processed/`.

Sources, access notes and release assumptions are listed in [Data](DATA.md). Its [data acquisition instructions](DATA.md#data-acquisition-and-refresh) give the download commands, EIA API-key setup, output paths, validation order and optional news/futures refresh steps. Downloading newer data changes the experiment and is unnecessary for reproducing this submission.

## 6. Find the implementation and training instructions

| Directory | Purpose |
| --- | --- |
| `data/` | Local source snapshots, processed tables and rebuild inputs |
| `scripts/` | Data preparation, validation and audit |
| `m1_time_series/` | Time-series regression and VECM |
| `m2_seasonality/` | Monthly seasonal decomposition and daily bridge |
| `m3_ml/lstm/` | Sequence forecasting |
| Other `m3_ml/` modules | ML point, distribution, risk and fusion models |
| `m4_reports/` | Final-model routing, evaluation and chart code |

The [retraining guide](RETRAINING.md) gives the two-stage dates, R dependencies, module commands, cache behavior, fusion execution order and troubleshooting. Earlier fusion directories are dependencies of the delivered model; the final manifest selects one final product. Directory suffixes are internal identifiers.

## Verification scope

The data audit and seven targeted tests were run from the public checkout for this documentation update. Historical/October figure generation, seasonality evidence calculation, saved final-fusion inference, the delivery audit, snapshot data rebuilding and deterministic rebuild checks were previously executed successfully on the same submitted code/data snapshot using Python 3.14.7. A new dependency installation and the complete model-training chain were not rerun for this update. Retraining commands were checked against the source entry points and dependency order; this is not a claim of a new clean-environment training replication.

## AI assistance acknowledgement

The [repository README](../../README.md#ai-assistance-acknowledgement) describes the use of AI agents in modelling discussions, code improvement and debugging, data-alignment checks, results analysis and preparation of project materials, and the use of Microsoft text-to-speech for video narration. The team takes responsibility for the methods, results, conclusions and accuracy of the submitted work.

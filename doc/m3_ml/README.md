# Machine learning and forecast evaluation

## Final product

The sole official forecast is daily rolling `short_bias_v6`, selected by [the delivery manifest](../../m4_reports/final_strategy_manifest.json). Read [FUSION_V6.md](FUSION_V6.md) for current evidence. Earlier versions remain reproducibility dependencies and development records, not competing final recommendations.

The system combines actual M1 VECM, M2 monthly STL, three-seed LSTM, ML point forecasts, and distribution/risk models. Component inputs can affect forecasts through inherited ensembles and residual-model features; inclusion does not prove an independent accuracy gain. There is no five-percent minimum component weight. The risk layer is centered on and validated against the integrated forecast, rather than attaching another model's intervals without calibration.

All v5 point predictions remain unchanged. The accepted final upgrade only adjusts five-day tails: common-window historical 90% coverage rises from 65.28% to 79.17%, CRPS improves by 0.87%, and width increases by 35%. October h1/h5 RMSE/RW ratios are 0.7275/0.8945 with 17/13 targets. Five-day October coverage rises from 10/13 to 13/13, which is not a future guarantee. The current final output contains 68 rolling forecasts across four horizons. Slides and video are finalized; see [M4](../m4_reports/README.md).

## Modules

| Directory | Purpose |
| --- | --- |
| `common/` | Dates, publication-aware labels, features, preprocessing, validation, metrics, and compute scheduling |
| `tabular/` | Direct Elastic Net, Random Forest, and LightGBM; [results](../../m3_ml/tabular/RESULTS.md) |
| `distribution/` | HS, EWMA, GARCH, EVT, scale forecasts, quantile regression, mixtures, and ensembles |
| `risk/` | Volatility, tail-event probabilities, and economic-value research |
| `spread/` | Brent–WTI and 3-2-1 crack spreads |
| `residual/` | ML correction of an independent ARIMA/calendar baseline; [guide](residual/README.md) |
| `lstm/` | Direct multi-step sequence forecasts; [guide](lstm/README.md) |
| `kronos/` | Experimental self-trained cross-asset OHLCV model; [results](../../m3_ml/kronos/RESULTS.md) |
| `compare/` | Strictly aligned cross-module evaluation; [guide](compare/README.md) |
| `long/` | Long-history data construction; [data](../data/long/README.md) |
| `fusion/`, `fusion_v2/` through `fusion_v6/` | Historical policy development and inherited final-model dependencies |

## Shared protocol

`common/protocol.py` defines `qf5210-frozen-october-2026`.

| Stage | Dates | Use |
| --- | --- | --- |
| Initial training | 2011-08-04–2023-08-31 | Initial fits for historical validation |
| Historical development | 2023-09-01–2026-09-30 | Model/feature/hyperparameter selection using labels available at each historical origin |
| Final fit | Information available by 2026-09-30 | Refit the selected specification, then freeze parameters and fitted preprocessing |
| October evaluation | Target dates 2026-10-01–2026-10-23 | Evaluation only; no tuning or refitting |

Horizons 1/5/20/60 count original quote rows. Do not remove unavailable labels and renumber time. The official product uses `daily_update`: inputs and filtered states change after publication, while fitted mappings, coefficients, preprocessing, and calibration stay frozen. State-dependent weights can change because the input changes; their learned mapping does not. The low-level protocol retains `fixed_origin` for historical audit, excluded from the final rolling presentation.

`wti_publication.csv` checks whether each label can be used. A date having occurred does not imply publication. `origin_price` is the latest observable quote, not necessarily the contemporaneous actual. Log-return models reconstruct prices from this anchor and exclude nonpositive endpoints from log targets. Price, absolute change, volatility, and spread losses are not ranked as if they were the same target.

Outputs under each module's `outputs/frozen_october_2026/` contain protocol ID, mode, origin, horizon, target date, actual, and score status. October has 17/13/0/0 realized targets. Future targets retain numerical forecasts but no invented quote dates or actuals; empty metrics have `n=0`, not zero loss. Calendar scenarios are explicitly labeled. The short October sample is descriptive: no DM/HAC significance or significant-win flags are reported there.

## Publication assumptions and evidence limits

Matched WPSR quotes provide dated archive evidence; other WTI labels assume availability on the fifth US federal business day after the quote week's Friday. Companion spot/spread legs require a declared common-delay assumption. Current historical snapshots, revisions, and incomplete original vintages limit claims of real-time performance.

Historical validation was repeatedly used during development. Previously inspected October data cannot become a new blind holdout merely by changing the protocol. Older cross-year results under module `outputs/` roots belong to earlier splits and are never silently relabeled as frozen October results. Neural checkpoints must correspond to the actual new split.

## News release calendar

The former uniform next-day rule failed because daily observation frequency is not daily publication frequency. AI-GPR only covered September; its short-window features became missing at nine October origins and its 30-day feature at four. This was a source-calendar problem, not a trainer/GPU problem.

The approved `news-release-calendar-v1` applies the same rules to historical validation, final fitting, and October:

| Source | Statistics through | Assumed availability and expiry |
| --- | --- | --- |
| Six AI-GPR series | Previous month-end | Next month's first day; expires at the next scheduled monthly release |
| Keyword GPR | Previous Sunday | Monday, deferred for US federal holidays; expires at the next scheduled weekly release |
| US EPU | Latest existing daily observation | Next day; at most seven calendar days of carry-forward |

The original 7/30/90-day statistics and all 24 news features remain. Raw observations are not interpolated, filled with zeros, or invented. Missing scheduled releases invalidate stale monthly/weekly values. On September 30, AI-GPR uses August-end statistics; October origins use September-end statistics. The October 12 holiday postpones the keyword-GPR update from October 11 statistics until October 13.

The 2026-10-29 refresh retained the unchanged September-end AI-GPR file, extended keyword GPR to October 26 and EPU to October 28, and also revised 30 historical GPR and 25 EPU values. Therefore both news variants were fully reselected and rerun. The assumptions remain distinct from independently verified vintage publication times and do not eliminate retrospective AI-scoring concerns. Official context: [AI-GPR](https://www.matteoiacoviello.com/ai_gpr.html), [method paper](https://www.matteoiacoviello.com/research_files/AI_GPR_PAPER.pdf), and [keyword GPR](https://www.matteoiacoviello.com/gpr.htm).

Each news run records policy ID, hashes, source coverage, and per-origin `data_through`, `available_date`, `age_days`, and completeness. Distribution selection also records the news policy and hashes and rejects incompatible feature snapshots. Refreshing sources is an explicit acquisition action; do not refresh them merely to reproduce frozen results.

```powershell
uv run python -m m3_ml.common.news --audit-output .local/news_release_audit.csv
uv run python -m m3_ml.common.refresh_news --write
```

## Completed base-module runs

The original dependency queue completed steps 1–23. Each news variant ran 437 candidate evaluations, selected 57 model/horizon combinations, and produced 435 forecasts plus 30,424 availability records. Distribution completed 208 final fits and 1,026 forecasts across 12 methods. Risk produced 609 volatility and 2,574 event forecasts; residual models produced 609 point and 522 quantile rows. M1, M2, LSTM, both spreads, and aligned comparison outputs are also complete. The old single-module video step was replaced by integrated presentation data.

The final available WTI quote at the September information cutoff is September 18. Base-module completion is not itself fusion: the actual member replay, fusion selection, risk calibration, and final delivery have their own modules and saved evidence. Use existing results for presentation rather than rerunning the entire exploratory queue.

## Compute and execution

`common/compute.py` controls thread budgets; `common/execution.py` schedules independent tasks and saves atomic caches. Results are collected in grid order to preserve deterministic tie handling. With 16 outer workers, each worker uses one thread, preventing nested oversubscription. No grid, fold, horizon, or Monte Carlo path count is reduced.

| Setting | Default | Contract |
| --- | --- | --- |
| `M3_CPU_THREADS` | `2` | Integer from one to available cores, or `all`; invalid values, including `-1`, fail |
| `M3_OUTER_JOBS` | `1` | Independent workers, no larger than the thread budget; `all` uses that budget |
| `M3_LIGHTGBM_DEVICE` | `cpu` | `cpu` or OpenCL `gpu`; this is separate from PyTorch CUDA |

The installed LightGBM wheel lacks a GPU tree learner. A requested unsupported GPU run fails explicitly rather than silently falling back. LSTM/Kronos can use CUDA. On Windows, `pin_workers.ps1` can distribute the current runner's worker children across allowed cores; it does not change global power settings or other applications.

The serial runner waits for dependencies, stops on nonzero exits, and records status locally. `--list` does not train. Caches require matching data, code, library, resource, and task signatures; comment cleanup may invalidate code signatures for a future rerun but does not alter saved model artifacts.

```powershell
uv run python -m m3_ml.run_serial --list
uv run python -m m3_ml.run_serial --from-step distribution_validate --through-step compare_analyze --cpu-threads all --workers all
uv run python -m m3_ml.tabular.run_direct
uv run python -m m3_ml.distribution.run validate
uv run python -m m3_ml.distribution.run test
uv run python -m m3_ml.risk.volatility validate
uv run python -m m3_ml.risk.volatility test
uv run python -m m3_ml.risk.events validate
uv run python -m m3_ml.risk.events test
uv run python -m m3_ml.spread.run validate --spread brent_wti
uv run python -m m3_ml.spread.run test --spread brent_wti
uv run python -m m3_ml.spread.run validate --spread crack_321
uv run python -m m3_ml.spread.run test --spread crack_321
uv run python -m m3_ml.residual.run
```

Direct prediction accepts `--denoise` and `--news`; distribution accepts optional `--methods`. A subset is not a substitute for a full protocol run. Run dependent module commands sequentially; parallelism belongs inside independent validation tasks. Existing frozen outputs, not local progress logs, are the submitted research evidence.

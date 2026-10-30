# Aligned frozen-parameter comparison

This comparator uses `qf5210-frozen-october-2026`: initial training through 2023-08-31, historical development through September 2026, final information cutoff 2026-09-30, and October 1–23 target dates. This is retrospective evaluation of existing data, not a newly untouched blind test.

## Alignment contract

The comparator keeps `fixed_origin` and `daily_update` separate. The final product presents only daily rolling forecasts. Horizons count original quote rows: `target_row = origin_row + horizon`. Fixed-origin paths cannot consume October features, actuals, or state updates. Daily updates freeze coefficients and preprocessing while updating available inputs.

`observable_price(table)` reconstructs the latest published WTI quote by observation date from `wti_publication.csv`; late old quotes cannot rewind it. It does not substitute the sometimes stale table feature or an unpublished same-day actual. Return quantiles share this anchor. Missing future quote dates and actuals remain missing, while numerical forecasts are retained.

Matched dated archives and assumed fifth-federal-business-day weekly releases are explicitly distinguished. Classical state models use a contiguous published prefix and bridge internal gaps without renumbering time. Companion variables in the Python comparator use its declared common-delay assumptions; WTI evidence does not independently prove publication of other series.

## Implementations

| Entry | Implementation |
| --- | --- |
| `part1.py` ARIMA | Drift-free ARIMA(0,1,1), fitted once for final evaluation; `append(refit=False)` updates state |
| `part1.py` ARIMAX | ARIMA errors with S&P regression; future S&P held at the last available value |
| `part1.py` VECM | Eight original candidates; initial-training fits, historical 1–20-step annual relative-RMSE geometric-mean selection, then one final refit; frozen VAR recursion with intercept `alpha @ const_coint.T` |
| `part1.py` ARIMA-GARCH | Two-stage ARIMA residual GARCH(1,1)-t, 5,000 paths; distinct from R fGarch joint estimation |
| `lstm_aligned.py` | Imports real LSTM frozen predictions and persistence; checks protocol, never trains a duplicate LSTM |
| `analyze.py` | Reads frozen tabular/distribution/Python research and actual R M1/M2/LSTM outputs; no legacy fallback |

Python classical research is not the R 37-candidate selector. The R selector averages key-horizon relative RMSE, while the Python VECM uses the rule above. Their model specifications and estimates can differ.

Actual R M1 forecasts are re-read from the dated final file and labeled `m1_<source method>`. The dated manifest must match the protocol, selection signature, input MD5, four-row selected configuration, and selected RDS. Old fixed-specification smoke outputs without selection signatures are rejected. M2 imports only `target_type=price`, retaining five real methods under `m2_<source method>`.

The explicit evaluation grid filters extra source horizons and terminal origins, then requires every method's complete grid. It never silently takes an intersection to hide missing methods. M1 zero-based and M2 one-based rows are normalized to zero-based; source row conventions, method names, and paths remain audit columns. Sparse R quantiles are not expanded into invented 63-quantile distributions; their point forecasts participate, but unsupported CRPS/FZ0 do not.

## Metrics and validation

Each mode/method/horizon reports `n`, `n_forecasts`, and status. October rolling counts are 17/13/0/0. Unsupported route-C h1 is `not_supported_by_model`; unrealized horizons retain empty metrics. Point metrics include dollar RMSE/MAE, aligned RW ratios, and nonzero directional-call accuracy. Distribution metrics use the native grid CRPS, HS/EWMA ratios, 90% coverage, and FZ0 differences.

October is too short for the reported asymptotic significance comparisons used in historical research. Compatibility significance columns remain blank. Duplicate/missing rows, nonfinite predictions, wrong protocol/date/index, and inconsistent anchors/actuals fail explicitly. Different target types and forecast modes are never pooled.

## Running and outputs

Generate compatible frozen R M1/M2, tabular, distribution, and LSTM outputs first. Then:

```powershell
uv run python -m m3_ml.compare.part1
uv run python -m m3_ml.compare.lstm_aligned
uv run python -m m3_ml.compare.analyze
```

`part1` performs actual selection/fitting and requires NumPy, pandas, SciPy, statsmodels, and arch. Current outputs are only under `m3_ml/compare/outputs/frozen_october_2026/`:

| Files | Contents |
| --- | --- |
| `part1_forecasts.csv`, `part1_quantiles.csv`, `part1_protocol.json` | Python classical forecasts, quantiles, assumptions, and simulation settings |
| `vecm_selection.csv`, `vecm_selection_by_year.csv` | Historical scores and counts |
| `lstm_forecasts.csv` | Imported real sequence forecasts |
| `point_forecasts.csv` | Unified point rows with actual source identities and row conventions |
| `point_metrics.csv`, `dist_metrics.csv` | Mode/method/horizon metrics, including n=0 rows |
| `methods_point.csv`, `methods_distribution.csv`, `protocol.json` | Method catalog, dates, limitations, and sample counts |

Older root-level outputs belong to the earlier rolling-refit study. They remain provenance and are not read by this evaluator or relabeled as October results.

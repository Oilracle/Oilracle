# Data files and processing

All model inputs are shared under `data/`. The main dataset has 3,803 quoted days from 2011-08-04 through 2026-10-23. File-level inventory and integrity checks are provided by `scripts/audit_submission_data.py`; source observations are preserved rather than rewritten during repository cleanup.

## Model-ready tables

| File in `data/processed/` | Contents and use |
| --- | --- |
| `wti_daily.csv` | Date, WTI target, 24 EIA features, and eight market features; shared by M1, M2, and M3 |
| `wti_daily.parquet` | The same table with typed dates and numeric columns |
| `features.csv` | Ordered column dictionary: group, units, frequency, source, availability, and meaning |
| `wti_publication.csv` | WTI label availability and source evidence, matched row by row to the main table |

`target_wti` is the outcome, not a forecast-time feature. The negative price on 2020-04-20 is preserved. `wti_spot_last_published` is a delayed feature and can differ from the canonical latest-published-price anchor used by the final models.

### Main-table construction

1. Extract six stock series, five production/trade/refinery series, four products-supplied series, and nine energy quotes from dated EIA Weekly Petroleum Status Reports (WPSR). Stocks use million barrels; flows use thousand barrels per day. Net imports use the report's “Net Imports (Including SPR)” row. Daily prices use the most recent reported week, not weekly/monthly comparison averages. The target is FRED `DCOILWTICO` in USD/barrel.
2. Forecast origins are 00:00 New York time. EIA features become usable on the first US federal business day after the actual report date. FRED/Yahoo market features use an assumed two-federal-business-day observation/close lag. Backward as-of matching selects the latest available observation; a late older report cannot overwrite a newer observation. EIA observations older than 21 calendar days and market observations older than seven days are initially blanked.
3. Construct a weekday grid, retain dates with a WTI quote, then forward-fill remaining feature gaps using only past values. The current export fills 63 cells: seven EIA price columns over nine rows from 2024-06-21 through 2024-07-03. It uses no interpolation, backward fill, zero fill, winsorization, standardization, or log transformation. CSV, Parquet, and the dictionary come from the same export.

Label availability is distinct from feature availability. A matching WPSR quote uses the first federal business day after its dated report (`matching_dated_archive`). Otherwise, availability is assumed to be the fifth federal business day after the quote week's Friday (`assumed_weekly_release`). These assumptions do not establish original historical vintages. Training and state updates use only labels available at each origin.

## Contemporaneous spot prices

`data/time_series/daily_spot.csv` follows the main table's dates and contains WTI, Brent, New York gasoline, heating oil, and the processed ten-year breakeven inflation feature. M1 and aligned VECM comparisons use it.

| Source in `data/time_series/raw/` | Processing |
| --- | --- |
| `brent_spot_daily.csv` | FRED `DCOILBRENTEU`, USD/barrel; select the study dates and forward-fill |
| `ny_conventional_gasoline_spot_daily.csv` | EIA daily gasoline, USD/gallon; multiply by 42 to obtain USD/barrel |
| `ny_heating_oil_spot_daily.csv` | EIA daily heating oil, USD/gallon; multiply by 42 to obtain USD/barrel |

Join by observation date and forward-fill gaps. The frozen VECM applies the WTI publication schedule to companion spot series as an explicit shared-calendar assumption; contemporaneous table dates do not imply same-day availability.

## Dated EIA reports

Each `data/raw/eia_reports/<release-date>/` directory contains source tables:

| Filename | Content |
| --- | --- |
| `table_01_petroleum_balance.csv` | Petroleum balance, inventories, and weekly flows; 15 selected series |
| `table_11_crude_and_gasoline_prices.csv` | WTI, Brent, New York/Gulf gasoline, and Los Angeles RBOB daily quotes |
| `table_12_other_product_prices.csv` | Heating oil, Gulf diesel/jet fuel, and Mont Belvieu propane quotes |

The parser handles Windows-1252 source encoding and segmented headers. It retains both observation periods and actual release dates. Crude prices remain USD/barrel; product prices in the main table remain USD/gallon. These dated source files are essential publication evidence, even when observations repeat across reports.

## FRED snapshots

| File under `data/raw/fred/` | Series and purpose |
| --- | --- |
| `wti_spot_daily.csv` | `DCOILWTICO`: main target, spread study, long-history construction |
| `brent_spot_daily.csv` | `DCOILBRENTEU`: Brent–WTI spread, long history, source for M1 |
| `treasury_yield_2y_daily.csv` | `DGS2`, percentage points; assumed two-business-day availability |
| `treasury_yield_10y_daily.csv` | `DGS10`, same treatment |
| `snapshots/2026-10-15/oil_volatility_ovx_daily.csv` | Initial `OVXCLS` snapshot |
| `snapshots/2026-10-15/real_treasury_yield_10y_daily.csv` | Initial `DFII10` snapshot |
| `snapshots/2026-10-15/breakeven_inflation_10y_daily.csv` | Initial `T10YIE` snapshot |
| `snapshots/2026-10-29/` with the same three filenames | Refreshed inputs used for the current main table |

OVX uses observations from 2009 onward. Yields and inflation retain percentage units. Long-history energy quotes use the weekly publication assumption described below. Spread models restrict their logarithmic calculations to positive, jointly observed prices; this does not remove the negative price from the shared main table.

## Market prices and futures

`data/raw/futures/sp500_daily.csv`, `vix_daily.csv`, and `dollar_index_daily.csv` are Yahoo `^GSPC`, `^VIX`, and `DX-Y.NYB` OHLCV snapshots. Main-table construction uses only Close after OHLC consistency checks, with a two-federal-business-day lag; adjusted close and volume are not substituted. Main-table market history begins in 2004; long-history research retains earlier valid observations.

`wti_nearby_futures_corpus_snapshot.csv` preserves the nearby WTI futures source used by the final futures/basis features. It is distinct from the discontinued EIA futures series in `data/long/`. The final feature builder uses Close and aligns basis values to the spot observation date, while respecting when that spot quote becomes available.

## EIA API reference series

| File under `data/raw/eia_api/` | Units and use |
| --- | --- |
| `commercial_crude_stocks_weekly.csv` | Thousand barrels; divide by 1,000 for the long-table stock feature |
| `crude_production_weekly.csv` | Thousand barrels/day; long-history production |
| `crude_imports_weekly.csv` | Thousand barrels/day; long-history imports and net imports |
| `crude_exports_weekly.csv` | Thousand barrels/day; long-history exports and net imports |
| `refinery_crude_inputs_weekly.csv` | Thousand barrels/day; long-history refinery activity |
| `spot_prices/ny_conventional_gasoline_spot_daily.csv` | USD/gallon; M1 source, crack spread, and long history |
| `spot_prices/ny_heating_oil_spot_daily.csv` | USD/gallon; M1 source, crack spread, and long history |
| `spot_prices/sources.csv` | Source identifiers, download date, coverage, and checksums |

Long-history weekly values use the fifth federal business day after the observation week's Friday. Long-history product prices remain USD/gallon. The 3-2-1 crack spread instead converts to USD/barrel and computes `(2 * gasoline * 42 + heating_oil * 42) / 3 - WTI` on common dates.

## News indices

| File under `data/raw/news/` | Selected data |
| --- | --- |
| `ai_gpr/ai_gpr_data_daily.csv` | Overall, oil, oil threats/acts, Middle East, and Russia AI-GPR series |
| `gpr/data_gpr_daily_recent.xls` | Date and keyword-based `GPRD`; event annotations are not predictors |
| `epu/All_Daily_Policy_Data.csv` | US daily policy uncertainty, combining year/month/day fields |
| `sources.csv` | Provenance, license, download date, coverage, and SHA-256 |

Features use `log(1+x)`, trailing seven- and 30-calendar-day means, and a seven-minus-90-day shock measure. Minimum valid counts are 4, 20, and 60 observations for the 7/30/90-day windows. Raw observations are not filled.

The approved `news-release-calendar-v1` protocol uses prior-month-end AI-GPR statistics assumed available on the first day of the next month; prior-Sunday keyword GPR statistics assumed available the following Monday, deferred for US federal holidays; and EPU assumed available the next day with at most seven calendar days of carry-forward. Monthly/weekly states expire at the next scheduled release: missing new source data remain missing rather than extending stale values indefinitely. Historical validation and October use identical rules.

These are release-calendar assumptions, not verified original publication timestamps. Revised data and retrospective AI scoring remain limitations. `python -m m3_ml.common.refresh_news --write` verifies official sources, archives replaced files locally, and updates provenance. The 2026-10-29 refresh covers AI-GPR through 2026-09-30, keyword GPR through 2026-10-26, and EPU through 2026-10-28. Downloading a later observation never makes it available earlier.

## Chart extracts and research data

`data/video/fan_integrated_h1.csv` and `fan_integrated_h5.csv`, with their JSON metadata, provide the final integrated fan-chart data. `fan_a_evt_h5.csv` is an earlier distribution-study extract: A-EVT five-day quantiles converted from log returns to prices using the origin price, ordered by target date, and rounded to four decimals. It is research provenance rather than the final forecast.

[Long-history data](long/README.md) remain a separate research dataset. Repeated source snapshots and CSV/Parquet pairs are intentional; cleanup does not merge vintages or replace the approved model inputs. Download support needed for rebuilding is documented alongside the submission data audit.

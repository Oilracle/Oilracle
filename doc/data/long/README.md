# Long-history research data

`data/long/` extends the historical coverage for research. The final forecast models do not directly read the long daily table.

## Tables

| File | Contents |
| --- | --- |
| `wti_daily_long.csv` | 8,268 WTI quoted days, 1991-02-15 through 2026-10-06; date, target, and 26 complete features |
| `wti_daily_long.parquet` | Typed equivalent of the CSV |
| `features_long.csv` | Units, frequency, source, availability rules, first observation, and stale carry-forward counts |
| `overlap_comparison.csv` | Column comparisons over 3,790 shared dates, 2011-08-04 through 2026-10-06 |

The overlap audit records exact matches, one/two-row lag matches, one-row lead matches, remaining differences, median deviations, and affected date ranges. It is produced by `m3_ml/long/build.py`.

## Construction

Use FRED WTI quote dates as target rows and retain raw prices. Current-vintage EIA, FRED, and Yahoo series supply 26 features. Weekly EIA observations, and daily energy quotes grouped into Saturday–Friday weeks, become available on the fifth US federal business day after Friday. Market closes use a two-federal-business-day lag. Only features available by 2026-10-15 enter this snapshot.

Convert stocks from thousand to million barrels; retain thousand barrels/day flows and USD/gallon product prices. Net imports equal same-week imports minus exports. Yahoo contributes only OHLC-consistent Close values. Backward as-of joins prefer newer observation periods and never let late older reports overwrite them. Values older than 21 days (EIA) or seven days (markets) are blanked and then causally forward-filled. No interpolation or backward fill is used.

## EIA series

| File in `eia_api/` | Description and use |
| --- | --- |
| `total_stocks_ex_spr_weekly.csv` | Total petroleum stocks excluding SPR; divide thousand barrels by 1,000 |
| `spr_crude_stocks_weekly.csv` | SPR stocks; same conversion |
| `gasoline_stocks_weekly.csv` | Gasoline stocks; same conversion |
| `distillate_stocks_weekly.csv` | Distillate stocks; same conversion |
| `jet_fuel_stocks_weekly.csv` | Jet-fuel stocks; same conversion |
| `total_products_supplied_weekly.csv` | Total products supplied, thousand barrels/day |
| `gasoline_products_supplied_weekly.csv` | Gasoline supplied, thousand barrels/day |
| `distillate_products_supplied_weekly.csv` | Distillates supplied, thousand barrels/day |
| `jet_fuel_products_supplied_weekly.csv` | Jet fuel supplied, thousand barrels/day |
| `gulf_conventional_gasoline_spot_daily.csv` | Gulf gasoline, USD/gallon; weekly availability rule |
| `gulf_jet_fuel_spot_daily.csv` | Gulf jet fuel, USD/gallon; weekly availability rule |
| `mont_belvieu_propane_spot_daily.csv` | Downloaded reference; not included in the long daily table |
| `wti_futures_contract1_daily.csv` through `wti_futures_contract4_daily.csv` | NYMEX contracts 1–4, USD/barrel, ending 2024-04-05; reference only, not the final futures input |
| `sources.csv` | Series IDs, URLs, download dates, coverage, units, row counts, and hashes |

## CFTC reference data

`cftc/wti_legacy_futures_only_weekly.csv` contains 1,938 legacy futures-only WTI reports from 1986-01-15 through 2026-10-20. The downloader selects contract code `067651`, retains original position fields, adds source filenames, concatenates years, and sorts by report date. No final prediction model reads this file. `cftc/sources.csv` records archive URLs, sizes, hashes, download dates, and extraction counts. Downloaded ZIP caches remain local.

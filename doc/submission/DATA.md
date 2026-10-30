# Data and publication protocol

Paths are relative to `project/`. The main dataset contains 3,803 WTI quoted days from 4 August 2011 to 23 October 2026. Stored source snapshots may extend beyond a model's fitting cutoff; availability filters determine what each fit or forecast can use.

## Main files

| File or directory | Contents |
| --- | --- |
| `data/processed/wti_daily.csv` and `.parquet` | Equivalent tables: date, WTI target, 24 EIA and eight market features |
| `data/processed/features.csv` | Column dictionary, units, source, frequency and availability |
| `data/processed/wti_publication.csv` | WTI label availability, evidence and source file |
| `data/time_series/daily_spot.csv` | Aligned WTI, Brent, gasoline, heating oil and breakeven inflation |
| `data/raw/` | Dated EIA reports, FRED/Yahoo snapshots and news indices |
| `data/rebuild/` | Manifests and source support for offline reconstruction |
| `data/long/`, `data/kronos/` | Separate long-history and sequence-model research inputs |

The target is FRED `DCOILWTICO`, in USD per barrel. `target_wti` is an outcome, never an available same-day predictor. Its negative observation on 20 April 2020 is retained; models requiring positive prices handle that restriction locally.

EIA reports supply petroleum inventories, flows and energy prices. Stocks use million barrels and flows use thousand barrels per day. Product prices in the main table remain USD per gallon. The time-series companion table multiplies gasoline and heating-oil prices by 42 to express them in USD per barrel. Market inputs include equity, dollar, volatility, yield, inflation and nearby WTI futures series. Consult `features.csv` for individual units; yields retain percentage units.

## Availability and missing data

The daily information boundary is 00:00 New York time. EIA features become usable on the first US federal business day after the actual report date. FRED/Yahoo market observations use an assumed two-federal-business-day lag. Backward as-of matching selects the latest available observation; a late older report cannot replace a newer observation.

WTI label timing is separate: a matched dated EIA report implies availability on the next federal business day. Unmatched quotes assume the fifth federal business day after the quote week's Friday. The VECM companion prices use the WTI publication calendar as an explicit shared-calendar assumption. Training excludes target windows whose required quotes have not yet been published.

EIA observations older than 21 calendar days and market observations older than seven days are initially masked. The exported main table then uses past-only forward filling: 63 cells in seven EIA price columns over nine dates in June–July 2024. It uses no interpolation or backward filling. The latest-published-price anchor is selected from the publication calendar and can differ from the delayed `wti_spot_last_published` feature.

## News features

`data/raw/news/sources.csv` records provenance, license information and checksums. News transformations use `log(1+x)`, trailing 7/30-day means and a 7-minus-90-day shock statistic.

| Source | Assumed availability |
| --- | --- |
| AI-GPR | Previous month-end statistics, available on the next month's first day |
| Keyword GPR | Previous Sunday statistics, released Monday; US federal holidays defer release |
| US EPU | Next day, with at most seven calendar days of carry-forward |

Monthly and weekly states expire at the next scheduled release if new source data are missing. Historical validation and October use the same rules.

These assumptions do not reconstruct complete real-time vintages. Revised histories and retrospective news scoring remain limitations. Separate long-history datasets have their own construction and should not replace the main table implicitly. Numeric files in `data/video/` are chart data used by report audits. Source archives and repeated snapshots are retained for reproducibility; the audit script checks them without downloading replacements.

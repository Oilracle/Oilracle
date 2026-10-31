# Data and publication protocol

Paths are relative to the repository root. The main dataset contains 3,803 WTI quoted days from 4 August 2011 to 23 October 2026. Stored source snapshots may extend beyond a model's fitting cutoff; availability filters determine what each fit or forecast can use.

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

## Data acquisition and refresh

The repository already includes the inputs required to reproduce the reported results. To rebuild those results, use the offline commands in [README.md](README.md#5-rebuild-data-from-the-included-snapshots). The commands below retrieve provider data for a **new research copy**. A new download can contain revised history and later observations, so it need not match the submitted snapshots or their hashes.

Use a separate clone with Git LFS files fully retrieved, and run from that repository root. Most downloaders have fixed destinations and overwrite files or manifests; they do not provide a general `--output-dir` option. Keep the reference checkout and its frozen outputs unchanged.

### 1. Environment and EIA credentials

Use the Python/uv setup in [README.md](README.md#1-set-up-the-environment). Internet access is required for downloads. The dated-report downloader also requires the `curl` executable on `PATH` (`curl.exe` on Windows); it is not installed by the Python lockfile.

```powershell
uv sync --locked
curl.exe --version
```

On Linux/macOS, check `curl --version` instead. FRED CSV exports, the Yahoo downloads used here, dated WPSR report files, CFTC archives and the news-index downloads do not use a project API key. The EIA API entry points require your own key from the [EIA registration page](https://www.eia.gov/opendata/register.php); the scripts use the legacy-series-ID route described in the [EIA API documentation](https://www.eia.gov/opendata/documentation.php).

Set the key for the current PowerShell session without putting its literal value in command history:

```powershell
$oilracleApiKey = Read-Host 'EIA API key' -AsSecureString
$env:EIA_API_KEY = [System.Net.NetworkCredential]::new('', $oilracleApiKey).Password
```

The EIA scripts also accept `EIA_API_KEY=your_key` in a local `.env` file at the repository root through `python-dotenv`; an existing environment variable takes precedence. No key or `.env` file is supplied in the public repository. Keep credentials out of shared files and source control. After completing all EIA download steps, clear the session variable with `Remove-Item Env:EIA_API_KEY`.

### 2. Download the main source groups

Run these commands sequentially and stop if any command fails:

```powershell
uv run --locked python -B scripts/download_no_api.py
uv run --locked python -B scripts/download_eia.py
uv run --locked python -B scripts/download_supplement.py
uv run --locked python -B scripts/validate_supplement.py
```

| Entry point | Sources and scope | Outputs in the supplied directory layout |
| --- | --- | --- |
| `scripts/download_no_api.py` | Five FRED series (`DCOILWTICO`, `DCOILBRENTEU`, `DTWEXBGS`, `DGS2`, `DGS10`); eight Yahoo tickers; disaggregated CFTC WTI positions | Normalized raw paths under `data/raw/fred/` and `data/raw/futures/`; retained auxiliary inputs under `data/rebuild/held-out/`; source archives under `data/rebuild/source_support/`; `data/rebuild/manifests/public_download.json` |
| `scripts/download_eia.py` | Seven EIA weekly series: commercial stocks, Cushing stocks, production, refinery inputs/utilization, imports and exports | `data/raw/eia_api/` plus held-out reference series; redacted response JSON under `data/rebuild/source_support/eia/`; `data/rebuild/manifests/eia_api_download.json` |
| `scripts/download_supplement.py` | FRED `OVXCLS`, `T10YIE`, `DFII10`, plus dated WPSR tables 1, 11 and 12 | `data/raw/fred/snapshots/<UTC-date>/`, `data/raw/eia_reports/<report-folder>/`, supporting report pages, and `data/rebuild/manifests/supplement/manifest_<UTC-date>.json` |
| `scripts/validate_supplement.py` | Offline checks of the complete supplement download | `.local/rebuild/checks/supplement_validation.json` |

The public downloader requests FRED CSV exports directly and uses `yfinance` for Yahoo. Its eight tickers are `CL=F`, `BZ=F`, `^GSPC`, `^VIX`, `XLE`, `GC=F`, `NG=F` and `DX-Y.NYB`. CFTC data are selected with WTI contract code `067651` from the [historical compressed reports](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm). Some downloaded series are reference-only; the builder does not automatically admit every downloaded column as a forecasting feature.

The supplement script reads the [official WPSR archive](https://www.eia.gov/petroleum/supply/weekly/archive/) from 2007 onward and retains release dates as well as observation dates. It caches individual report files and uses dated market snapshots, so rerunning can resume cached downloads. The EIA API's latest weekly history is a separate reference source; it must not replace dated WPSR publication evidence.

`scripts/data_layout.py` translates older logical path names into the locations above. Both EIA and supplement downloaders also write derived tables under `.local/rebuild/download_tables/`. `data/rebuild/` is used when present, as in this public release; otherwise the registry falls back to `.local/rebuild/`. Consult the printed manifest for exact physical paths, source URLs, retrieval times and hashes.

For a small supplement download probe, use `uv run --locked python -B scripts/download_supplement.py --limit 5`. This writes `_smoke` tables/manifests and may populate raw caches. It is not a complete build input: the preparation script deliberately ignores `_smoke` manifests. The full downloader must complete before the validation command above can pass.

### 3. Prepare and validate newly acquired inputs

After the three source downloads and supplement checks succeed, create an isolated prepared release:

```powershell
uv run --locked python -B scripts/prepare_data.py --output-dir .local/acquisition_check
uv run --locked python -B scripts/validate_prepared.py --output-dir .local/acquisition_check --check-rerun
```

The builder selects the newest complete, non-smoke supplement manifest, verifies source hashes and applies the publication-alignment rules. It refuses unresolved archive errors. The validator checks input/output hashes, temporal alignment, CSV/Parquet consistency and a deterministic rerun. Review `.local/acquisition_check/dataset_info/validation_report.json` and `quality_summary.json` before using the release.

An optional `--as-of 2026-10-29` on `prepare_data.py` restricts the forecast grid to the original snapshot cutoff; it cannot restore historical provider vintages. The date cannot exceed the newest local snapshot date. Downloads themselves generally retrieve available history rather than accepting a common historical end-date option.

If a maintainer deliberately wants to export refreshed model-ready tables, run the following **only in the research copy**, after reviewing the separate build:

```powershell
uv run --locked python -B scripts/prepare_data.py
uv run --locked python -B scripts/validate_prepared.py --check-rerun
uv run --locked python -B scripts/export_team_data.py
```

The default build destination is `.local/rebuild/release/`; the exporter reads that destination and overwrites `data/processed/wti_daily.csv`, `wti_daily.parquet`, `features.csv` and `wti_publication.csv`. Its study window is fixed in code to 4 August 2011-23 October 2026. It is not an automatic extension to a new forecasting period. The submitted models hash their original inputs, so replacing data requires the deliberate retraining workflow in [RETRAINING.md](RETRAINING.md). The submission comparison audit is intended for the retained reference snapshots; a newly downloaded revision should not be expected to reproduce their hashes.

### 4. Companion prices for time-series regression and spreads

To acquire the two New York product-price series, use the same EIA key:

```powershell
uv run --locked python -B -m m3_ml.spread.fetch_products
```

This writes gasoline and heating-oil CSVs and `sources.csv` under `data/raw/eia_api/spot_prices/`. It requests EIA series `PET.EER_EPMRU_PF4_Y35NY_DPG.D` and `PET.EER_EPD2F_PF4_Y35NY_DPG.D`, corresponding to FRED `DGASNYH` and `DHOILNYH`.

The M1 table builder reads dedicated copies in `data/time_series/raw/`. After refreshing the main table and downloading the products above, copy the three companion snapshots and rebuild the table in the research copy:

```powershell
Copy-Item -LiteralPath 'data/raw/fred/brent_spot_daily.csv' -Destination 'data/time_series/raw/brent_spot_daily.csv'
Copy-Item -LiteralPath 'data/raw/eia_api/spot_prices/ny_conventional_gasoline_spot_daily.csv' -Destination 'data/time_series/raw/ny_conventional_gasoline_spot_daily.csv'
Copy-Item -LiteralPath 'data/raw/eia_api/spot_prices/ny_heating_oil_spot_daily.csv' -Destination 'data/time_series/raw/ny_heating_oil_spot_daily.csv'
Rscript -e 'install.packages(c("dplyr","tidyr"), repos="https://cloud.r-project.org")'
Rscript m1_time_series/daily_spot_df.R
```

The final command overwrites `data/time_series/daily_spot.csv`, uses the main table's dates and converts product prices from USD/gallon to USD/barrel by multiplying by 42. The companion-price publication-calendar assumption remains unchanged.

### 5. News-index refresh

First compare the current official downloads with the local snapshots without writing them:

```powershell
uv run --locked python -B -m m3_ml.common.refresh_news
```

This still uses the network. To accept the checked downloads in the research copy:

```powershell
uv run --locked python -B -m m3_ml.common.refresh_news --write
```

The script reads the three URLs listed in `data/raw/news/sources.csv` and verifies them against its allowed source URLs. It validates date ordering, numeric values, local hashes and coverage before writing. Replaced files and the original provenance table are archived under `.local/news_source_archive/<timestamp>/`; refreshed files and `sources.csv` remain under `data/raw/news/`. It updates source snapshots, not the monthly/weekly/daily release assumptions described above. This is a refresh of the supplied snapshots, not a bootstrap command for an empty news directory.

### 6. Optional futures corpus and long-history research inputs

The supplied final futures/basis features already have a dedicated CSV snapshot. A bounded update of the existing Yahoo WTI corpus is available with:

```powershell
uv run --locked python -B -m m3_ml.kronos.corpus refresh-wti --through 2026-10-23
```

This fills only a missing suffix in `data/kronos/corpus/CL_F.parquet` and updates `m3_ml/kronos/corpus_manifest.csv`; it does not replace earlier bars or run if the requested date is already covered. It also does not update `data/raw/futures/wti_nearby_futures_corpus_snapshot.csv` automatically. The final market-feature layer reads that dedicated CSV, so a deliberate new-study refresh must keep it consistent with the corpus. The conversion used by that layer is:

```powershell
uv run --locked python -B -c "import pandas as pd; frame = pd.read_parquet('data/kronos/corpus/CL_F.parquet').reset_index().rename(columns={'Date':'quote_date','index':'quote_date'}); frame.to_csv('data/raw/futures/wti_nearby_futures_corpus_snapshot.csv', index=False)"
```

Run this conversion only when intentionally replacing the research copy's futures snapshot, then retain the source-overlap checks in the fusion preparation stages. Reviewing the saved reference results does not need this conversion.

To re-download the broader experimental OHLCV corpus, use `uv run --locked python -B -m m3_ml.kronos.corpus download`. It requests the configured universe from 2000 onward, uses the supplied S&P 500 membership/change files, overwrites successful ticker files and records per-ticker failures in `corpus_manifest.csv`. Review its `error` and `rows` columns rather than relying only on process exit status. It is not required to review the final forecasts. In a copy without `data/kronos/corpus/`, the script falls back to `.local/kronos/corpus/`.

Additional long-history source collection and table building are available separately:

```powershell
uv run --locked python -B -m m3_ml.long.download eia cftc
uv run --locked python -B -m m3_ml.long.build
```

The downloader writes `data/long/eia_api/`, `data/long/cftc/` and their `sources.csv` files, retaining CFTC ZIPs in `.local/long/cftc/`. Its current entry point checks `EIA_API_KEY` even if only `cftc` is requested, and its CFTC annual list ends in 2026. The builder reads these and existing raw sources; it writes `data/long/wti_daily_long.csv`/`.parquet`, `features_long.csv` and an overlap comparison. Its dates are fixed to an earlier research window (through 6 October 2026, with a 15 October availability cutoff). It does not replace the final main table. Run the main and product downloads first when reconstructing these long-history inputs.

### 7. Download failures and verification scope

Check `$LASTEXITCODE` after each PowerShell command. Stop on missing keys, failed HTTP requests, empty series, schema changes or unresolved archive errors. `download_supplement.py` records failed reports in its manifest and supports retrying cached work; quarantined date-conflict reports remain explicitly excluded. Do not substitute a current report for a missing dated report or manually change recorded hashes to suppress a failure.

The commands and paths in this section were checked against the packaged source, and the supported argparse help screens were exercised without downloading data. Live bulk acquisition was not rerun for this documentation update. Provider availability, historical revisions and schema changes can affect a future download; the bundled snapshots remain the basis for reproducing the submitted results.

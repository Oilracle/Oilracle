# Data directory

Use `processed/wti_daily.parquet` or its identical CSV as the shared model table.
It has 3,803 quoted days from August 4, 2011 to October 23, 2026, 32 features and
one outcome, `target_wti`. The contemporaneous outcome is never an input.
`processed/features.csv` defines columns and units; `wti_publication.csv` defines
verified or assumed label availability. Negative WTI on April 20, 2020 is retained.

| Directory | Contents and purpose |
|---|---|
| `processed/` | Final common model table, column dictionary and publication schedule |
| `raw/` | Unmodified EIA reports, FRED/Yahoo market snapshots and news indices with source records |
| `time_series/` | M1/M2 aligned market table and additional raw series; gasoline/heating oil are converted from USD/gallon to USD/barrel |
| `long/` | Separate 1991–2026 long-history research panel and its sources; it is not interchangeable with the main publication-aligned panel |
| `kronos/corpus/` | 588 cross-asset OHLCV Parquet inputs for the locally trained experimental Kronos module |
| `rebuild/` | Portable source manifests, raw JSON/HTML/archive support and deliberately held-out source snapshots needed for offline rebuilding |
| `video/` | Small derived chart extracts; not model-training inputs |
| `metadata/` | File inventory with SHA-256 checksums and the latest independent submission-data audit |

The Kronos corpus manifest is `m3_ml/kronos/corpus_manifest.csv`; its two frozen
stage checkpoints are in `m3_ml/kronos/checkpoints/`. Stored source snapshots may
extend beyond a training or evaluation cutoff. Each model reader enforces its
cutoff; storing later data does not make it eligible for fitting.

The raw and processed copies, source snapshots and held-out files are deliberate:
they preserve provenance, vintage comparisons, exclusion evidence and offline
reproduction. CSV/Parquet pairs provide readable and typed representations.
Raw source archives in `rebuild/source_support/` are data inputs, not nested
submission packages. Historical path strings in immutable download manifests are
resolved in memory by `scripts/data_layout.py`.

## Offline verification

From the project root:

```powershell
uv run python -B scripts/audit_submission_data.py
uv run python -B scripts/prepare_data.py --output-dir .local/data_check
uv run python -B scripts/validate_prepared.py --output-dir .local/data_check --check-rerun
uv run python -B scripts/audit_submission_data.py --rebuilt .local/data_check
```

These checks do not fetch data or train models. The first command verifies
source hashes, CSV/Parquet equivalence, quotation dates, publication evidence,
news completeness, corpus/checkpoint inputs and archive integrity. The optional
rebuild reconstructs the wider audited release, independently validates temporal
alignment and deterministic outputs, then compares its trimmed export with the
approved main table and publication schedule. It writes only to the chosen
local build folder. Do not overwrite `processed/` just to perform an audit.

Publication assumptions and incomplete historical vintages remain limitations;
integrity checks do not establish a fully real-time dataset. Full definitions
are in `doc/data/README.md` and `doc/data/long/README.md`.

# Oilracle: retraining guide

This guide complements the [setup and reproduction guide](README.md). Commands run from the repository root. Use a separate clone with Git LFS files fully retrieved for training, and retain an untouched checkout and its commit SHA as the reference. The commands describe the existing implementation; they do not change the submitted models or their reported results.

> **WARNING — Full retraining requires substantial time and compute.** Plan for extended training runs with a working CUDA GPU for GPU-capable neural components. Repeated historical fits, candidate searches and member replays can remain expensive even with GPU acceleration.
>
> **CPU-only full retraining is strongly discouraged: it is extremely time-consuming.** Treat a CPU-only full run as a separate adaptation task. Review and, where necessary, adapt device/tensor placement, batch sizes, process/thread scheduling and checkpoint handling, then validate the resulting execution path. Do not treat an automatic CPU fallback as an endorsed or validated full-retraining configuration. This guide does not provide a separately adapted CPU-only full-training recipe or a measured CPU completion-time estimate.
>
> **Scope matters.** VECM, STL and several ML/fusion stages are CPU-native. The supplied standalone LSTM implementation also uses CPU tensors and its own thread settings; installing CUDA does not move it to GPU. GPU acceleration of that implementation requires a separate code change. GPU-capable neural research such as Kronos should use CUDA. The warning concerns full research retraining, not the CPU-compatible saved-result review, data rebuild or conditional final-calibration checks.

## 1. Time boundaries and the two-stage protocol

The shared constants are in `m3_ml/common/protocol.py`. The main daily dataset starts in August 2011; a module may need a shorter usable sample because of lags, feature availability or monthly aggregation.

| Stage | Period or cutoff | Purpose |
| --- | --- | --- |
| Initial training | Information available through 31 August 2023 | Fit the initial candidate models |
| Historical development validation | 1 September 2023 through 30 September 2026 | Compare candidates and select rules using only labels already available at each selection/refit date |
| Final fitting | Information available by 30 September 2026 | Fit the selected specifications and freeze fitted parameters, scalers and calibration |
| Final retrospective evaluation | Target dates 1-23 October 2026 | Update observable inputs daily and evaluate without tuning on October outcomes |

The full development window is not the sample size of every reported metric. The final fusion's outer point-validation origins start in September 2025 and its scored targets end on 18 September 2026. Its matched risk-calibration comparison starts in June 2026. Standalone seasonality uses a longer first-stage validation window. See [METHOD.md](METHOD.md).

Use publication availability, not merely a source row dated before the cutoff. A target return can enter fitting only after its outcome is published. Preserve the release assumptions for spot/fundamentals, news and other inputs in any rerun.

Final horizons are **1, 5, 20 and 60 quoted days**, not calendar days. The final product uses `daily_update`. Some standalone scripts also produce `fixed_origin` outputs; these are retained research outputs and should not be substituted into the final rolling comparison. Do not introduce 17D, 30D or other horizons when reproducing this submission.

October is evaluation only. Do not change search grids, fusion weights, calibration caps or promotion gates in response to its scores. October was observed in earlier research, so this remains a retrospective evaluation, not a newly untouched test.

## 2. Environment, resources and run records

Complete `uv sync --locked` as described in [README.md](README.md) before the environment checks below.

Before launching GPU-capable neural training, verify that the installed PyTorch environment can actually use CUDA:

```powershell
uv run --locked python -B -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable: resolve the GPU environment before GPU training; CPU-only full retraining is strongly discouraged'; print(torch.cuda.get_device_name(0))"
```

This checks GPU availability only; it does not change a module's device selection or move CPU models/tensors. A CUDA-enabled PyTorch package alone does not establish that the driver, GPU and module implementation are configured correctly. Fix an unavailable CUDA environment before starting a GPU-dependent job, and inspect its startup device/logs rather than accepting an unnoticed CPU fallback.

R training and the member replay scripts additionally need R and these packages:

```powershell
Rscript -e 'install.packages(c("forecast","fGarch","vars","urca","car","sandwich","lmtest","jsonlite"), repos="https://cloud.r-project.org")'
Rscript -e 'sessionInfo()'
```

R packages are not pinned by `uv.lock`. Save `sessionInfo()` and package versions with the rerun. Python packages are locked, but different operating systems, numerical libraries, hardware or serialization versions can still change fitted coefficients or checkpoint bytes. Numerical agreement and time-boundary checks are separate from byte-for-byte archive identity.

The following settings apply only to CPU-native classical jobs and the CPU portions of the mixed workload. They are **not** a recommendation or adaptation recipe for CPU-only full retraining. Classical Python jobs default to two CPU threads and one outer job (use one thread on a one-CPU host):

```powershell
$env:M3_CPU_THREADS = "2"
$env:M3_OUTER_JOBS = "1"
$env:M3_LIGHTGBM_DEVICE = "cpu"
```

For CPU-native stages on a dedicated machine, `M3_CPU_THREADS=all` opts into its available CPU allocation. `M3_OUTER_JOBS` must stay within that budget. LightGBM GPU mode requires a working OpenCL setup and does not silently fall back if initialization fails; it is distinct from PyTorch CUDA. These controls do not configure R or every neural implementation; LSTM uses its own CPU settings and seeds.

Validation jobs may resume compatible entries from `.local/frozen_validation_cache`. Signatures include inputs, code, library versions and compute settings. A fast repeat run may therefore reuse completed fits. A fresh clone contains no such local cache.

Record commands, environment, data hashes, selected settings, seeds and logs. Stop at the first nonzero exit code; do not continue using stale outputs. On PowerShell, inspect `$LASTEXITCODE` after a command. There is no single validated command that retrains the entire delivered system. In particular, `m3_ml.run_serial` is an older module queue that includes video-preparation code excluded from the public source release; use the commands below.

## 3. Rerun an individual module

> **Training commands below can be long-running.** Read the resource warning above before executing them. CPU-native statistical modules can run individually on CPU; this does not make a complete CPU-only research rerun practical or recommended.

M1, seasonality and LSTM accept a separate output directory. Tabular/distribution/risk entry points use canonical paths, so run those in a separate research clone.

### Time-series regression / VECM

```powershell
Rscript m1_time_series/for_test_set.R --as-of=2026-10-23 --output=.local/retraining/m1
```

This runs historical selection, fits the selected bundle through the September availability cutoff, then forecasts and scores through the stated evaluation date. `--as-of` limits evaluation data; it does not move the training cutoff into October. Options use `--name=value`. Add `--selection-only=true` to stop after historical selection.

Outputs include selection scores/configuration, `frozen_fit_selected_2026-09-30.rds`, fitted parameters, and forecast/metric/manifest CSVs. Defaults are 10,000 simulations and seed 5210. Compatible selection caches and an existing final bundle are reused. Use a new empty output directory for a genuine new fit; an incompatible existing bundle is rejected. `validation_selection.R` contains the selection logic. The Python comparison proxy is not a substitute for this R implementation.

### Seasonality

```powershell
Rscript m2_seasonality/seasonality.R .local/retraining/m2
```

The output directory is a **positional argument**. This fits initial and September-final seasonal models and writes `validation_predictions.csv`, `final_predictions.csv`, scores and `frozen_fits.rds`. Monthly estimation uses complete available months; the daily bridge applies seasonal changes to the observable spot anchor.

`--rescore-validation` reuses existing fits/predictions and updates scoring and metadata; it is not retraining. This script has no general `--help` parser: an unrecognized argument can be treated as an output-directory name.

### LSTM

> This command performs actual neural fitting and repeated historical selection. The supplied implementation is CPU-native; it does not automatically use CUDA. Inspect/adapt the device implementation separately if GPU execution is intended, and allow substantial runtime. Do not infer from this standalone command that CPU-only full retraining is recommended.

```powershell
uv run --locked python -B m3_ml/lstm/src/oil_lstm.py --output .local/retraining/lstm
```

The default command performs historical configuration/epoch selection, final fitting and October evaluation. It uses a 60-quote input sequence and a 60-step output head, averaging three fitted seeds (42, 43 and 44). The submitted selection uses 32 hidden units, 20% dropout and 15 epochs. Default `--max-epochs 80` is the historical search ceiling, not the selected final epoch count. Scaling and final fitting obey the pre-October cutoff.

Outputs include metadata, search results, checkpoints, predictions and metrics. To score existing checkpoints instead of fitting them:

```powershell
uv run --locked python -B m3_ml/lstm/src/oil_lstm.py --test-only --output .local/retraining/lstm
```

`--forecast-only` produces a latest-origin forecast, not the October evaluation. `--data` and `--dictionary` can override the input table and feature dictionary; retain their bundled defaults for reproduction.

### ML point forecasts and predictive distributions

All four tabular variants are needed before reconstructing the foundation fusion specification:

```powershell
uv run --locked python -B -m m3_ml.tabular.run_direct
uv run --locked python -B -m m3_ml.tabular.run_direct --denoise
uv run --locked python -B -m m3_ml.tabular.run_direct --news
uv run --locked python -B -m m3_ml.tabular.run_direct --denoise --news
uv run --locked python -B -m m3_ml.distribution.run validate
uv run --locked python -B -m m3_ml.distribution.run test
```

Each tabular command performs historical selection, September fitting and October evaluation, writing `direct*` parameter/forecast files under `m3_ml/tabular/outputs/frozen_october_2026/`. Distribution `validate` selects methods/settings from history; distribution `test` fits the selected specifications at the September cutoff and evaluates them. Thus `test` in this module can include fitting, unlike final-fusion `test`. Run all distribution methods for the dependency chain; arbitrary subsets may leave required files incomplete.

### Additional standalone risk and spread research

These commands reproduce broader module research; final fusion has its own distribution calibration:

```powershell
uv run --locked python -B -m m3_ml.risk.volatility validate
uv run --locked python -B -m m3_ml.risk.volatility test
uv run --locked python -B -m m3_ml.risk.events validate
uv run --locked python -B -m m3_ml.risk.events test
uv run --locked python -B -m m3_ml.spread.run validate --spread brent_wti
uv run --locked python -B -m m3_ml.spread.run test --spread brent_wti
uv run --locked python -B -m m3_ml.residual.run
```

The spread module also supports `--spread crack_321`. `validate` selects from history, then `test` fits through September and evaluates October. The residual command performs its stages in one invocation. Outputs are beneath each module's `outputs/` directory. Kronos is an experimental companion rather than a required final fusion constituent; training it is unnecessary for reproducing the final system. The bundled futures corpus is still needed as a data source for market-feature stages.

## 4. Preserve frozen outputs before fusion retraining

Fusion entry points use fixed canonical directories without a general `--output` option. Sealed models include hashes of preceding stages. Do not mix newly trained upstream modules with previously sealed downstream selections.

In a **separate research clone**, archive the complete output directories for the stages you will regenerate. Do not remove only `test_audit.json` or edit hashes to bypass protection against replacing an evaluated model.

This PowerShell helper checks that both paths remain inside the current project, retains the moved directory and refuses an existing archive destination. Define it once in the research copy:

```powershell
$researchProject = (Resolve-Path -LiteralPath '.').Path
if (-not (Test-Path -LiteralPath (Join-Path $researchProject 'pyproject.toml'))) {
    throw 'Run this from the research repository root.'
}
function Preserve-TrainingOutput([string]$relativePath) {
    $prefix = $researchProject.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $sourcePath = [IO.Path]::GetFullPath((Join-Path $researchProject $relativePath))
    $backupPath = [IO.Path]::GetFullPath((Join-Path $researchProject ('.local/retraining_archive/' + $relativePath)))
    foreach ($candidatePath in @($sourcePath, $backupPath)) {
        if (-not $candidatePath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Refusing a path outside the research project.'
        }
    }
    if (Test-Path -LiteralPath $backupPath) { throw 'Archive destination already exists.' }
    if (-not (Test-Path -LiteralPath $sourcePath)) { return }
    New-Item -ItemType Directory -Path (Split-Path -Parent $backupPath) -Force | Out-Null
    Move-Item -LiteralPath $sourcePath -Destination $backupPath
}
```

For **final-layer retraining only**, preserve just:

```powershell
Preserve-TrainingOutput 'm3_ml/fusion_v6/outputs/frozen_october_2026'
```

Then proceed to section 6. This conditionally retrains the final correction/calibration layer from saved earlier research; it does not refit VECM, STL, LSTM or preceding fusion models.

For a **complete fusion reconstruction**, preserve all these directories before any training in section 5:

```powershell
$trainingOutputs = @(
    'm1_time_series/outputs/frozen_october_2026',
    'm2_seasonality/outputs/frozen_october_2026',
    'm3_ml/lstm/outputs/frozen_october_2026',
    'm3_ml/tabular/outputs/frozen_october_2026',
    'm3_ml/distribution/outputs/frozen_october_2026',
    'm3_ml/fusion/outputs/frozen_october_2026',
    'm3_ml/fusion_v2/outputs/frozen_october_2026',
    'm3_ml/fusion_v3/outputs/frozen_october_2026',
    'm3_ml/fusion_v4/outputs/frozen_october_2026',
    'm3_ml/fusion_v5/outputs/frozen_october_2026',
    'm3_ml/fusion_v6/outputs/frozen_october_2026'
)
foreach ($relativeOutput in $trainingOutputs) { Preserve-TrainingOutput $relativeOutput }
```

Keep bundled data and source unchanged. A fresh clone has no validation cache. In an older research copy, preserve `.local/frozen_validation_cache` with the same helper if an intentionally uncached run is required.

## 5. Reconstruct the complete fusion dependency chain

> **LONG-RUNNING TRAINING — Do not launch this chain as a quick reproduction check.** Reserve substantial compute time and CUDA/GPU resources for GPU-capable neural work. CPU-only reconstruction is strongly discouraged and must be treated as a separate code/resource adaptation and validation task. The CPU-native commands and thread controls below describe existing dependencies; they do not automatically accelerate those stages on GPU.

Execute the blocks in order and check each command's exit code. Earlier directories are successive training dependencies of one final product, not several competing delivered strategies. The complete chain is expensive; no new full-chain runtime estimate or fresh full-training replication is claimed for this documentation update.

### A. Fit member modules to their canonical destinations

```powershell
Rscript m1_time_series/for_test_set.R --as-of=2026-10-23
Rscript m2_seasonality/seasonality.R
uv run --locked python -B m3_ml/lstm/src/oil_lstm.py
uv run --locked python -B -m m3_ml.tabular.run_direct
uv run --locked python -B -m m3_ml.tabular.run_direct --denoise
uv run --locked python -B -m m3_ml.tabular.run_direct --news
uv run --locked python -B -m m3_ml.tabular.run_direct --denoise --news
uv run --locked python -B -m m3_ml.distribution.run validate
uv run --locked python -B -m m3_ml.distribution.run test
```

Do not use section 3's `.local/retraining/...` destinations here: downstream fusion reads canonical paths. Confirm that selected configurations, final predictions and distribution quantiles exist before continuing.

### B. Foundation and horizon-specific fusion

The foundation LSTM replay sets three outer workers internally. It therefore needs a CPU allocation of at least three and `M3_CPU_THREADS` of at least three; the two-thread default is insufficient for this specific step. The block below uses three. On a smaller host, use the supplied member replay outputs for review instead of attempting this unchanged full reconstruction.

```powershell
$env:M3_CPU_THREADS = "3"
uv run --locked python -B -m m3_ml.fusion.prepare spec
Rscript m3_ml/fusion/replay_members.R
uv run --locked python -B -m m3_ml.fusion.prepare ml
uv run --locked python -B -m m3_ml.fusion.prepare lstm
uv run --locked python -B -m m3_ml.fusion.run select
uv run --locked python -B -m m3_ml.fusion.run test
uv run --locked python -B -m m3_ml.fusion_v2.prepare spec
Rscript m3_ml/fusion_v2/replay_members.R
uv run --locked python -B -m m3_ml.fusion_v2.prepare lstm
uv run --locked python -B -m m3_ml.fusion_v2.run select
uv run --locked python -B -m m3_ml.fusion_v2.run test
```

R replays require `jsonlite`. The foundation retains intermediate path horizons and its original minimum-weight constraint as historical dependencies; the later horizon-specific layer removes that constraint. These are not extra final reported horizons or a restored minimum weight in the delivered strategy. The second replay explicitly expects the selected VECM and monthly STL bridge. Different member winners can stop it on that assertion: investigate the changed selection rather than disabling the check. Supporting another model family is a separate research change.

### C. Market-feature fusion

```powershell
uv run --locked python -B -c "from m3_ml.fusion_v3.data import specification; specification()"
uv run --locked python -B -m m3_ml.fusion_v3.point
uv run --locked python -B -m m3_ml.fusion_v3.risk
uv run --locked python -B -m m3_ml.fusion_v3.evaluate
uv run --locked python -B -m m3_ml.fusion_v3.run seal
uv run --locked python -B -m m3_ml.fusion_v3.run test
```

The explicit function call is necessary because this data module has no executable `__main__` entry point. This stage reads the bundled nearby-futures snapshot/corpus and checks overlapping sources; no new download is required.

### D. Longer-horizon and market-state models

```powershell
uv run --locked python -B -m m3_ml.fusion_v4.data
uv run --locked python -B -m m3_ml.fusion_v4.long_models
uv run --locked python -B -m m3_ml.fusion_v4.stack
uv run --locked python -B -m m3_ml.fusion_v4.guard
uv run --locked python -B -m m3_ml.fusion_v4.evaluate
uv run --locked python -B -m m3_ml.fusion_v4.risk
uv run --locked python -B -m m3_ml.fusion_v4.run seal
uv run --locked python -B -m m3_ml.fusion_v4.run test
```

### E. Short-horizon refinement and risk safeguards

```powershell
uv run --locked python -B -m m3_ml.fusion_v5.data
uv run --locked python -B -m m3_ml.fusion_v5.point
uv run --locked python -B -m m3_ml.fusion_v5.evaluate
uv run --locked python -B -m m3_ml.fusion_v5.risk
uv run --locked python -B -m m3_ml.fusion_v5.support
uv run --locked python -B -m m3_ml.fusion_v5.evaluate
uv run --locked python -B -m m3_ml.fusion_v5.damp
uv run --locked python -B -m m3_ml.fusion_v5.evaluate
uv run --locked python -B -m m3_ml.fusion_v5.risk
uv run --locked python -B -m m3_ml.fusion_v5.risk_safety
uv run --locked python -B -m m3_ml.fusion_v5.run fit
uv run --locked python -B -m m3_ml.fusion_v5.run seal
uv run --locked python -B -m m3_ml.fusion_v5.run test
```

Support, damping and risk-safety steps implement the recorded historical amendments. Repeated evaluations are intentional dependencies, not permission to select a new variant from October scores. Continue with section 6.

## 6. Fit and seal the final correction/calibration layer

Run after preserving only the final output directory in section 4, or after completing all of section 5:

```powershell
uv run --locked python -B -m m3_ml.fusion_v6.research
uv run --locked python -B -m m3_ml.fusion_v6.tighten
uv run --locked python -B -m m3_ml.fusion_v6.run seal
uv run --locked python -B -m m3_ml.fusion_v6.run test
```

`research` registers candidates, fits historical short-horizon corrections and evaluates the risk proposal using pre-October information. `tighten` applies the recorded historical amendment limiting price-interval expansion to 1.35. `seal` requires the historical promotion gate to pass and records dependency hashes. `test` applies saved final models to October forecasts and evaluates them; it does not select parameters.

Inspect `point_gate.json`, `risk_gate.json`, `promotion_gate.json`, `point_audit.json`, `risk_audit.json`, `width_amendment.json`, `selected.json` and `test_audit.json`. In the submitted reference, additional short-horizon point corrections were not promoted; the accepted change recalibrated five-day tails while retaining point forecasts and the other horizon distributions. Different rerun selections require investigation, not edits to gates until a desired outcome appears.

## 7. Regenerate and check the delivery

```powershell
uv run --locked python -B -m m4_reports.final_strategy
uv run --locked python -B -m m4_reports.short_strategy --audit
uv run --locked python -B -m m4_reports.plot_final_forecast
uv run --locked python -B -m m4_reports.plot_historical_forecast
uv run --locked python -B m4_reports/prepare_seasonality_slide_evidence.py
uv run --locked python -B -m pytest -q m3_ml/tests/test_fusion_v6.py
```

Compare with the [README.md](README.md) result table and retained reference checkout. Check dates, horizons, target counts, published-spot baseline, interval coverage/width and selected specifications before comparing rounded scores. A fresh fit can change the final selected hash because serialized models and inputs are hashed. That differs from an unexpected mismatch within one run, which must pass the integrity checks.

Reporting checks the submitted product's invariants, including unchanged final point forecasts and the five-day-only risk adjustment. A materially different research result may need a separate reporting implementation; do not silently relabel it as the submitted model.

## 8. Troubleshooting

| Symptom | Check or action |
| --- | --- |
| `ModuleNotFoundError` or missing relative files | Confirm the working directory is the repository root, then run `uv sync --locked`. |
| Missing `Rscript` or R package | Install R, make `Rscript` available to the terminal and install section 2's packages. |
| Cannot replace an evaluated model | Use the research clone and preserve the whole affected output directory before fitting. Retain its audit trail. |
| Input-hash mismatch | Check for changed code/data or mixed new/old sealed stages. Restore the reference or rebuild downstream stages in order. |
| Missing specification/selection/quantile file | Complete the preceding preparation, validation or fitting stage. |
| R replay rejects a selected family | Compare selection with the submitted VECM/STL specifications, which the replay supports. |
| A validation run finishes unusually quickly | Check cached-task reuse. M1 can also reuse its saved final bundle. |
| `outer workers must fit within M3_CPU_THREADS` in foundation LSTM replay | That replay sets three workers internally. Allocate at least three CPUs and set `M3_CPU_THREADS` to at least three, within the host allocation. |
| GPU initialization fails | Stop the GPU-dependent job and fix the CUDA/driver/device setup. CPU-only full retraining is strongly discouraged and requires separate adaptation and validation; do not silently switch the full workload to CPU. |
| October 20D/60D RMSE is blank | Expected: no outcomes at those quoted-day horizons mature within 1-23 October. Use historical scores. |
| Refit scores differ slightly | Check library/R-package versions, numerical settings, seeds and snapshots before investigating model logic. |

The verification scope and executed review checks are listed in [README.md](README.md#verification-scope). Full member and fusion retraining was not rerun. Entry points and order were checked against the source; a fresh complete training replication remains a separate, potentially lengthy task.

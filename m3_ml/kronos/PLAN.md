# Kronos architecture research and current protocol

> **Protocol cutover: the plan below is historical, not the current experiment.**
> Current stage 1 trains on futures/corpus bars through 2023-08-31 and selects
> tokenizer/predictor durations on purged historical validation through
> 2026-09-30. Sampling temperature/top-p are selected on that interval only,
> minimizing mean CRPS over h1/5/20/60 to produce one coherent path sampler.
> Stage 2 trains genuinely from scratch through 2026-09-30 for those fixed
> durations, without early stopping or any October validation.
> Accordingly stage 2 has no validation sampler: requesting one raises an explicit
> error before loading the corpus. Regression tests separately enforce both
> stages' training cutoffs and stage 1's complete future-label validation purge.
>
> All per-ticker means/stds are fitted on each stage's training rows, never
> re-estimated on validation/test contexts. Final weights and normalization are
> frozen throughout October. Fixed-origin generation uses only futures OHLCV
> through Sep30; daily-update generation admits only bars through that origin.
> CL daily session closes are assumed available after exchange close, separately
> from delayed WPSR spot publication. This market-bar convention is not an audit
> of release times or revised historical futures vintages.
>
> Validation spot targets must have publication availability by Sep30 using the
> project evidence/retrospective weekly-rule schedule. New results are exclusively
> `outputs/frozen_october_2026/`; new checkpoints are `m3_ml/kronos/checkpoints/`.
> Historical old checkpoints cannot be silently reused. No download of official
> large pretrained weights occurs. Use `train --stage 1`, `run validate`,
> `train --stage 2`, then `run test`, each as `uv run python -m m3_ml.kronos...`.
> The old validation stop rule/results below do not stand in for new results.
> The existing `data/kronos/corpus/` cache remains the corpus input; a new
> checkpoint directory does not require redownloading it.
> Before October daily evaluation, update the cached CL suffix with
> `uv run python -m m3_ml.kronos.corpus refresh-wti --through 2026-10-23`;
> the original cache ended October 16. This downloads only missing futures bars,
> not a model or the entire cross-asset corpus.
> Training and inference use CUDA when available. Inference accepts
> `--device auto|cpu|cuda` (default `auto`); explicit CUDA failures are not
> retried on CPU. `--batch-sequences 200` bounds concurrent origin/path
> sequences, splitting paths when needed without reducing Monte Carlo totals.
> All chunks are pooled before computing quantiles. Changing the cap changes
> random-number assignment, not the selected checkpoint or sampling grid.
> Grid and batch logs report origins, paths and completed/total counts.
> Run heavy commands serially; for example:
> `uv run python -m m3_ml.kronos.run validate --device cuda --batch-sequences 200`,
> then stage-2 training, then
> `uv run python -m m3_ml.kronos.run test --device cuda --batch-sequences 200`.
> Omit `--paths` for full protocol counts; that override is for test/forecast
> smoke runs only. There is no inference-specific environment override.
> After each refresh, `run forecast` loads
> stage-2 weights/scalers and emits `latest_forecast.csv` for the newest origin,
> without refitting or overwriting the October evaluation. A scheduler can call
> this CLI after the project ingestion job; the model does not fetch future data.
>
> Test targets are October 1–23, separate fixed-origin and daily-update modes.
> Fixed-origin emits a full h1–60 path; daily-update emits h1/5/20/60. Beyond
> observed quote rows, target dates/actuals stay NA while numerical forecasts are
> retained. Date embeddings beyond known quotes use an explicitly assumed
> federal-business-day scenario, not a claimed future quotation calendar. Missing
> past exchange sessions are simulated as bridge steps, not compressed out of
> horizons. CRPS is validated on log ratios; final quantiles/point forecasts are
> converted to spot USD/barrel using the observable published WTI anchor.
> Spot publication evidence and upstream feature conventions do not establish
> fully real-time data vintages, and previously inspected October data mean this
> cutover remains a retrospective evaluation.


## Archived preregistered design (October 17, 2026)

Kronos encodes each OHLCV bar into coarse/fine BSQ tokens and predicts them with
a decoder-only Transformer. Multiple sampled paths supply a distribution.
Official pretrained weights overlapped the old test period, so this project
uses the architecture trained locally from scratch. Model/tokenizer definitions
are vendored under MIT at commit `67b630e`; see `vendor/VENDOR.md`.

Training, data preparation and path generation are project code. Full-window
normalization in the official CSV example would include future rows and is not
used. The custom generator retains individual paths rather than averaging them.
Full-position fine-token attention fixes the official inference/training rotary
position mismatch without altering the architecture.

### Corpus

The fixed cross-asset universe includes energy, metals/agriculture, equity/rate
futures, FX, indices and ETFs. S&P 500 membership is reconstructed at each cutoff
by reversing later changes. Missing delisted Yahoo histories leave residual
survivorship bias. OHLC uses split adjustment without dividend adjustment
(`auto_adjust=False`); volume and amount are zero for assets without volume.
Amount equals volume times mean OHLC. Missing/nonpositive bars are dropped except
the negative WTI futures bar on April 20, 2020. Input starts in 2000 and every
reader enforces its stage cutoff, even when the stored corpus contains later data.

Portable input is `data/kronos/corpus/`; checkpoints are in
`m3_ml/kronos/checkpoints/`. These snapshots replace hidden local-cache dependencies.
The legacy `.local` paths are fallbacks for older workspaces only.

### Original stages and fixed architecture

The archived stage 1 trained windows ending by 2018-12-31, stopped on 2019 and
selected sampling on 2020–2022. Archived stage 2 trained through 2021-12-31 and
stopped on 2022. The current cutover above supersedes these dates and normalization
rules. Each stage trains from scratch with the same configuration and seed.

Tokenizer: input 6, model dimension 256, four heads, feed-forward 512, encoder/
decoder depth four (three actual blocks), 10 bits per coarse/fine token, beta
0.05, gamma0 1.0, gamma 1.1, zeta 0.05, group size five and no dropout.
Predictor: dimension 256, four layers/heads, feed-forward 512, feed-forward and
residual dropout 0.2, attention/token dropout zero, learned time embeddings.

Windows contain 256 context and 64 future rows. The archived plan standardized
using context only; the current implementation uses frozen stage-training
per-ticker statistics. Values clip to ±5. Calendar inputs are weekday/day/month
with zero minutes/hours. Sampling first chooses an asset then an origin;
cross-asset and equity groups each receive 50%, and five energy futures have
triple weight within the cross-asset group.

Tokenizer loss averages coarse reconstruction, full reconstruction and BSQ loss.
AdamW uses learning rate 2e-4, betas 0.9/0.95, decay 0.1, OneCycle with 3% warmup
and div 10, batch 64 and gradient clip 2.0. Original maximum: 30,000 steps,
validation every 1,000, patience five. Predictor freezes the tokenizer and uses
mean coarse/fine next-token cross entropy, learning rate 5e-4, the same optimizer
family/schedule, batch 64, clip 3.0 and bf16. Original maximum: 60,000 steps,
validation every 2,000, patience five. Fixed validation uses 2,048 windows and
teacher-forced coarse tokens for deterministic fine-token loss. Current stage 2
uses stage-1-selected durations with no October early stopping.

### Generation and archived evaluation

Input is the latest 256 CL=F daily bars available at the origin. The sampler
generates 60 steps with 200 paths per validation origin and 1,000 for test.
Decoded origin close is the ratio denominator, canceling tokenizer reconstruction
bias. Quantiles of future/origin close ratios are logged and converted to spot
prices with the published anchor. Nonpositive ratios floor at 0.001.
Future timestamps are an explicitly assumed federal-business-day scenario.
Temperature 0.8/1.0 and top-p 0.9/1.0 form the four candidates; top-k is disabled.

The archived evaluation selected annual 2020–2022 CRPS/HS geometric means and
planned a fixed-weight 2023–October 6, 2026 test. Its stop rule was failure to
converge or validation CRPS more than 5% worse than HS at every horizon. That
rule triggered and the old test was not run. It does not replace the current
frozen October training/inference evidence.

### Corrections and implementation checks

Before the first archived validation result, fine-token attention in evaluation
was found to be noncausal because the vendor module used `is_causal=self.training`.
Validation now activates training mode only for that zero-dropout attention
module. At the retained step-2,000 model, fine-token cross entropy changed from
2.117 to 2.093; coarse loss remained 3.840. The predictor was retrained with an
independent SEED+2, reusing the unaffected tokenizer. Regression checks cover
causality and the corrected loss.

GPU generation exhausted memory and was followed by Windows hypervisor crashes
(0x20001). KV caching shares each origin context across paths and only processes
new tokens. Bounded batches cap concurrent sequences; predictor caches are freed
before tokenizer decoding. Path totals and selected settings do not change,
although batch size changes random-number assignment. Historical checks matched
cached and full-recomputation logits/paths; they do not certify every later CUDA
batch configuration. An October 29 smoke run with two origins, 200 paths each
and batch cap 100 completed in 8.48 seconds, producing four finite, ordered
(2, 63) arrays. This was a smoke check, not a full-grid acceptance result.

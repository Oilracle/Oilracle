# LSTM: frozen October protocol

The current implementation is `src/oil_lstm.py`. **Existing files directly under
`outputs/`, the executed notebook, old speaking notes and binary slides are
historical artifacts from a superseded experiment, not October-protocol results.**
The new runner writes only `outputs/frozen_october_2026/`; old checkpoints are not
accepted as approved fits and are never silently relabeled.

The completed canonical-anchor run is summarized in `results_summary.md`:
3,659 final-fit origins, three seeds and 15 fixed epochs. Daily-update LSTM RMSE
is 12.939878 at h1 (n=17) and 17.273680 at h5 (n=13), worse than canonical
persistence (9.110146 and 11.253163). h20/h60 have n=0; no long-horizon accuracy
claim is supported by this target window.

## Protocol

- Initial training: information available through 2023-08-31.
- Historical validation: target dates 2023-09-01 through 2026-09-30, with all
  selected labels published/assumed available by September 30.
- Final fit: information available through 2026-09-30, then freeze neural weights,
  feature/label scalers, architecture, epochs and ensemble membership.
- Final targets: 2026-10-01 through 2026-10-23. Report modes separately and always
  report counts for h=1,5,20,60. A missing h20/60 realized outcome is `n=0`, not a
  missing numerical forecast and not a reason to shift the final window.

The model is a direct 60-output LSTM, not recursive one-step regression. One
feature window produces an entire path without future actual exogenous inputs.
`fixed_origin` uses only the feature window ending on September 30; every feature
was public before its row date per the supplied feature dictionary.
`daily_update` advances only that causal feature window; weights and
preprocessing do not refit. Origin dates identify the feature-window ending row,
and h1 predicts the next quoted target row. Unknown future target dates stay blank;
`target_row` preserves the row index. Seed range is not a prediction interval.

Every supervised sample has 60 labels. Any path containing a label whose
`available_date` exceeds the relevant cutoff is purged, preserving row spacing.
`data/processed/wti_publication.csv` combines actual matched WPSR evidence and an
explicit **retrospective weekly-release assumption** for unmatched dates.
Covariate publication conventions and current vintages remain limitations; this
is not proof of a fully real-time vintage replay. Earlier development inspected
parts of October, so the new cutover is retrospective, not untouched prospective
evidence.

The anchor channel is transformed in memory using the common evidence-asof
`observable_price` helper, choosing the newest quote whose publication schedule
permits it; a delayed old release cannot rewind that anchor. Both residual
training targets and the LSTM input channel use this same transformation. The
original supplied CSV is not edited. Only leading pre-table cold-start rows use
their supplied already-published quote when no scheduled table quote exists.

## Run from the repository root

```powershell
uv run python m3_ml/lstm/src/oil_lstm.py
uv run python m3_ml/lstm/src/oil_lstm.py --test-only
uv run python m3_ml/lstm/src/oil_lstm.py --forecast-only
uv run python -m unittest discover -s m3_ml/lstm/tests -v
uv run python m3_ml/lstm/build_handoff.py --execute
```

The first command genuinely trains candidates, selects epochs on historical
validation and trains the final three seeds. Existing historical checkpoints
cannot replace this operation. The next two require the newly fitted approved
checkpoints. `--forecast-only` loads fixed weights/scalers after upstream data
refresh; new targets may be absent. `refresh_forecast.ps1` provides the same
inference action; downloading/alignment/scheduling belongs to the project data
pipeline, not model fitting. The handoff generator refuses absent or legacy
results and only then replaces the historical summary/notebook with real new
outputs. It does not overwrite binary slides.

Machine-readable predictions include `mode,origin_date,target_date,target_row,
horizon,method,origin_price,forecast,actual` plus seed diagnostics and protocol
provenance. `test_metrics_key_horizons.csv` reports separate mode/method/horizon
counts, MAE and RMSE. No scores are fabricated before training is run.

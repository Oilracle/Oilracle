# ML correction of actual member models

This historical feasibility study tests ML residual correction of native VECM and monthly STL, not the independent ARIMA/calendar baseline in `m3_ml/residual/`. It did not fit/evaluate October or replace the then-sealed v2 policy. Evidence mainly supports further research into longer-horizon VECM corrections, not an upgrade at every horizon.

The rule is `native prediction + gamma * ML(origin-known features)`, with gamma in 0/0.25/0.5/1 and no-correction/constant-bias controls. Targets are genuine historical out-of-origin dollar errors from 63-origin native refits, not fitted residuals. h1/5/20/60 use three regularized Elastic Nets and two shallow LightGBMs with calendar, market-state, and interaction feature groups. Missing mature-error history uses a zero state plus zero availability count, never future backfill. There are 136 parallel tasks and 49 candidates per member/horizon including shrinkage.

Quarterly cross-fits begin March 2024, past-score-only selection September 2024. All labels are available before October; the latest publication is September 24. Inherited base-model selection still makes this conditional development evidence.

Negative percentages below mean lower RMSE relative to the same native model, not to RW.

| Horizon | n | M1 + ML vs original VECM | M2 + ML vs original STL |
| --- | ---: | ---: | ---: |
| 1 day | 508 | −0.42% | −0.19% |
| 5 day | 504 | +1.53% | +0.47% |
| 20 day | 489 | +0.67% | +2.40% |
| 60 day | 449 | **−6.47%** | −0.62% |

M1 h60 RMSE falls from USD 17.5201 to 16.3874, versus RW 17.5314 (ratio about 0.9347). Six of eight quarterly periods improve, but first/last periods are partial and 2025 Q1 has ratio 1.3323. Overlapping 60-day targets are not 449 independent observations. A hindsight full-history candidate gives 11.34% improvement, recorded separately and not substituted for forward selection.

Calendar-only ML did not improve M2 in the forward comparison. Prespecified calendar-plus-state subgroups yield h60 ratios 0.9242 for M1 and 0.9653 for M2, versus the full selector's 0.9353/0.9938. These are diagnostic subgroup results, not replacements chosen after seeing scores. Constant-bias correction does not explain the main M1 gain. Calendar interactions may warrant separate risk/volatility research, which this study did not test.

Code: `scripts/probe_member_ml.py`. Outputs: `m3_ml/residual/outputs/member_ml_probe/`, including comparison, forward forecasts, quarterly scores, candidate choices, feature ablations, and audits. Future-loss/unpublished-error perturbations leave earlier selection/features unchanged; official v2 hashes were preserved. Reproduce with `python -m scripts.probe_member_ml` in a separate research copy, optionally setting `M3_CPU_THREADS=16` and `M3_OUTER_JOBS=16`.

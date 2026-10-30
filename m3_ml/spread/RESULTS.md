# Spread forecasting results and historical archive

## Current frozen evaluation

Both Brent–WTI and 3-2-1 crack studies completed the shared frozen October protocol. Their outputs are in `outputs/frozen_october_2026/`. Final fitting uses September 30 available labels once per horizon for both modes; October updates published state only. Intermediate fixed-path horizons inherit the next larger key-horizon hyperparameters while fitting their own labels. OU preserves the true quote-row distance between anchor and target.

All legs use the declared common WTI publication-delay assumption, not independently verified Brent/product release histories. Features consume published spread/EIA values. Only realized October target dates are scored, with n=0/empty metrics for unrealized endpoints. `inference=descriptive_short_window` suppresses October DM p-values and significance/winner flags. Current-vintage and overlapping-target limitations remain.

Run `python -m m3_ml.spread.run validate --spread brent_wti`, then `test`, then `python -m m3_ml.spread.analyze --spread brent_wti`; repeat with `crack_321`. Mode-specific metrics, trading illustrations, and figures are generated.

## Legacy Brent–WTI results

The following values and root-level `outputs/` describe the original 2023-01-03–2026-10-06 experiment (selection records `36039a2`/`c393fcc`), not the new October-only window. Current commands no longer reproduce the old split.

| Horizon | Best method | RMSE/RW | DM p | Direction accuracy |
| --- | --- | ---: | ---: | ---: |
| 1 | Elastic Net | .955 | .029 | 59.8% |
| 5 | Elastic Net | .984 | .39 | 63.5% |
| 20 | MA deviation | .902 | .11 | 63.0% |
| 60 | OU / AR(1) | .772 | .18 | 71.3% |

h1 meets the original significance rule. Longer-horizon RMSE improves 10–23% but overlapping samples give weak DM evidence. Direction F1 is .59–.75. The h20 MA-deviation spot rule produces USD 156/barrel cumulative P&L, Sharpe 1.75, USD 28 drawdown, 60% win rate, 139 trades, and 4.6-day average holds.

### Asynchronous quote diagnostic

Brent's London assessment and WTI's New York assessment differ by about four hours, inducing artificial short-run negative autocorrelation. The historical training comparison with approximately simultaneous futures settlement gives:

| Series | Lag-one change autocorrelation | VR20 | VR60 |
| --- | ---: | ---: | ---: |
| Spot spread | −.285 | .36 | .24 |
| Futures spread | −.101 | .69 | .56 |

Much of the spot h1 advantage and high short-term trading Sharpe (h5 about 2.15) is therefore not an executable gain. Futures retain weaker longer-horizon mean reversion. A tradability claim requires a separate roll-adjusted futures study. The 2026 stress interval widens the spread to about USD 20–35 and drives the largest drawdown. These diagnostics were added after testing.

## Legacy crack-spread results

The three New York-assessed legs reduce the asynchronous-quote issue. Training lag-one change autocorrelation is −.043 and VR60 .47.

| Horizon | Best method | RMSE/RW | DM p | Direction accuracy |
| --- | --- | ---: | ---: | ---: |
| 1 | Elastic Net | .993 | .18 | 51.5% |
| 5 | MA deviation | .986 | .18 | 54.3% |
| 20 | MA deviation | .978 | .26 | 58.6% |
| 60 | Historical drift | 1.005 | .25 | 46.3% |

No model significantly beats RW. h1–20 gains are only .7–2.2%; h60 mean-reversion models have ratios 1.07–1.25 as the 2026 crack spread rises to USD 79/barrel. h20 ratios during 2023–2025 are .89–.98; direction F1 is .57–.61. The h20 MA-deviation spot rule produces USD 126/barrel, Sharpe 1.10, USD 22 drawdown, 94 trades, and 8.6-day holds. This remains a spot-price illustration, not verified futures returns.

Spreads show some interpretable directional/mean-reversion structure, but accuracy gains are limited after accounting for assessment timing and structural shocks. They are supporting research, not the project's final unified forecast or an independently validated trading fund.

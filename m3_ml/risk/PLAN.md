# Risk forecasting research plan

## Current protocol

Current execution uses initial training through 2023-08-31, historical development through September 2026, a final fit using information available by 2026-09-30, and October 1–23 evaluation. Parameters are frozen in October. Outputs live under `outputs/frozen_october_2026/`; fixed-origin and rolling results are separate, and unrealized targets are not scored. The official final product presents rolling forecasts only. Publication assumptions and current-vintage limits follow the [shared protocol](../../doc/m3_ml/README.md).

The plan below was recorded on 2026-10-18 before implementation. Its original 2016–2022 annual validation folds and 2023-01-01–2026-10-06 monthly-refit test are historical design context, superseded for current execution. Distribution-derived probabilities and hedging evaluations reused previously seen distribution-test forecasts and are explicitly post-test analyses. Newly trained volatility/classification models selected settings on the historical validation split. Markov regimes were considered but not implemented; continuous volatility models already represent high/low risk states.

## Volatility forecasts

The target is future realized variance `RV(t,h)=sum(r(t+i)^2, i=1..h)` for h1/5/20/60, with WTI daily log returns. Windows containing the negative-price transition on 2020-04-20/21 are excluded from log-target fitting/scoring. Close-to-close variance is a noisy proxy; the original rationale follows Patton (2011) for robust loss comparisons with an unbiased proxy. The forecast is expected RV.

Inputs include log mean squared returns over 1/5/20/60 days; log mean futures Garman–Klass variance over 1/5/20 days after OHLC/nonpositive-bar checks; and log OVX²/VIX² with declared availability lags. Garman–Klass omits overnight gaps and is an input, not the evaluation target.

| Method | Specification/grid |
| --- | --- |
| Historical variance | h times trailing daily variance; window 20/60/250 |
| EWMA | h times daily EWMA; decay 0.94/0.97/0.99 |
| GARCH | Zero-mean GARCH or GJR(1,1), Student-t; analytical multi-step variance sum |
| Raw OVX | h × (OVX/100)² / 252; no fitting |
| Calibrated OVX | Expanding OLS of log RV on log OVX² |
| HAR Ridge | Alpha 0.1/1/10/100; close-only, plus Garman–Klass, then plus OVX/VIX |
| LightGBM | Learning rate .02, minimum child 50, row/column sampling .8; leaves 4/8, trees 200/500; volatility block or block plus 56 shared features |

Log-variance models use Duan smearing from purged/embargoed cross-fit residuals to recover a mean, not simply exponentiate a log prediction and call it a mean. Original selection minimized mean annual QLIKE. Legacy evaluation reports QLIKE `RV/pred - log(RV/pred) - 1`, variance MSE, DM/Newey–West lag h−1, Mincer–Zarnowitz intercept/slope tests and R², years, and volatility regimes. Historical “beats GARCH/option market” rules required lower QLIKE and DM p<.05 versus GARCH/calibrated OVX. These significance rules do not apply to short October evaluation.

## Tail-event probabilities

Fixed log-return thresholds are 3%/7%/15%/25% for h1/5/20/60, defining separate positive and negative events. Fixed symmetric thresholds avoid the strongly asymmetric early-training empirical tails. Distribution probabilities interpolate quantile CDFs (63 levels in the old test, main grid in validation). Direct classifiers use L2 logistic regression C=.01/.1/1/10 on volatility features or LightGBM leaves 4/8, trees 100/300 on volatility plus shared features. Expanding training event frequency is the climatology baseline.

Original classifier selection minimizes annual-fold Brier score. Legacy evaluation includes Brier skill, log loss, AUC, ten-bin reliability, regimes, and DM differences with h−1 lags. The prespecified historical success rule requires lower Brier and p<.05 versus climatology. Current October results remain descriptive.

## Economic-value experiments

Volatility targeting uses `w=min(2, .30/predicted_annual_volatility)` from h1 variance. Fixed exposure uses .30 divided by pre-test training volatility; unscaled w=1 is another baseline. P&L is next-day spot-return exposure minus 5bp times absolute turnover, with 0/10bp sensitivity. Report annual return/volatility, Sharpe, drawdown, 97.5% daily ES, worst day, turnover, and annual deviation from the 30% target. Spot exposure excludes futures rolls and is not a directly tradable strategy.

VaR-constrained hedging considers long producers and short consumers at h20. The 5% loss limit is 10% log return; hedge ratio is `clip(1-.10/predicted_VaR,0,1)`, with producer VaR = −q05 and consumer VaR = q95. Report limit-exceedance rate, average hedge share, post-hedge ES, and adjustments larger than .05. HS, EWMA, GARCH, GARCH-EVT, A-Ridge, A-EVT, linear quantiles, and ensembles are compared. Lower hedge share is desirable only with comparable risk-limit performance. Distribution-based decisions reuse existing forecast evidence.

## Recorded correction

After initial volatility validation and before the old test, 18 h1 zero-return days (0.48%) produced zero RV and infinite QLIKE/extreme log targets. Zero-RV origins were treated as missing for fitting/scoring and validation was rerun. h≥5 was unaffected. This correction and subsequent protocol changes are disclosed rather than retroactively presented as the original design.

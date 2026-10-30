# Oil spread forecasting research plan

## Current protocol and original scope

Current execution follows initial training through 2023-08-31, development through September 2026, final September 30 available-information fitting, and frozen October 1–23 evaluation. Both modes are saved separately; the final project uses daily rolling. Source legs use a common delay no earlier than the same-date WTI publication, explicitly an assumption rather than independently verified Brent/product vintages. New outputs are under `outputs/frozen_october_2026/`.

The original plan was recorded on 2026-10-18 before the spread experiment. Its then-unseen 2023–2026 spread test is now historical evidence and must not be relabeled as a new October blind test. Initial 2012–2022 motivation was Brent–WTI ADF p=.031 versus WTI p=.105, estimated mean-reversion half-life about 34 quoted days, and in-sample h20 deviation-regression R² 8.6% versus zero for WTI. Early structural convergence may explain these in-sample findings.

## Targets and data

Brent–WTI is `Brent-WTI` in USD/barrel from FRED DCOILBRENTEU/DCOILWTICO. The original common-positive-price sample has 3,488 dates, 2012-08-23–2026-10-06. The 3-2-1 crack spread is `(2*42*gasoline+42*heating_oil)/3-WTI`. FRED was unavailable during original acquisition, so equivalent EIA source series `PET.EER_EPMRU_PF4_Y35NY_DPG.D` and `PET.EER_EPD2F_PF4_Y35NY_DPG.D` were used before crack validation. `fetch_products.py` and `data/raw/eia_api/spot_prices/sources.csv` preserve provenance. The three crack legs use New York assessments; Brent–WTI has an asynchronous-assessment limitation.

The original same-close availability assumption is superseded by the current common publication delay. Nonpositive WTI dates are excluded from the spread study, including 2020-04-20; the shared WTI main table still retains that observation. The target is spread change over 1/5/20/60 rows of the spread's common quote grid.

## Models

| Model | Grid/specification |
| --- | --- |
| RW | Zero spread change |
| Drift | Historical mean h-step change |
| OU / AR(1) | `mean + phi^h*(current-mean)`; expanding, 1,500-day, or 750-day fits |
| Moving-average deviation | OLS on spread minus its MA; windows 20/60/120/250 |
| Elastic Net | Alpha/alpha_max 1/.3/.1/.03/.01; l1_ratio .1/.5/.9; spread-only or plus EIA |
| LightGBM | Learning rate .02, minimum child 50, row/column sampling .8; leaves 4/8, trees 100/300; same feature groups |

Spread features are deviations from 20/60/120/250-day means, changes over 1/5/20 days, and 20-day change volatility. EIA inputs are latest and year-on-year changes in crude exports/imports/net imports/production, joined backward by available date. Rolling windows permit adaptation after the 2015 export-policy structural break.

## Historical evaluation design

The original design uses 2016–2022 annual validation folds with quarterly refits, mature labels only, and geometric-mean RMSE/RW selection. The old test runs monthly refits from 2023 through 2026-10-06. Metrics include dollar RMSE/MAE, relative RW, DM with Newey–West lag h−1, nonzero-change directional accuracy/precision/recall/F1, years, and volatility regimes. The original success rule requires lower RMSE and DM p<.05; current short October output does not claim significance.

The illustrative h20 trading rule holds +1 spread above a predicted USD .25 change, −1 below −.25, otherwise zero. P&L is position times next-day spread change minus USD .05 per unit position change. h5 is supplementary. Report cumulative P&L, annual Sharpe, drawdown, win rate, trades, holding period, and per-trade profit. Spot spreads are not directly executable rolled-futures strategies. Post-test timing diagnostics are separately disclosed in the results.

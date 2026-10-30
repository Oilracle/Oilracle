"""Mode-separated accuracy summaries, including unreached horizon counts."""
import numpy as np
import pandas as pd


def summarize(forecasts):
    """Score only realized targets; retain n=0 for every emitted method/horizon.

    Price errors are USD/barrel; return RMSE uses the published origin anchor.
    Random-walk comparisons are paired by mode, origin and target horizon.
    """
    keys = (['mode'] if 'mode' in forecasts else []) + ['horizon', 'model']
    rows = []
    for key, emitted in forecasts.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key))
        g = emitted.dropna(subset=['forecast_log_return', 'actual_log_return'])
        err = g['forecast_log_return'] - g['actual_log_return']
        row.update(n=len(g), rmse=np.sqrt((err ** 2).mean()), mae=err.abs().mean())
        moved = (g['forecast_log_return'] != 0) & (g['actual_log_return'] != 0)
        row['direction'] = (np.sign(g.loc[moved, 'forecast_log_return']) ==
                            np.sign(g.loc[moved, 'actual_log_return'])).mean()
        reference = forecasts[forecasts['model'].eq('random_walk') &
                              forecasts['horizon'].eq(row['horizon'])]
        if 'mode' in row:
            reference = reference[reference['mode'].eq(row['mode'])]
        reference = reference.set_index('origin_date')['actual_log_return'].reindex(g['origin_date'])
        denominator = np.sqrt((reference ** 2).mean())
        row['rmse_vs_rw'] = row['rmse'] / denominator if denominator > 0 else np.nan
        if {'forecast', 'actual'} <= set(g):
            row['mae_usd'] = (g['forecast'] - g['actual']).abs().mean()
            row['mape_pct'] = 100 * ((g['forecast'] - g['actual']).abs() / g['actual'].abs()).mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index(keys).sort_index().round(4)

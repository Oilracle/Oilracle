"""Import the real sequence protocol output; never train a duplicate aligned model.

Run m3_ml/lstm/src/oil_lstm.py under its frozen October protocol first, then:
    uv run python -m m3_ml.compare.lstm_aligned
"""
import pandas as pd

from .common import HORIZONS, OUT, ROOT, forecast_horizons, forecast_origins, load_prices, read_forecasts

NAMES = {'LSTM': 'lstm', 'Persistence': 'lstm_persistence'}


def main():
    source = ROOT / 'm3_ml/lstm/outputs/frozen_october_2026/test_predictions.csv'
    frame = read_forecasts(source, ['forecast', 'actual'])
    if not set(NAMES).issubset(set(frame['method'])):
        raise ValueError(f'{source}: missing sequence methods {sorted(set(NAMES) - set(frame["method"]))}')
    frame = frame[frame['method'].isin(NAMES)].copy()
    table, _ = load_prices()
    fixed_horizons = forecast_horizons(table, int(forecast_origins(table, 'fixed_origin')[0]), 'fixed_origin')
    frame = frame[(frame['mode'].eq('fixed_origin') & frame['horizon'].isin(fixed_horizons))
                  | (frame['mode'].eq('daily_update') & frame['horizon'].isin(HORIZONS))]
    frame['method'] = frame['method'].map(NAMES)
    if not frame.loc[frame['method'].eq('lstm_persistence'), 'forecast'].sub(
            frame.loc[frame['method'].eq('lstm_persistence'), 'origin_price']).abs().lt(1e-6).all():
        raise ValueError('Sequence persistence does not equal its observable anchor')
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / 'lstm_forecasts.csv', index=False, date_format='%Y-%m-%d')
    print('imported real frozen sequence forecasts:', OUT / 'lstm_forecasts.csv', len(frame))


if __name__ == '__main__':
    main()

Held out of data/raw on 2026-10-29. Not deleted.

These files are not used by any model, test, or the 1991 long table.
The local builder still reads them from here through scripts/data_layout.py.

Restore by moving each file back to the path after held-out/:

  data/raw/cftc/wti_positions_weekly.csv
  data/raw/fred/trade_weighted_dollar_daily.csv
  data/raw/futures/brent_futures_daily.csv
  data/raw/futures/energy_select_sector_etf_daily.csv
  data/raw/futures/gold_futures_daily.csv
  data/raw/futures/natural_gas_futures_daily.csv
  data/raw/futures/wti_nearby_futures_daily.csv
  data/raw/eia_api/cushing_crude_stocks_weekly.csv
  data/raw/eia_api/refinery_utilization_weekly.csv

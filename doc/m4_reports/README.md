# Final forecast outputs

The official policy is routed by [`m4_reports/final_strategy_manifest.json`](../../m4_reports/final_strategy_manifest.json). It combines time-series regression, seasonality, LSTM, point forecasts and distribution/risk forecasts using the historical-development protocol.

- [Final methodology and results](../m3_ml/FUSION_V6.md)
- [Historical forecast charts](../../m4_reports/plot_historical_forecast.py)
- [October forecast charts](../../m4_reports/plot_final_forecast.py)
- [VECM versus fusion diagnosis](../../m4_reports/diagnose_vecm_fusion.py)

From the repository root:

```powershell
uv run python -B -m m4_reports.plot_final_forecast
uv run python -B -m m4_reports.plot_historical_forecast
```

These commands use saved outputs and do not retrain or tune on October. Generated charts are written beneath the selected forecast output directories. The internal model identifier identifies the final accepted implementation, not an additional product.

Course presentation slides, member information, registration details, production source decks, video assets and the standalone speaker guide are outside this public source release. They remain in the separate course submission and private working materials.

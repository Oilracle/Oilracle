# Residual correction

The current `run.py` follows the common September 2023–September 2026 development window, September final fit, and frozen October evaluation. This module's baseline is its own ARIMA-plus-calendar regression. A later experiment applying ML correction to the actual member VECM/STL is documented in [MEMBER_ML_PROBE.md](../MEMBER_ML_PROBE.md); that historical-only probe did not replace the official fusion policy or evaluate October.

The residual experiment asks whether fundamentals and market features explain errors left by a time-series/calendar baseline: `final forecast = baseline + g(X)`.

The original baseline fits daily log returns on month/weekday effects, refitting every 21 quoted days, then fits drift-free ARIMA(0,1,1) to seasonally adjusted log prices. MA coefficients refit on the same historical schedule and innovations update daily. Horizon returns combine ARIMA's adjusted return with known future calendar effects, never future prices.

Elastic Net, Random Forest, and LightGBM use the same features as direct prediction. The original experiment selected hyperparameters on 2020–2022 and reported 2023 onward after seeing direct-model results; those legacy results are not blind tests. Current frozen-protocol outputs must be distinguished from that earlier split.

Risk forecasts add the past baseline-residual empirical distribution to the baseline or corrected center. Elastic Net can predict absolute residual magnitude to scale the distribution; EWMA decay and distribution settings inherit the distribution study's historical selection.

```powershell
uv run python -m m3_ml.residual.run
uv run python -m m3_ml.residual.analyze
```

See [RESULTS.md](../../../m3_ml/residual/RESULTS.md). `m3_ml/tests/test_residual.py` checks that baseline forecasts do not consume prices after the origin.

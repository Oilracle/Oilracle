# Official final forecast

The only final product is daily rolling `short_bias_v6`, selected by [the official manifest](../../m4_reports/final_strategy_manifest.json). Read [FUSION_V6.md](FUSION_V6.md) for methods, evidence, and limitations.

Parameters and risk settings are selected using information available before October, then frozen at the September information cutoff. Published inputs update daily. The product integrates M1 time-series forecasts, M2 seasonality, LSTM, ML point forecasts, predictive distributions, and tail risk. It does not require a minimum weight for every component.

Version identifiers are internal provenance. Earlier policies, fixed-origin paths, and exploratory studies remain available to explain development; they are not additional final recommendations. [The original strategy record](FINAL_STRATEGY_V1_ARCHIVE.md) is explicitly archived. October is a short retrospective evaluation and must not be used to select weights or calibration parameters.

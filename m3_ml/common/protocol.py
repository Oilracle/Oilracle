"""Project-wide temporal protocol for historical validation and frozen final forecasts."""

TRAIN_END = '2023-08-31'
VALIDATION_START = '2023-09-01'
VALIDATION_END = '2026-09-30'
FIT_END = '2026-09-30'
TEST_START = '2026-10-01'
TEST_END = '2026-10-23'
HORIZONS = (1, 5, 20, 60)
MODES = ('fixed_origin', 'daily_update')
PROTOCOL_ID = 'qf5210-frozen-october-2026'

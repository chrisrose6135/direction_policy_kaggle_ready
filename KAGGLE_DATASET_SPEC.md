# Raw forex Kaggle dataset specification

Use this when asking Kaggle to create or attach the raw data dataset.

## Dataset purpose

This dataset should contain raw M5 OHLC forex data for the symbols to train, for example `EURUSD`, `GBPUSD`, and `USDJPY`. The project will generate engineered features and future TP/SL direction labels inside the notebook.

## Preferred file structure

One CSV per symbol/timeframe:

```text
EURUSD_M5.csv
GBPUSD_M5.csv
USDJPY_M5.csv
```

## Preferred CSV schema

```text
time,open,high,low,close,tick_volume,spread
```

Where:

- `time` is UTC or parseable datetime.
- `open`, `high`, `low`, `close` are OHLC prices.
- `tick_volume` is optional but preferred.
- `spread` is optional but preferred. It should be broker spread points if available.

## Alternative combined CSV schema

One combined file is also supported if it has a `symbol` column:

```text
symbol,time,open,high,low,close,tick_volume,spread
```

Then run the Kaggle wrapper with:

```bash
--combined-csv /kaggle/input/<dataset>/<combined-file>.csv
```

## Column aliases accepted

The helper can map common aliases such as `DateTime`, `timestamp`, `Open`, `High`, `Low`, `Close`, `Volume`, `tickvol`, `Spread`, and `Symbol`.

## Timeframe

The current project configs expect M5. If the raw data is not M5, generate or resample to M5 before training.

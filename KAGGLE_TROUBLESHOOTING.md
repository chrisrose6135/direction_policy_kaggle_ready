# Kaggle troubleshooting

If a notebook cell only shows `CalledProcessError`, rerun the pipeline with the patched `kaggle_run_pipeline.py`. It now prints each inner command stdout/stderr before stopping.

## Check Kaggle input paths

Kaggle input paths are case-sensitive and usually lower-case. For example, a dataset called `MT5_Dataset` may appear as:

```python
!find /kaggle/input -maxdepth 3 -type f | head -100
```

Use the dataset root or the folder containing the CSVs:

```python
RAW_INPUT_DIR = "/kaggle/input/mt5-dataset"
# or
RAW_INPUT_DIR = "/kaggle/input/mt5-dataset/raw"
```

## Common causes

1. The raw input path does not exist.
2. The raw CSV files are not named like `EURUSD_M5.csv`, or the symbol requested does not match the file name.
3. The date range does not overlap the raw CSV date range.
4. The raw time column was not parsed correctly.
5. The generated direction dataset is empty/header-only.

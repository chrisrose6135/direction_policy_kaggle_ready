# Kaggle-ready direction-policy project

This package is designed to run the **data-preparation, labelled dataset generation, training, replay, and five-model comparison** on Kaggle.

Kaggle is suitable for:

- raw CSV -> processed feature CSV generation
- processed feature CSV -> labelled direction dataset generation
- training one or more of the five neural architectures
- replay-each-epoch model selection
- architecture grid comparison

Kaggle is **not** suitable for live/demo MT5 trading, because `MetaTrader5` requires a local Windows MT5 terminal. The MT5/live files are left in the project for local use, but the Kaggle requirements intentionally do not install `MetaTrader5`.

---

## Expected Kaggle raw-data dataset

Create/upload a Kaggle dataset containing either:

### Option A: one CSV per symbol

Recommended filenames:

```text
EURUSD_M5.csv
GBPUSD_M5.csv
USDJPY_M5.csv
```

### Option B: one combined CSV

A single CSV containing a `symbol` column.

### Accepted raw CSV columns

The raw-prep helper accepts common aliases, but the safest schema is:

```text
time, open, high, low, close, tick_volume, spread
```

`spread` should be broker spread points if available. If it is missing, the project will fall back to the configured default spread settings.

---

## Quick notebook setup

In a Kaggle notebook, add two datasets:

1. this project ZIP as a code/input dataset
2. your raw M5 forex CSV dataset

Then copy/unzip the project into `/kaggle/working` and install the Kaggle requirements:

```bash
!unzip -q /kaggle/input/<project-dataset>/direction_policy_kaggle_ready.zip -d /kaggle/working/project
%cd /kaggle/working/project
!pip install -r requirements_kaggle.txt
```

If you upload the project as a folder rather than a zip, copy it instead:

```bash
!cp -r /kaggle/input/<project-dataset> /kaggle/working/project
%cd /kaggle/working/project
!pip install -r requirements_kaggle.txt
```

---

## End-to-end run from raw data

This command does all of the following:

1. standardises Kaggle raw CSVs into `data/raw/SYMBOL_M5.csv`
2. builds feature CSVs in `data/processed_m5`
3. builds labelled direction CSVs in `data/direction`
4. trains all five architectures with replay-each-epoch selection

```bash
!python kaggle_run_pipeline.py \
  --raw-input-dir /kaggle/input/<your-raw-forex-dataset> \
  --symbols EURUSD \
  --timeframe M5 \
  --train-start 2024-01-01 \
  --train-end 2025-03-01 \
  --replay-start 2025-03-01 \
  --replay-end 2025-06-01 \
  --epochs 50 \
  --batch-size 512 \
  --device cuda \
  --mode train-all
```

For a quick smoke test:

```bash
!python kaggle_run_pipeline.py \
  --raw-input-dir /kaggle/input/<your-raw-forex-dataset> \
  --symbols EURUSD \
  --timeframe M5 \
  --raw-max-rows-per-symbol 120000 \
  --train-start 2024-01-01 \
  --train-end 2024-06-01 \
  --replay-start 2024-06-01 \
  --replay-end 2024-08-01 \
  --epochs 3 \
  --batch-size 256 \
  --train-max-rows 30000 \
  --device cuda \
  --mode train-one
```

---

## Train a single architecture

```bash
!python kaggle_run_pipeline.py \
  --raw-input-dir /kaggle/input/<your-raw-forex-dataset> \
  --symbols EURUSD \
  --configs config/direction_settings_tcn.yaml \
  --train-start 2024-01-01 \
  --train-end 2025-03-01 \
  --replay-start 2025-03-01 \
  --replay-end 2025-06-01 \
  --epochs 50 \
  --batch-size 512 \
  --device cuda \
  --mode train-one
```

Available configs:

```text
config/direction_settings_residual_mlp.yaml
config/direction_settings_tcn.yaml
config/direction_settings_inception_time.yaml
config/direction_settings_small_transformer.yaml
config/direction_settings_mixture_of_experts.yaml
```

---

## Run the architecture grid

```bash
!python kaggle_run_pipeline.py \
  --raw-input-dir /kaggle/input/<your-raw-forex-dataset> \
  --symbols EURUSD \
  --train-start 2024-01-01 \
  --train-end 2025-03-01 \
  --replay-start 2025-03-01 \
  --replay-end 2025-06-01 \
  --epochs 50 \
  --batch-size 512 \
  --device cuda \
  --mode grid \
  --grid config/direction_param_grid_model_architectures.yaml
```

---

## Prepare only, then train later

```bash
!python kaggle_run_pipeline.py \
  --raw-input-dir /kaggle/input/<your-raw-forex-dataset> \
  --symbols EURUSD GBPUSD USDJPY \
  --train-start 2024-01-01 \
  --replay-end 2025-06-01 \
  --mode prepare-only
```

Then train using already prepared data:

```bash
!python kaggle_run_pipeline.py \
  --skip-raw-copy \
  --skip-feature-prep \
  --skip-direction-prep \
  --symbols EURUSD \
  --configs config/direction_settings_tcn.yaml \
  --train-start 2024-01-01 \
  --train-end 2025-03-01 \
  --replay-start 2025-03-01 \
  --replay-end 2025-06-01 \
  --epochs 50 \
  --batch-size 512 \
  --device cuda \
  --mode train-one
```

---

## Download outputs

Important outputs are written under the Kaggle working directory:

```text
models/
logs/
data/direction/
config/generated_spread_risk.yaml
```

To create one downloadable archive:

```bash
!zip -r /kaggle/working/direction_policy_outputs.zip models logs data/direction config/generated_spread_risk.yaml
```

---

## Notes

- Use GPU acceleration in the Kaggle notebook settings for Transformer/TCN/InceptionTime/MoE runs.
- Use `--device cpu` if no GPU is available.
- Use `--raw-max-rows-per-symbol`, `--feature-max-rows`, `--direction-max-rows`, and `--train-max-rows` only for smoke tests. Remove them for real training.
- The wrapper sets `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` to reduce CPU thread stalls during repeated training/replay runs.


## Project dataset already unzipped by Kaggle

If you uploaded `direction_policy_kaggle_ready.zip` as a Kaggle data source, Kaggle may automatically extract it. In that case there may be no `.zip` file inside `/kaggle/input/...`. The notebook now supports this. Set:

```python
PROJECT_INPUT = '/kaggle/input/<your-project-dataset-name>'
```

or leave `PROJECT_INPUT = None` and the notebook will scan `/kaggle/input` for a folder containing `kaggle_run_pipeline.py`, `requirements_kaggle.txt`, `src/`, and `config/`.

The notebook copies the read-only Kaggle input project into `/kaggle/working/project` before training, because Kaggle input datasets cannot be written to.


## Troubleshooting note

Kaggle input paths are case-sensitive and usually lower-case/hyphenated. If `/kaggle/input/MT5_Dataset/raw` fails, run `!find /kaggle/input -maxdepth 3 -type f | head -100` and use the exact path printed by Kaggle. The patched `kaggle_run_pipeline.py` also prints a brief path listing and the stderr/stdout for failing inner commands.

## Strong-setup label update

The Kaggle package now defaults the architecture configs to `labels.method: strong_setup_v1`. This produces side-specific, event-based BUY/SELL setup labels with hard negatives and sampled background no-trade rows. It is designed to reduce noisy every-bar labelling and give the multi-model comparison a cleaner supervised problem.

Read `STRONG_SETUP_LABELS.md` before changing the label thresholds.

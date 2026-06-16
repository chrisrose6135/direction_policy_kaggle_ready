# Direction Policy Model Training Commands

This project now supports five neural model families through the same training, replay, grid-search, live/demo, and testing entry points. The model is selected by the `model.architecture` key in the YAML config.

Supported architecture values:

| Model family | Config file | `model.architecture` |
|---|---|---|
| Residual tabular MLP | `config/direction_settings_residual_mlp.yaml` | `residual_mlp_gate_direction_v1` |
| Temporal CNN / TCN | `config/direction_settings_tcn.yaml` | `hierarchical_tcn_edge_v1` |
| Small Transformer encoder | `config/direction_settings_small_transformer.yaml` | `small_transformer_gate_direction_v1` |
| InceptionTime-style CNN | `config/direction_settings_inception_time.yaml` | `inception_time_gate_direction_v1` |
| Mixture of Experts | `config/direction_settings_mixture_of_experts.yaml` | `mixture_of_experts_direction_v1` |

Each architecture-specific config writes to a separate model/log directory, so running them one after another will not overwrite the previous model:

```text
models/residual_mlp/
models/tcn/
models/small_transformer/
models/inception_time/
models/mixture_of_experts/
```

The default `config/direction_settings_generic_multisymbol_31_symbols.yaml` remains a TCN-based config, but the five explicit configs are clearer for architecture comparisons.

## 1. Residual MLP

This is the fastest neural sanity check. It uses the current feature row instead of learning a temporal sequence encoder.

Standard training:

```bash
python -m src.train_direction_policy ^
  --config config/direction_settings_residual_mlp.yaml ^
  --symbols EURUSD
```

Train and replay every epoch:

```bash
python -m src.train_direction_policy_replay_each_epoch ^
  --config config/direction_settings_residual_mlp.yaml ^
  --symbols EURUSD ^
  --date-start 2024-01-01 ^
  --date-end 2025-03-01 ^
  --replay-start 2025-03-01 ^
  --replay-end 2025-06-01 ^
  --epochs 50 ^
  --batch-size 512
```

Replay saved epoch checkpoints:

```bash
python -m src.replay_direction_epochs ^
  --config config/direction_settings_residual_mlp.yaml ^
  --symbols EURUSD ^
  --eval-start 2025-03-01 ^
  --eval-end 2025-06-01 ^
  --verbose
```

## 2. Temporal CNN / TCN

This is the current sequence baseline.

Standard training:

```bash
python -m src.train_direction_policy ^
  --config config/direction_settings_tcn.yaml ^
  --symbols EURUSD
```

Train and replay every epoch:

```bash
python -m src.train_direction_policy_replay_each_epoch ^
  --config config/direction_settings_tcn.yaml ^
  --symbols EURUSD ^
  --date-start 2024-01-01 ^
  --date-end 2025-03-01 ^
  --replay-start 2025-03-01 ^
  --replay-end 2025-06-01 ^
  --epochs 50 ^
  --batch-size 512
```

Replay saved epoch checkpoints:

```bash
python -m src.replay_direction_epochs ^
  --config config/direction_settings_tcn.yaml ^
  --symbols EURUSD ^
  --eval-start 2025-03-01 ^
  --eval-end 2025-06-01 ^
  --verbose
```

## 3. Small Transformer

This tests whether self-attention over the recent M5 window is better than convolutional sequence processing.

Standard training:

```bash
python -m src.train_direction_policy ^
  --config config/direction_settings_small_transformer.yaml ^
  --symbols EURUSD
```

Train and replay every epoch:

```bash
python -m src.train_direction_policy_replay_each_epoch ^
  --config config/direction_settings_small_transformer.yaml ^
  --symbols EURUSD ^
  --date-start 2024-01-01 ^
  --date-end 2025-03-01 ^
  --replay-start 2025-03-01 ^
  --replay-end 2025-06-01 ^
  --epochs 50 ^
  --batch-size 512
```

Replay saved epoch checkpoints:

```bash
python -m src.replay_direction_epochs ^
  --config config/direction_settings_small_transformer.yaml ^
  --symbols EURUSD ^
  --eval-start 2025-03-01 ^
  --eval-end 2025-06-01 ^
  --verbose
```

## 4. InceptionTime-style CNN

This tests multiple convolution kernel widths in parallel, which can pick up short and longer setup shapes in the same model.

Standard training:

```bash
python -m src.train_direction_policy ^
  --config config/direction_settings_inception_time.yaml ^
  --symbols EURUSD
```

Train and replay every epoch:

```bash
python -m src.train_direction_policy_replay_each_epoch ^
  --config config/direction_settings_inception_time.yaml ^
  --symbols EURUSD ^
  --date-start 2024-01-01 ^
  --date-end 2025-03-01 ^
  --replay-start 2025-03-01 ^
  --replay-end 2025-06-01 ^
  --epochs 50 ^
  --batch-size 512
```

Replay saved epoch checkpoints:

```bash
python -m src.replay_direction_epochs ^
  --config config/direction_settings_inception_time.yaml ^
  --symbols EURUSD ^
  --eval-start 2025-03-01 ^
  --eval-end 2025-06-01 ^
  --verbose
```

## 5. Mixture of Experts

This uses analytic signal features as router/context and learns separate expert representations for different setup types.

Standard training:

```bash
python -m src.train_direction_policy ^
  --config config/direction_settings_mixture_of_experts.yaml ^
  --symbols EURUSD
```

Train and replay every epoch:

```bash
python -m src.train_direction_policy_replay_each_epoch ^
  --config config/direction_settings_mixture_of_experts.yaml ^
  --symbols EURUSD ^
  --date-start 2024-01-01 ^
  --date-end 2025-03-01 ^
  --replay-start 2025-03-01 ^
  --replay-end 2025-06-01 ^
  --epochs 50 ^
  --batch-size 512
```

Replay saved epoch checkpoints:

```bash
python -m src.replay_direction_epochs ^
  --config config/direction_settings_mixture_of_experts.yaml ^
  --symbols EURUSD ^
  --eval-start 2025-03-01 ^
  --eval-end 2025-06-01 ^
  --verbose
```

## Compare all five with the optimiser

The bundled grid file runs one configuration for each architecture. Use this when you want the optimiser to perform a direct architecture comparison while preserving the usual train/replay workflow.

```bash
python -m src.optimise_direction_training_params ^
  --config config/direction_settings_generic_multisymbol_31_symbols.yaml ^
  --symbols EURUSD ^
  --grid config/direction_param_grid_model_architectures.yaml ^
  --train-start 2024-01-01 ^
  --train-end 2025-03-01 ^
  --replay-start 2025-03-01 ^
  --replay-end 2025-06-01 ^
  --epochs 50 ^
  --batch-size 512 ^
  --threads 1
```

## Parallel multi-symbol training with replay

Use the parallel launcher when you want one process per symbol. This keeps the same arguments as the replay-each-epoch path and adds `--parallel-symbols`.

```bash
python -m src.train_direction_policy_replay_parallel ^
  --config config/direction_settings_tcn.yaml ^
  --symbols EURUSD GBPUSD USDJPY ^
  --parallel-symbols 2 ^
  --date-start 2024-01-01 ^
  --date-end 2025-03-01 ^
  --replay-start 2025-03-01 ^
  --replay-end 2025-06-01 ^
  --epochs 50 ^
  --batch-size 512
```

## Standalone replay of a best saved model

```bash
python -m src.test_saved_direction_policy ^
  --config config/direction_settings_tcn.yaml ^
  --symbol EURUSD ^
  --eval-start 2025-03-01 ^
  --eval-end 2025-06-01
```

## Live/demo trading

The live/demo file uses the same config architecture and the feature list saved during training. Use the matching config for the model you trained.

```bash
python -m src.live_direction_policy ^
  --config config/direction_settings_tcn.yaml ^
  --mode demo ^
  --data-source mt5 ^
  --poll-seconds 20
```

For a different architecture, switch only the config path, for example:

```bash
python -m src.live_direction_policy ^
  --config config/direction_settings_mixture_of_experts.yaml ^
  --mode demo ^
  --data-source mt5 ^
  --poll-seconds 20
```

---

## Strong-setup labelling mode

This package now includes `labels.method: strong_setup_v1`. It creates an event-based direction dataset rather than supervising every candle. Non-event rows are written with `direction_target = -1` and are ignored as sequence endpoints while still being available as historical context.

Recommended preparation command:

```bash
python -m src.prepare_direction_dataset \
  --config config/direction_settings_residual_mlp.yaml \
  --symbols EURUSD \
  --workers 2 \
  --date-start 2021-01-01 \
  --date-end 2025-09-01 \
  --out-dir data/direction
```

Then train the model architectures using the pregenerated strong-setup CSVs:

```bash
python -m src.train_all_direction_models \
  --symbols EURUSD \
  --epochs 50 \
  --batch-size 512 \
  --device cpu
```

See `STRONG_SETUP_LABELS.md` for the label rules and tuning parameters.

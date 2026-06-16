# Strong setup labelling for the multi-model direction project

This project now supports a leakage-safe `strong_setup_v1` labelling mode designed for M5 FX direction modelling.

The old labelling style treated most bars as supervised BUY / SELL / NO_TRADE endpoints. That creates a very noisy problem: many neighbouring bars in the same move become positives, and most negatives are uninformative background bars.

The new mode changes the training problem to side-specific event ranking:

- simulate both BUY and SELL over the configured future horizon using the existing live-style bid/ask TP/SL logic;
- keep only clean, high-margin BUY/SELL setup endpoints as positive labels;
- keep hard negatives from failed analytic setups and near-positive false setups;
- optionally keep a controlled sample of background NO_TRADE rows;
- mark all other rows as `IGNORE` (`direction_target = -1`) so they can still provide historical sequence context but are not supervised sequence endpoints.

The direction class convention is unchanged:

```text
0 = SELL
1 = NO_TRADE
2 = BUY
-1 = IGNORE / not a supervised endpoint
```

## Config block

The architecture-specific Kaggle configs now include this block under `labels:`:

```yaml
labels:
  method: strong_setup_v1

  horizon_bars: 24
  take_profit_pips: 8.0
  stop_loss_pips: 5.0
  use_live_bid_ask_simulation: true
  entry_on_next_bar_open: true
  same_bar_tp_sl_policy: stop_first

  strong_setup:
    output_mode: event_based
    random_seed: 43

    positive_defaults:
      require_tp_before_sl: true
      min_net_pips: 3.0
      min_mfe_pips: 10.0
      max_mae_pips: 4.0
      min_side_edge_pips: 2.0
      allow_clean_without_analytic: true

    positive:
      buy:
        min_net_pips: 3.0
        min_mfe_pips: 10.0
        max_mae_pips: 4.0
        min_side_edge_pips: 2.0
      sell:
        min_net_pips: 3.0
        min_mfe_pips: 10.0
        max_mae_pips: 4.0
        min_side_edge_pips: 1.5

    hard_negatives:
      enabled: true
      analytic_score_min: 4.0
      near_positive_mfe_fraction: 0.75
      max_ratio_to_positive: 2.0

    background_no_trade:
      enabled: true
      ratio_to_positive: 2.0
```

## What the positive rules mean

A BUY positive requires:

- BUY TP hit before SL;
- realised BUY net pips is at least `min_net_pips`;
- BUY maximum favourable excursion is at least `min_mfe_pips`;
- BUY maximum adverse excursion is no more than `max_mae_pips`;
- BUY has enough advantage over the simultaneous SELL candidate.

SELL uses the same logic with SELL-specific thresholds.

The `sig_*` analytic features are also converted to causal BUY/SELL setup scores. Those scores are used for hard-negative selection and may be used as stricter positive filters by setting `min_analytic_score` and `allow_clean_without_analytic: false`.

## Why event-based output is different

With `output_mode: event_based`, the dataset does not supervise every bar. It keeps:

- positive setup endpoints;
- hard negative setup endpoints;
- sampled background NO_TRADE endpoints;
- ignored rows as context only.

This is intended to make the model learn **which setups are worth trading**, not just classify every M5 candle.

## Kaggle commands

Prepare the strong-setup dataset:

```bash
python -m src.prepare_direction_dataset \
  --config config/direction_settings_residual_mlp.yaml \
  --symbols EURUSD \
  --workers 2 \
  --date-start 2021-01-01 \
  --date-end 2025-09-01 \
  --out-dir data/direction
```

Train all model architectures against the same pregenerated strong-setup data:

```bash
python -m src.train_all_direction_models \
  --symbols EURUSD \
  --epochs 50 \
  --batch-size 512 \
  --device cpu
```

Or train one architecture with replay-each-epoch:

```bash
python -m src.train_direction_policy_replay_each_epoch \
  --config config/direction_settings_residual_mlp.yaml \
  --symbols EURUSD \
  --epochs 50 \
  --batch-size 512 \
  --model-selection-metric replay_score \
  --date-start 2021-01-01 \
  --date-end 2025-01-01 \
  --replay-start 2025-03-01 \
  --replay-end 2025-09-01 \
  --device cpu
```

## Tuning guidance

More selective positives:

```yaml
min_mfe_pips: 12.0
max_mae_pips: 3.5
min_side_edge_pips: 2.5
```

More training endpoints:

```yaml
hard_negatives:
  max_ratio_to_positive: 3.0
background_no_trade:
  ratio_to_positive: 3.0
```

More aggressive positives:

```yaml
min_mfe_pips: 9.0
max_mae_pips: 4.5
min_side_edge_pips: 1.5
```

For the current EURUSD M5 diagnostics, start with the default strong-setup block, then tune BUY and SELL separately.

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIGS = [
    'config/direction_settings_residual_mlp.yaml',
    'config/direction_settings_tcn.yaml',
    'config/direction_settings_inception_time.yaml',
    'config/direction_settings_small_transformer.yaml',
    'config/direction_settings_mixture_of_experts.yaml',
]


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    print('\n$ ' + ' '.join(map(str, cmd)), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def add_optional(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def main() -> None:
    p = argparse.ArgumentParser(description='Kaggle end-to-end raw-data -> features -> labels -> train/replay pipeline.')
    p.add_argument('--raw-input-dir', default=None, help='Kaggle raw data directory, e.g. /kaggle/input/forex-m5-raw')
    p.add_argument('--combined-csv', default=None, help='Optional combined raw CSV with a symbol column.')
    p.add_argument('--symbols', nargs='+', default=['EURUSD'])
    p.add_argument('--timeframe', default='M5')
    p.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS, help='Model configs to train. Default: all five neural architectures.')
    p.add_argument('--mode', choices=['prepare-only', 'train-one', 'train-all', 'grid'], default='train-all')
    p.add_argument('--grid', default='config/direction_param_grid_model_architectures.yaml')
    p.add_argument('--train-start', default=None)
    p.add_argument('--train-end', default=None)
    p.add_argument('--replay-start', default=None)
    p.add_argument('--replay-end', default=None)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--learning-rate', type=float, default=None)
    p.add_argument('--device', default=None, help='cuda, cpu, or leave blank for script default.')
    p.add_argument('--raw-max-rows-per-symbol', type=int, default=None, help='Optional tail rows per symbol copied from Kaggle raw input.')
    p.add_argument('--feature-max-rows', type=int, default=None, help='Optional tail rows per symbol during feature prep.')
    p.add_argument('--direction-max-rows', type=int, default=None, help='Optional rows during labelled direction data generation.')
    p.add_argument('--train-max-rows', type=int, default=None, help='Optional rows during training load. Useful for smoke tests.')
    p.add_argument('--prepare-workers', type=int, default=2)
    p.add_argument('--force-raw', action='store_true')
    p.add_argument('--force-features', action='store_true')
    p.add_argument('--skip-raw-copy', action='store_true')
    p.add_argument('--skip-feature-prep', action='store_true')
    p.add_argument('--skip-direction-prep', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

    symbols = [s.upper() for s in args.symbols]
    first_config = args.configs[0]

    if not args.skip_raw_copy:
        if not args.raw_input_dir and not args.combined_csv:
            raise SystemExit('Provide --raw-input-dir or --combined-csv, or use --skip-raw-copy if data/raw is already populated.')
        cmd = [
            sys.executable, '-m', 'src.kaggle_prepare_raw_data',
            '--input-dir', args.raw_input_dir or str(Path(args.combined_csv).parent),
            '--output-dir', 'data/raw',
            '--timeframe', args.timeframe,
            '--symbols', *symbols,
        ]
        add_optional(cmd, '--combined-csv', args.combined_csv)
        add_optional(cmd, '--max-rows-per-symbol', args.raw_max_rows_per_symbol)
        if args.force_raw:
            cmd.append('--force')
        run(cmd, dry_run=args.dry_run)

    if not args.skip_feature_prep:
        cmd = [
            sys.executable, '-m', 'src.prepare_mt5_data',
            '--config', first_config,
            '--symbols', *symbols,
            '--timeframe', args.timeframe,
            '--workers', str(args.prepare_workers),
        ]
        add_optional(cmd, '--max-rows', args.feature_max_rows)
        if args.force_features:
            cmd.append('--force')
        run(cmd, dry_run=args.dry_run)

    if not args.skip_direction_prep:
        cmd = [
            sys.executable, '-m', 'src.prepare_direction_dataset',
            '--config', first_config,
            '--symbols', *symbols,
            '--workers', str(args.prepare_workers),
        ]
        add_optional(cmd, '--date-start', args.train_start)
        # The labelled dataset needs enough tail data for replay too. If a replay end
        # is supplied, generate labels up to replay_end; otherwise use train_end.
        add_optional(cmd, '--date-end', args.replay_end or args.train_end)
        add_optional(cmd, '--max-rows', args.direction_max_rows)
        run(cmd, dry_run=args.dry_run)

    if args.mode == 'prepare-only':
        print('\nPreparation complete. Direction CSVs should now be in data/direction/.')
        return

    if args.mode == 'grid':
        cmd = [
            sys.executable, '-m', 'src.optimise_direction_training_params',
            '--config', first_config,
            '--symbols', *symbols,
            '--grid', args.grid,
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--threads', '1',
        ]
        add_optional(cmd, '--train-start', args.train_start)
        add_optional(cmd, '--train-end', args.train_end)
        add_optional(cmd, '--replay-start', args.replay_start)
        add_optional(cmd, '--replay-end', args.replay_end)
        add_optional(cmd, '--learning-rate', args.learning_rate)
        run(cmd, dry_run=args.dry_run)
        return

    configs = args.configs[:1] if args.mode == 'train-one' else args.configs
    for config in configs:
        cmd = [
            sys.executable, '-m', 'src.train_direction_policy_replay_each_epoch',
            '--config', config,
            '--symbols', *symbols,
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--model-selection-metric', 'replay_score',
        ]
        add_optional(cmd, '--date-start', args.train_start)
        add_optional(cmd, '--date-end', args.train_end)
        add_optional(cmd, '--replay-start', args.replay_start)
        add_optional(cmd, '--replay-end', args.replay_end)
        add_optional(cmd, '--max-rows', args.train_max_rows)
        add_optional(cmd, '--learning-rate', args.learning_rate)
        add_optional(cmd, '--device', args.device)
        run(cmd, dry_run=args.dry_run)

    print('\nKaggle pipeline complete. Download /kaggle/working outputs: models/, logs/, data/direction/, config/generated_spread_risk.yaml')


if __name__ == '__main__':
    main()

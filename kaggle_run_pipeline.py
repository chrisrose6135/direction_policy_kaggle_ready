from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _list_dir_brief(path: Path, max_items: int = 80) -> None:
    try:
        if not path.exists():
            print(f'[path check] {path} does not exist', flush=True)
            return
        print(f'[path check] Listing {path}:', flush=True)
        count = 0
        for child in sorted(path.rglob('*')):
            if count >= max_items:
                print(f'  ... truncated after {max_items} entries', flush=True)
                break
            try:
                rel = child.relative_to(path)
            except Exception:
                rel = child
            suffix = '/' if child.is_dir() else ''
            print(f'  {rel}{suffix}', flush=True)
            count += 1
    except Exception as exc:
        print(f'[path check] Could not list {path}: {exc}', flush=True)


def _normalise_path_token(text: str) -> str:
    import re
    return re.sub(r'[^a-z0-9]+', '', str(text).lower())


def _resolve_kaggle_input_path(raw_path: str | None) -> str | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.exists():
        return str(path)
    # Kaggle input slugs are usually lower-case / hyphenated. Try to repair
    # common paths such as /kaggle/input/MT5_Dataset/raw -> /kaggle/input/mt5-dataset/raw.
    if str(path).startswith('/kaggle/input'):
        root = Path('/kaggle/input')
        if root.exists():
            parts = path.parts
            wanted_dataset = parts[3] if len(parts) > 3 else None
            tail = Path(*parts[4:]) if len(parts) > 4 else Path()
            if wanted_dataset:
                wanted_norm = _normalise_path_token(wanted_dataset)
                for ds in root.iterdir():
                    if ds.is_dir() and _normalise_path_token(ds.name) == wanted_norm:
                        candidate = ds / tail
                        if candidate.exists():
                            print(f'[path check] Resolved raw input path: {path} -> {candidate}', flush=True)
                            return str(candidate)
                        # If /raw was supplied but absent, fall back to dataset root.
                        if tail.name.lower() == 'raw':
                            print(f'[path check] {candidate} not found; using dataset root {ds}', flush=True)
                            return str(ds)
    return str(path)


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
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.stdout:
        print(result.stdout, end='', flush=True)
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr, flush=True)
    if result.returncode != 0:
        cmd_str = ' '.join(map(str, cmd))
        raise SystemExit(
            f'Command failed with exit code {result.returncode}: {cmd_str}\n'
            'Scroll up for the command output printed immediately above this message.'
        )


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
    args.raw_input_dir = _resolve_kaggle_input_path(args.raw_input_dir)
    if args.raw_input_dir is not None:
        raw_path = Path(args.raw_input_dir)
        if not raw_path.exists():
            print(f'[path check] raw input directory not found: {raw_path}', flush=True)
            _list_dir_brief(Path('/kaggle/input'))
            raise SystemExit('Raw input directory does not exist. Use the exact lower-case Kaggle dataset path, or set --raw-input-dir to the dataset root.')
        _list_dir_brief(raw_path, max_items=30)

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
        if not args.dry_run:
            missing_raw = [sym for sym in symbols if not Path('data/raw', f'{sym}_{args.timeframe.upper()}.csv').exists()]
            if missing_raw:
                raise SystemExit(f'Raw copy/prep did not produce expected files for: {missing_raw}. Check CSV names and symbol list.')

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
        if not args.dry_run:
            for sym in symbols:
                direction_path = Path('data/direction', f'{sym}_{args.timeframe.upper()}_direction_training.csv')
                if not direction_path.exists():
                    raise SystemExit(f'Direction dataset was not created: {direction_path}')
                try:
                    line_count = sum(1 for _ in direction_path.open('r', encoding='utf-8', errors='ignore'))
                except Exception:
                    line_count = -1
                if line_count <= 1:
                    raise SystemExit(
                        f'Direction dataset is empty or header-only: {direction_path}. '
                        'This usually means the requested --train-start/--replay-end date range does not overlap the raw CSV, '
                        'or the raw time column was not parsed correctly. Check the printed row/date filter output above.'
                    )
                print(f'[check] {direction_path}: {line_count-1:,} data rows', flush=True)

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

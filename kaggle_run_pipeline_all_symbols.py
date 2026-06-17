from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SYMBOLS = [
    'EURUSD',
    'GBPUSD',
    'USDJPY',
    'USDCHF',
    'USDCAD',
    'AUDUSD',
    'NZDUSD',
    'EURJPY',
    'GBPJPY',
    'AUDJPY',
    'CADJPY',
    'CHFJPY',
    'EURGBP',
    'EURCHF',
    'GBPCHF',
    'GBPAUD',
    'GBPCAD',
    'EURAUD',
    'EURCAD',
    'AUDNZD',
    'AUDCAD',
    'AUDCHF',
    'CADCHF',
    'EURNZD',
    'GBPNZD',
    'NZDCAD',
    'NZDCHF',
    'NZDJPY',
    'USDSGD',
    'GBPSGD',
    'SGDJPY',
]


DEFAULT_CONFIGS = [
    'config/direction_settings_residual_mlp.yaml',
    'config/direction_settings_tcn.yaml',
    'config/direction_settings_inception_time.yaml',
    'config/direction_settings_llm_transformer',
    'config/direction_settings_mixture_of_experts.yaml',
    'config/direction_settings_llm_transformer.yaml',
]


KAGGLE_WORKING = Path(os.environ.get('KAGGLE_WORKING_DIR', '/kaggle/working'))
DEFAULT_OUTPUT_ROOT = str(KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path('.'))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _run(cmd: list[str], *, dry_run: bool = False, check: bool = True, cwd: str | Path | None = None) -> int:
    print('\n$ ' + ' '.join(map(str, cmd)), flush=True)
    if dry_run:
        return 0
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    if result.stdout:
        print(result.stdout, end='', flush=True)
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr, flush=True)
    if result.returncode != 0 and check:
        raise RuntimeError(f'Command failed with exit code {result.returncode}: ' + ' '.join(map(str, cmd)))
    return int(result.returncode)


def _add_optional(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _normalise_symbols(symbols: list[str] | None) -> list[str]:
    """Return a clean symbol list.

    Passing no symbols, ALL, all, or * trains every configured default symbol.
    """
    if not symbols:
        return list(DEFAULT_SYMBOLS)

    cleaned = [str(s).upper().strip() for s in symbols if str(s).strip()]
    if not cleaned or any(s in {'ALL', '*'} for s in cleaned):
        return list(DEFAULT_SYMBOLS)

    out: list[str] = []
    seen: set[str] = set()
    for symbol in cleaned:
        if symbol not in seen:
            out.append(symbol)
            seen.add(symbol)
    return out


def _normalise_sides(sides: list[str] | None) -> list[str]:
    out: list[str] = []
    for side in sides or ['buy', 'sell']:
        side_l = str(side).strip().lower()
        if side_l == 'long':
            side_l = 'buy'
        if side_l == 'short':
            side_l = 'sell'
        if side_l == 'both':
            expanded = ['buy', 'sell']
        elif side_l in {'buy', 'sell'}:
            expanded = [side_l]
        else:
            raise ValueError(f'Unsupported side {side!r}; use buy, sell, or both.')
        for expanded_side in expanded:
            if expanded_side not in out:
                out.append(expanded_side)
    return out or ['buy', 'sell']


def _config_token(config_path: str | Path, cfg: dict[str, Any]) -> str:
    paths = cfg.get('paths', {}) or {}
    model_dir = str(paths.get('model_dir', '') or '')
    if model_dir:
        token = Path(model_dir).name
    else:
        token = Path(config_path).stem.replace('direction_settings_', '')
    token = token.strip().replace(' ', '_')
    return token or Path(config_path).stem


def _normalise_configs(configs: list[str] | None, *, skip_missing: bool = True) -> list[str]:
    """Clean config list and optionally skip unavailable default configs."""
    raw = configs or list(DEFAULT_CONFIGS)
    if len(raw) == 1 and str(raw[0]).strip().lower() in {'all', '*'}:
        raw = list(DEFAULT_CONFIGS)

    out: list[str] = []
    missing: list[str] = []
    for cfg in raw:
        cfg_s = str(cfg).strip()
        if not cfg_s:
            continue
        if Path(cfg_s).exists():
            out.append(cfg_s)
        else:
            missing.append(cfg_s)

    if missing and not skip_missing:
        raise SystemExit('Missing config file(s): ' + ', '.join(missing))
    if missing and skip_missing:
        print('Skipping missing config file(s): ' + ', '.join(missing), flush=True)
    if not out:
        raise SystemExit('No model config files were found. Check --configs or --project-dir.')
    return out


def _candidate_input_subdir(input_root: Path, subdir: str) -> Path | None:
    for candidate in (
        input_root / 'data' / subdir,
        input_root / subdir,
    ):
        if candidate.exists():
            return candidate
    return None


def _safe_remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _stage_input_data(
    *,
    input_data_root: str | Path | None,
    project_dir: Path,
    copy_input_data: bool,
    overwrite_data_links: bool,
) -> dict[str, Any]:
    """Make Kaggle input data visible as project_dir/data/{raw,processed_m5,direction}.

    Kaggle /kaggle/input datasets are read-only. For train-only runs, symlinking is
    space-efficient. For preparation stages that need to write into data folders, use
    --copy-input-data or do not provide --input-data-root and let the pipeline create
    writable data/ folders under /kaggle/working.
    """
    staged: dict[str, Any] = {'input_data_root': str(input_data_root) if input_data_root else None, 'items': []}
    if not input_data_root:
        return staged

    input_root = Path(input_data_root).expanduser().resolve()
    if not input_root.exists():
        raise SystemExit(f'--input-data-root does not exist: {input_root}')

    data_dir = project_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ('raw', 'processed_m5', 'direction'):
        src = _candidate_input_subdir(input_root, subdir)
        if src is None:
            continue
        dst = data_dir / subdir
        if dst.exists() or dst.is_symlink():
            if overwrite_data_links:
                _safe_remove_path(dst)
            else:
                staged['items'].append({'subdir': subdir, 'src': str(src), 'dst': str(dst), 'status': 'kept_existing'})
                continue
        if copy_input_data:
            shutil.copytree(src, dst)
            status = 'copied'
        else:
            os.symlink(src, dst, target_is_directory=True)
            status = 'symlinked'
        staged['items'].append({'subdir': subdir, 'src': str(src), 'dst': str(dst), 'status': status})
    return staged


def _make_side_config(base_config_path: str | Path, symbol: str, side: str, *, generated_root: Path, output_root: Path) -> Path:
    cfg = deepcopy(_load_yaml(base_config_path))
    symbol = symbol.upper()
    side = side.lower()
    token = _config_token(base_config_path, cfg)

    cfg.setdefault('project', {})['name'] = f"{cfg.get('project', {}).get('name', token)}_{symbol}_{side}"
    cfg.setdefault('trading', {})['symbols'] = [symbol]

    # Separate outputs per architecture/symbol/side so parallel jobs never write
    # the same checkpoint, scaler, feature list, report or summary JSON.
    model_dir = output_root / 'models' / token / symbol / side
    log_dir = output_root / 'logs' / token / symbol / side
    cfg.setdefault('paths', {})['model_dir'] = str(model_dir).replace('\\', '/')
    cfg.setdefault('paths', {})['log_dir'] = str(log_dir).replace('\\', '/')

    tcfg = cfg.setdefault('training', {})
    tcfg['side_setup_train_side'] = side
    tcfg['train_side'] = side
    tcfg['target_mode'] = 'side_setup_ranking'
    tcfg['use_pregenerated_direction_data'] = True
    tcfg['require_pregenerated_direction_data'] = True
    tcfg['buy_setup_loss_weight'] = 1.0 if side == 'buy' else 0.0
    tcfg['sell_setup_loss_weight'] = 1.0 if side == 'sell' else 0.0
    tcfg['replay_output_dir'] = str(log_dir / 'epoch_replay').replace('\\', '/')

    mcfg = cfg.setdefault('model', {})
    mcfg['use_side_setup_heads'] = True
    mcfg['decision_output_mode'] = 'side_setup'
    mcfg['use_setup_quality_head'] = bool(mcfg.get('use_setup_quality_head', True))

    rcfg = cfg.setdefault('replay', {})
    rcfg['threshold_mode'] = rcfg.get('threshold_mode', 'rolling_score_quantile')
    rcfg['allow_buy'] = side == 'buy'
    rcfg['allow_sell'] = side == 'sell'
    rcfg['output_dir'] = str(log_dir / 'replay').replace('\\', '/')

    out_path = generated_root / f'{Path(base_config_path).stem}_{symbol}_{side}.yaml'
    _write_yaml(out_path, cfg)
    return out_path


def _infer_task_model_token(config_path: str | Path, cfg: dict[str, Any], symbol: str, side: str) -> str:
    symbol_u = symbol.upper()
    side_l = side.lower()
    paths = cfg.get('paths', {}) or {}
    for key in ('log_dir', 'model_dir'):
        raw_path = str(paths.get(key, '') or '')
        if not raw_path:
            continue
        parts = Path(raw_path).parts
        if len(parts) >= 3 and parts[-1].lower() == side_l and parts[-2].upper() == symbol_u:
            return parts[-3]

    stem = Path(config_path).stem
    if stem.startswith('direction_settings_'):
        stem = stem[len('direction_settings_'):]
    suffix = f'_{symbol_u}_{side_l}'
    if stem.endswith(suffix):
        stem = stem[:-len(suffix)]
    return stem or 'model'


def _copy_side_named_reports(
    *,
    config_path: str | Path,
    symbol: str,
    side: str,
    timeframe: str,
    output_root: Path,
    report_dir: str | Path,
) -> dict[str, Any]:
    """Copy trainer outputs to zip-friendly filenames containing model/symbol/side."""
    cfg = _load_yaml(config_path)
    symbol_u = symbol.upper()
    side_l = side.lower()
    tf = str(timeframe).upper()
    model_token = _infer_task_model_token(config_path, cfg, symbol_u, side_l)
    log_dir = Path(str((cfg.get('paths', {}) or {}).get('log_dir', '') or output_root / 'logs' / model_token / symbol_u / side_l))

    target_dir = Path(report_dir)
    if not target_dir.is_absolute():
        target_dir = output_root / target_dir
    target_dir = target_dir / model_token
    target_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[tuple[str, Path, Path]] = [
        (
            'training_report',
            log_dir / f'{symbol_u}_{tf}_direction_training_report.json',
            target_dir / f'{model_token}_{symbol_u}_{tf}_{side_l}_direction_training_report.json',
        ),
        (
            'replay_each_epoch_summary',
            log_dir / f'direction_training_replay_each_epoch_summary_{tf}.json',
            target_dir / f'{model_token}_{symbol_u}_{tf}_{side_l}_direction_training_replay_each_epoch_summary.json',
        ),
    ]

    copied: dict[str, Any] = {'model': model_token, 'log_dir': str(log_dir), 'copies': [], 'missing': []}
    for kind, src, dst in candidates:
        if src.exists():
            shutil.copy2(src, dst)
            copied['copies'].append({'kind': kind, 'src': str(src), 'dst': str(dst)})
        else:
            copied['missing'].append({'kind': kind, 'src': str(src)})
    return copied


def _train_task(cmd: list[str], env_overrides: dict[str, str] | None = None, cwd: str | Path | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    env.setdefault('NUMEXPR_NUM_THREADS', '1')
    # Keep CUDA allocation a little more stable on Kaggle notebooks.
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    if env_overrides:
        env.update(env_overrides)
    print('\n$ ' + ' '.join(map(str, cmd)), flush=True)
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=cwd)
    if result.stdout:
        print(result.stdout, end='', flush=True)
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr, flush=True)
    return {
        'cmd': cmd,
        'returncode': int(result.returncode),
        'stdout_tail': (result.stdout or '')[-4000:],
        'stderr_tail': (result.stderr or '')[-4000:],
    }


def _zip_outputs(output_root: Path, zip_name: str, *, include_dirs: list[str] | None = None) -> Path | None:
    include_dirs = include_dirs or ['models', 'logs', 'config/generated_kaggle']
    zip_path = output_root / zip_name
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    found_any = False
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel_dir in include_dirs:
            base = output_root / rel_dir
            if not base.exists():
                continue
            found_any = True
            if base.is_file():
                zf.write(base, arcname=rel_dir)
                continue
            for path in base.rglob('*'):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(output_root)))
    if not found_any:
        try:
            zip_path.unlink()
        except FileNotFoundError:
            pass
        return None
    return zip_path


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            'Kaggle raw-data -> features -> strong setup labels -> all-symbol, '
            'side-specific model training pipeline.'
        )
    )
    p.add_argument('--project-dir', default='.', help='Project/code directory. The script chdirs here before running src modules.')
    p.add_argument('--input-data-root', default=None, help='Optional Kaggle input dataset root containing data/raw, data/processed_m5, and/or data/direction.')
    p.add_argument('--copy-input-data', action=argparse.BooleanOptionalAction, default=False, help='Copy input data into project data/ instead of symlinking. Use this if preparation stages need writable data.')
    p.add_argument('--overwrite-data-links', action='store_true', help='Replace existing data/raw, data/processed_m5, or data/direction symlinks/folders when staging --input-data-root.')
    p.add_argument('--raw-input-dir', default=None, help='Directory containing per-symbol CSVs, for example /kaggle/input/my-raw-csvs')
    p.add_argument('--combined-csv', default=None, help='Optional combined raw CSV containing a symbol column.')
    p.add_argument('--symbols', nargs='+', default=['ALL'], help='Symbols to process. Default/all/* trains all 31 default symbols.')
    p.add_argument('--timeframe', default='M5')
    p.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS, help='Model config files to train. Use all/* for the default model set.')
    p.add_argument('--sides', nargs='+', default=['buy', 'sell'], help='Sides to train separately: buy sell. Use both to expand to buy and sell.')
    p.add_argument('--mode', choices=['prepare-only', 'train-side-all', 'train-all', 'train-one', 'train-combined-all'], default='train-side-all', help='train-side-all/train-all/train-one build side-specific tasks. train-one is a backwards-compatible alias.')
    p.add_argument('--train-start', default=None)
    p.add_argument('--train-end', default=None)
    p.add_argument('--replay-start', default=None)
    p.add_argument('--replay-end', default=None)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--learning-rate', type=float, default=None)
    p.add_argument('--device', default='cuda', help='cpu, cuda, or leave blank for script default. Kaggle default is cuda.')
    p.add_argument('--parallel-jobs', type=int, default=1, help='Concurrent training subprocesses. Kaggle GPUs usually need 1; use 2 only if memory allows.')
    p.add_argument('--raw-max-rows-per-symbol', type=int, default=None)
    p.add_argument('--feature-max-rows', type=int, default=None)
    p.add_argument('--direction-max-rows', type=int, default=None)
    p.add_argument('--train-max-rows', type=int, default=None)
    p.add_argument('--prepare-workers', type=int, default=2)
    p.add_argument('--force-raw', action='store_true')
    p.add_argument('--force-features', action='store_true')
    p.add_argument('--skip-raw-copy', action=argparse.BooleanOptionalAction, default=True, help='Default true on Kaggle. Use --no-skip-raw-copy to build data/raw.')
    p.add_argument('--skip-feature-prep', action=argparse.BooleanOptionalAction, default=True, help='Default true on Kaggle. Use --no-skip-feature-prep to build features.')
    p.add_argument('--skip-direction-prep', action=argparse.BooleanOptionalAction, default=True, help='Default true on Kaggle. Use --no-skip-direction-prep to build direction labels.')
    p.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT, help='Writable root for Kaggle outputs. Default is /kaggle/working when available.')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--continue-on-model-error', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--skip-missing-configs', action=argparse.BooleanOptionalAction, default=True, help='Skip default model config files that are not present. Use --no-skip-missing-configs to fail fast.')
    p.add_argument('--max-tasks', type=int, default=None, help='Optional cap for smoke tests, e.g. --max-tasks 2.')
    p.add_argument('--side-named-report-copies', action=argparse.BooleanOptionalAction, default=True, help='Copy generic trainer reports into side-named zip-friendly files.')
    p.add_argument('--side-named-report-dir', default='logs/side_named_reports', help='Report collection directory, relative to output-root unless absolute.')
    p.add_argument('--zip-outputs', action=argparse.BooleanOptionalAction, default=True, help='Zip models/logs/generated configs at the end for Kaggle download.')
    p.add_argument('--zip-name', default='kaggle_training_outputs.zip')
    args = p.parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    if not project_dir.exists():
        raise SystemExit(f'--project-dir does not exist: {project_dir}')
    os.chdir(project_dir)

    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (project_dir / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    generated_root = output_root / 'config' / 'generated_kaggle'

    staged_data = _stage_input_data(
        input_data_root=args.input_data_root,
        project_dir=project_dir,
        copy_input_data=bool(args.copy_input_data),
        overwrite_data_links=bool(args.overwrite_data_links),
    )

    if args.mode in {'train-all', 'train-one'}:
        args.mode = 'train-side-all'

    symbols = _normalise_symbols(args.symbols)
    sides = _normalise_sides(args.sides)
    args.configs = _normalise_configs(args.configs, skip_missing=args.skip_missing_configs)
    first_config = args.configs[0]

    print(f'Project dir: {project_dir}', flush=True)
    print(f'Output root: {output_root}', flush=True)
    if staged_data.get('items'):
        print('Input data staging:', json.dumps(staged_data['items'], indent=2), flush=True)
    print(
        f'Pipeline selection: {len(args.configs)} config(s) x {len(symbols)} symbol(s) x '
        f'{len(sides) if args.mode != "train-combined-all" else 1} side task(s)',
        flush=True,
    )
    print('Symbols: ' + ', '.join(symbols), flush=True)
    print('Configs: ' + ', '.join(args.configs), flush=True)
    if args.mode != 'train-combined-all':
        print('Sides: ' + ', '.join(sides), flush=True)

    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    if not args.skip_raw_copy:
        if not args.raw_input_dir and not args.combined_csv:
            raise SystemExit('Provide --raw-input-dir or --combined-csv, or use --skip-raw-copy when data/raw already contains SYMBOL_M5.csv files.')
        cmd = [
            sys.executable, '-m', 'src.kaggle_prepare_raw_data',
            '--input-dir', args.raw_input_dir or str(Path(args.combined_csv).parent),
            '--output-dir', 'data/raw',
            '--timeframe', args.timeframe,
            '--symbols', *symbols,
        ]
        _add_optional(cmd, '--combined-csv', args.combined_csv)
        _add_optional(cmd, '--max-rows-per-symbol', args.raw_max_rows_per_symbol)
        if args.force_raw:
            cmd.append('--force')
        _run(cmd, dry_run=args.dry_run, cwd=project_dir)

    if not args.skip_feature_prep:
        cmd = [
            sys.executable, '-m', 'src.prepare_mt5_data',
            '--config', first_config,
            '--symbols', *symbols,
            '--timeframe', args.timeframe,
            '--workers', str(args.prepare_workers),
        ]
        _add_optional(cmd, '--max-rows', args.feature_max_rows)
        if args.force_features:
            cmd.append('--force')
        _run(cmd, dry_run=args.dry_run, cwd=project_dir)

    if not args.skip_direction_prep:
        cmd = [
            sys.executable, '-m', 'src.prepare_direction_dataset',
            '--config', first_config,
            '--symbols', *symbols,
            '--workers', str(args.prepare_workers),
        ]
        _add_optional(cmd, '--date-start', args.train_start)
        # Prepare through replay_end so replay has labelled rows. The training
        # loader still tail-drops horizon rows at date_end for the fitting split.
        _add_optional(cmd, '--date-end', args.replay_end or args.train_end)
        _add_optional(cmd, '--max-rows', args.direction_max_rows)
        _run(cmd, dry_run=args.dry_run, cwd=project_dir)

    if args.mode == 'prepare-only':
        print('\nKaggle preparation complete. Direction CSVs are under data/direction/.', flush=True)
        return

    tasks: list[tuple[str, str, str, list[str]]] = []
    if args.mode == 'train-combined-all':
        for config in args.configs:
            for symbol in symbols:
                cmd = [
                    sys.executable, '-m', 'src.train_direction_policy_replay_each_epoch',
                    '--config', config,
                    '--symbols', symbol,
                    '--epochs', str(args.epochs),
                    '--batch-size', str(args.batch_size),
                    '--model-selection-metric', 'replay_score',
                    '--train-side', 'both',
                ]
                _add_optional(cmd, '--date-start', args.train_start)
                _add_optional(cmd, '--date-end', args.train_end)
                _add_optional(cmd, '--replay-start', args.replay_start)
                _add_optional(cmd, '--replay-end', args.replay_end)
                _add_optional(cmd, '--max-rows', args.train_max_rows)
                _add_optional(cmd, '--learning-rate', args.learning_rate)
                _add_optional(cmd, '--device', args.device)
                tasks.append((config, symbol, 'both', cmd))
    else:
        for config in args.configs:
            for symbol in symbols:
                for side in sides:
                    cfg_path = _make_side_config(config, symbol, side, generated_root=generated_root, output_root=output_root)
                    cmd = [
                        sys.executable, '-m', 'src.train_direction_policy_replay_each_epoch',
                        '--config', str(cfg_path),
                        '--symbols', symbol,
                        '--epochs', str(args.epochs),
                        '--batch-size', str(args.batch_size),
                        '--model-selection-metric', 'replay_score',
                        '--train-side', side,
                    ]
                    _add_optional(cmd, '--date-start', args.train_start)
                    _add_optional(cmd, '--date-end', args.train_end)
                    _add_optional(cmd, '--replay-start', args.replay_start)
                    _add_optional(cmd, '--replay-end', args.replay_end)
                    _add_optional(cmd, '--max-rows', args.train_max_rows)
                    _add_optional(cmd, '--learning-rate', args.learning_rate)
                    _add_optional(cmd, '--device', args.device)
                    tasks.append((str(cfg_path), symbol, side, cmd))

    if args.max_tasks is not None:
        tasks = tasks[: max(0, int(args.max_tasks))]
        print(f'Max task cap applied: {len(tasks)} task(s).', flush=True)

    if args.dry_run:
        print(f'\nDry run: prepared {len(tasks)} training task(s).')
        for _, _, _, cmd in tasks:
            print('$ ' + ' '.join(map(str, cmd)))
        return

    max_workers = max(1, int(args.parallel_jobs or 1))
    print(f'\nStarting {len(tasks)} training task(s) with parallel_jobs={max_workers}', flush=True)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {
            ex.submit(_train_task, cmd, None, project_dir): {'config': config, 'symbol': symbol, 'side': side, 'cmd': cmd}
            for config, symbol, side, cmd in tasks
        }
        for fut in cf.as_completed(future_map):
            meta = future_map[fut]
            try:
                res = fut.result()
            except Exception as exc:
                res = {'cmd': meta['cmd'], 'returncode': -999, 'error': repr(exc)}
            res.update({k: v for k, v in meta.items() if k != 'cmd'})
            if int(res.get('returncode', 1)) == 0 and args.side_named_report_copies:
                try:
                    res['side_named_reports'] = _copy_side_named_reports(
                        config_path=meta['config'],
                        symbol=meta['symbol'],
                        side=meta['side'],
                        timeframe=args.timeframe,
                        output_root=output_root,
                        report_dir=args.side_named_report_dir,
                    )
                except Exception as exc:
                    res['side_named_reports_error'] = repr(exc)
            results.append(res)
            if int(res.get('returncode', 1)) != 0:
                failures.append(res)
                print(f"[FAILED] {meta['symbol']} {meta['side']} {meta['config']} rc={res.get('returncode')}", flush=True)
                if not args.continue_on_model_error:
                    raise SystemExit('Stopping because --no-continue-on-model-error was set.')
            else:
                print(f"[OK] {meta['symbol']} {meta['side']} {meta['config']}", flush=True)

    summary_path = output_root / 'logs' / 'kaggle_side_setup_training_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({'tasks': results, 'failures': failures, 'staged_data': staged_data}, f, indent=2)
    print(f'\nWrote Kaggle training summary: {summary_path}', flush=True)

    if args.zip_outputs:
        zip_path = _zip_outputs(output_root, args.zip_name)
        if zip_path is not None:
            print(f'Wrote zipped Kaggle outputs: {zip_path}', flush=True)
        else:
            print('No model/log/generated-config outputs found to zip.', flush=True)

    if failures:
        print(f'Completed with {len(failures)} failed task(s). See summary JSON for stdout/stderr tails.', flush=True)
    else:
        print('All Kaggle side-specific training tasks completed successfully.', flush=True)


if __name__ == '__main__':
    main()

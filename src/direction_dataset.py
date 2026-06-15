from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .dataset import assert_no_feature_leakage, choose_feature_columns
from .analytic_signals import ensure_analytic_signal_features


@dataclass
class DirectionPreparedArrays:
    X_seq: np.ndarray
    y_direction: np.ndarray
    feature_columns: list[str]
    row_indices: np.ndarray
    scaler: StandardScaler
    buy_edge_pips: np.ndarray | None = None
    sell_edge_pips: np.ndarray | None = None
    has_edge_targets: np.ndarray | None = None
    analytic_signal_class: np.ndarray | None = None


def _require_direction_target(df: pd.DataFrame) -> None:
    if 'direction_target' not in df.columns:
        raise KeyError(
            "Missing direction_target. Regenerate the pregenerated direction dataset, "
            "or use data with OHLC columns so targets can be generated first."
        )


def prepare_direction_arrays(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    scaler: StandardScaler | None = None,
    feature_columns: list[str] | None = None,
    fit_scaler: bool = True,
) -> DirectionPreparedArrays:
    """Prepare sequence arrays for the simple BUY/SELL/NO_TRADE classifier."""
    seq_len = int((cfg.get('model') or {}).get('sequence_length', 64))
    fill = float((cfg.get('features') or {}).get('fillna_value', 0.0))
    _require_direction_target(df)
    df = ensure_analytic_signal_features(df, cfg)

    if feature_columns is not None:
        feature_columns = list(feature_columns)
        assert_no_feature_leakage(feature_columns)
    else:
        feature_columns = choose_feature_columns(df, cfg)

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        preview = ', '.join(missing[:20])
        more = ' ...' if len(missing) > 20 else ''
        raise KeyError(f'Missing required model feature columns: {preview}{more}.')

    X_flat = (
        df[feature_columns]
        .apply(pd.to_numeric, errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
        .fillna(fill)
        .to_numpy(np.float32)
    )
    scaler = scaler or StandardScaler()
    if fit_scaler:
        X_flat = scaler.fit_transform(X_flat).astype(np.float32)
    else:
        X_flat = scaler.transform(X_flat).astype(np.float32)

    # direction_target convention:
    #   0 = SELL, 1 = NO_TRADE, 2 = BUY, -1 = IGNORE
    # IGNORE rows are not used as supervised sequence endpoints. The rows remain
    # in the dataframe, so they can still provide historical context for nearby
    # labelled sequence endpoints.
    raw_y = pd.to_numeric(df['direction_target'], errors='coerce').fillna(-1).astype(int).to_numpy(np.int64)

    # Optional future-derived regression targets for the hierarchical edge/pips
    # head. These columns are never used as input features because the feature
    # selector rejects *_target columns. If older pregenerated datasets do not
    # contain them, the model simply trains without edge loss.
    buy_edge_col = 'buy_edge_pips_target' if 'buy_edge_pips_target' in df.columns else (
        'buy_candidate_net_pips' if 'buy_candidate_net_pips' in df.columns else None
    )
    sell_edge_col = 'sell_edge_pips_target' if 'sell_edge_pips_target' in df.columns else (
        'sell_candidate_net_pips' if 'sell_candidate_net_pips' in df.columns else None
    )
    if buy_edge_col and sell_edge_col:
        buy_edge_raw = pd.to_numeric(df[buy_edge_col], errors='coerce').to_numpy(np.float32)
        sell_edge_raw = pd.to_numeric(df[sell_edge_col], errors='coerce').to_numpy(np.float32)
    else:
        buy_edge_raw = np.full(len(df), np.nan, dtype=np.float32)
        sell_edge_raw = np.full(len(df), np.nan, dtype=np.float32)

    if 'sig_analytic_signal_class' in df.columns:
        analytic_signal_raw = (
            pd.to_numeric(df['sig_analytic_signal_class'], errors='coerce')
            .fillna(1)
            .clip(lower=0, upper=2)
            .astype(int)
            .to_numpy(np.int64)
        )
    else:
        analytic_signal_raw = np.ones(len(df), dtype=np.int64)

    xs: list[np.ndarray] = []
    ys: list[int] = []
    indices: list[int] = []
    buy_edges: list[float] = []
    sell_edges: list[float] = []
    edge_masks: list[bool] = []
    analytic_signal_classes: list[int] = []
    for end in range(seq_len - 1, len(df)):
        label = int(raw_y[end])
        if label not in (0, 1, 2):
            continue
        be = float(buy_edge_raw[end])
        se = float(sell_edge_raw[end])
        has_edge = bool(np.isfinite(be) and np.isfinite(se))
        xs.append(X_flat[end - seq_len + 1:end + 1])
        ys.append(label)
        indices.append(end)
        buy_edges.append(be if has_edge else 0.0)
        sell_edges.append(se if has_edge else 0.0)
        edge_masks.append(has_edge)
        analytic_signal_classes.append(int(analytic_signal_raw[end]))

    return DirectionPreparedArrays(
        X_seq=np.asarray(xs, dtype=np.float32),
        y_direction=np.asarray(ys, dtype=np.int64),
        feature_columns=list(feature_columns),
        row_indices=np.asarray(indices, dtype=np.int64),
        scaler=scaler,
        buy_edge_pips=np.asarray(buy_edges, dtype=np.float32),
        sell_edge_pips=np.asarray(sell_edges, dtype=np.float32),
        has_edge_targets=np.asarray(edge_masks, dtype=bool),
        analytic_signal_class=np.asarray(analytic_signal_classes, dtype=np.int64),
    )


class DirectionDataset(Dataset):
    def __init__(self, arr: DirectionPreparedArrays):
        self.arr = arr

    def __len__(self) -> int:
        return len(self.arr.X_seq)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            'x': torch.tensor(self.arr.X_seq[idx], dtype=torch.float32),
            'direction': torch.tensor(self.arr.y_direction[idx], dtype=torch.long),
        }
        if self.arr.buy_edge_pips is not None and self.arr.sell_edge_pips is not None:
            item['buy_edge_pips'] = torch.tensor(float(self.arr.buy_edge_pips[idx]), dtype=torch.float32)
            item['sell_edge_pips'] = torch.tensor(float(self.arr.sell_edge_pips[idx]), dtype=torch.float32)
            has_edge = bool(self.arr.has_edge_targets[idx]) if self.arr.has_edge_targets is not None else False
            item['has_edge_targets'] = torch.tensor(has_edge, dtype=torch.bool)
        if self.arr.analytic_signal_class is not None:
            item['analytic_signal_class'] = torch.tensor(int(self.arr.analytic_signal_class[idx]), dtype=torch.long)
        return item

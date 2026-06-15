from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .forex import pips_from_price_delta, price_delta_from_pips
from .spread_risk_config import symbol_default_spread_points, symbol_max_spread_pips

# Legacy names are kept only for diagnostics/backwards-compatible CSV columns.
DIRECTION_NAMES = {0: 'SELL', 1: 'NO_TRADE', 2: 'BUY'}
DECISION_NAMES = {0: 'BLOCK', 1: 'ALLOW'}
OUTCOME_NAMES = {0: 'SL', 1: 'TIME_EXIT', 2: 'TP'}
SIDE_NAMES = ('BUY', 'SELL')


@dataclass
class TradeOutcome:
    pips: float
    outcome: int  # 0=SL, 1=TIME_EXIT, 2=TP
    bars_to_outcome: int
    ambiguous: bool = False


def _label_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get('labels', {}) or {}


def _spread_cost_pips(spread_points: float, cfg: dict[str, Any]) -> float:
    lcfg = _label_cfg(cfg)
    return float(spread_points) * float(lcfg.get('spread_pips_per_point', 0.1))


def _spread_delta(symbol: str, spread_points: float, cfg: dict[str, Any]) -> float:
    return price_delta_from_pips(symbol, _spread_cost_pips(spread_points, cfg), cfg)


def _same_bar_policy(cfg: dict[str, Any]) -> str:
    lcfg = _label_cfg(cfg)
    policy = str(lcfg.get('same_bar_tp_sl_policy', '') or '').strip().lower()
    if policy in {'stop_first', 'sl_first', 'conservative'}:
        return 'stop_first'
    if policy in {'take_profit_first', 'tp_first', 'optimistic'}:
        return 'take_profit_first'
    if policy in {'discard', 'time_exit'}:
        return 'discard'
    # Backwards-compatible behaviour.
    return 'stop_first' if bool(lcfg.get('conservative_same_bar_hits', True)) else 'take_profit_first'


def _resolve_same_bar(tp_pips: float, sl_pips: float, bars_to_outcome: int, cfg: dict[str, Any]) -> TradeOutcome:
    policy = _same_bar_policy(cfg)
    if policy == 'take_profit_first':
        return TradeOutcome(tp_pips, 2, bars_to_outcome, ambiguous=True)
    if policy == 'discard':
        return TradeOutcome(0.0, 1, bars_to_outcome, ambiguous=True)
    return TradeOutcome(-sl_pips, 0, bars_to_outcome, ambiguous=True)


def _first_hit_outcome(
    symbol: str,
    entry: float,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    side: str,
    cfg: dict[str, Any],
) -> TradeOutcome:
    """Legacy close/high/low barrier simulation.

    This path is preserved for backwards compatibility when
    labels.use_live_bid_ask_simulation is false or missing.
    """
    lcfg = _label_cfg(cfg)
    tp_pips = float(lcfg.get('take_profit_pips', 7.0))
    sl_pips = float(lcfg.get('stop_loss_pips', 5.0))
    side = str(side).upper()

    if side == 'BUY':
        tp = entry + price_delta_from_pips(symbol, tp_pips, cfg)
        sl = entry - price_delta_from_pips(symbol, sl_pips, cfg)
        for j, (hi, lo) in enumerate(zip(highs, lows), start=1):
            hit_tp = hi >= tp
            hit_sl = lo <= sl
            if hit_tp and hit_sl:
                return _resolve_same_bar(tp_pips, sl_pips, j, cfg)
            if hit_tp:
                return TradeOutcome(tp_pips, 2, j)
            if hit_sl:
                return TradeOutcome(-sl_pips, 0, j)
        end = closes[-1] if len(closes) else entry
        return TradeOutcome(pips_from_price_delta(symbol, end - entry, cfg), 1, len(closes))

    if side == 'SELL':
        tp = entry - price_delta_from_pips(symbol, tp_pips, cfg)
        sl = entry + price_delta_from_pips(symbol, sl_pips, cfg)
        for j, (hi, lo) in enumerate(zip(highs, lows), start=1):
            hit_tp = lo <= tp
            hit_sl = hi >= sl
            if hit_tp and hit_sl:
                return _resolve_same_bar(tp_pips, sl_pips, j, cfg)
            if hit_tp:
                return TradeOutcome(tp_pips, 2, j)
            if hit_sl:
                return TradeOutcome(-sl_pips, 0, j)
        end = closes[-1] if len(closes) else entry
        return TradeOutcome(pips_from_price_delta(symbol, entry - end, cfg), 1, len(closes))

    raise ValueError(f'Unsupported side {side!r}; expected BUY or SELL')


def _first_hit_outcome_live_bidask(
    symbol: str,
    bid_entry: float,
    bid_highs: np.ndarray,
    bid_lows: np.ndarray,
    bid_closes: np.ndarray,
    spread_points: np.ndarray,
    side: str,
    cfg: dict[str, Any],
    *,
    entry_spread_points: float,
) -> TradeOutcome:
    """Live-style barrier simulation using bid OHLC plus spread.

    Assumption: MT5/rates OHLC are bid prices.

    BUY:
      - opens at ask = bid + spread + slippage
      - TP/SL are triggered by future bid high/low

    SELL:
      - opens at bid - slippage
      - TP/SL are triggered by future ask low/high = future bid + future spread
    """
    lcfg = _label_cfg(cfg)
    tp_pips = float(lcfg.get('take_profit_pips', 7.0))
    sl_pips = float(lcfg.get('stop_loss_pips', 5.0))
    slippage_pips = float(lcfg.get('slippage_pips', 0.0) or 0.0)
    slippage_delta = price_delta_from_pips(symbol, slippage_pips, cfg)
    side = str(side).upper()

    if len(bid_closes) == 0:
        return TradeOutcome(0.0, 1, 0)

    future_spreads = np.asarray(spread_points, dtype=float)
    if len(future_spreads) < len(bid_closes):
        pad = np.full(len(bid_closes) - len(future_spreads), float(entry_spread_points), dtype=float)
        future_spreads = np.concatenate([future_spreads, pad])

    if side == 'BUY':
        ask_entry = bid_entry + _spread_delta(symbol, entry_spread_points, cfg) + slippage_delta
        tp = ask_entry + price_delta_from_pips(symbol, tp_pips, cfg)
        sl = ask_entry - price_delta_from_pips(symbol, sl_pips, cfg)
        for j, (bid_hi, bid_lo) in enumerate(zip(bid_highs, bid_lows), start=1):
            hit_tp = bid_hi >= tp
            hit_sl = bid_lo <= sl
            if hit_tp and hit_sl:
                return _resolve_same_bar(tp_pips, sl_pips, j, cfg)
            if hit_tp:
                return TradeOutcome(tp_pips, 2, j)
            if hit_sl:
                return TradeOutcome(-sl_pips, 0, j)
        exit_bid = bid_closes[-1]
        return TradeOutcome(pips_from_price_delta(symbol, exit_bid - ask_entry, cfg), 1, len(bid_closes))

    if side == 'SELL':
        sell_entry = bid_entry - slippage_delta
        tp = sell_entry - price_delta_from_pips(symbol, tp_pips, cfg)
        sl = sell_entry + price_delta_from_pips(symbol, sl_pips, cfg)
        for j, (bid_hi, bid_lo, sp) in enumerate(zip(bid_highs, bid_lows, future_spreads), start=1):
            ask_hi = bid_hi + _spread_delta(symbol, sp, cfg)
            ask_lo = bid_lo + _spread_delta(symbol, sp, cfg)
            hit_tp = ask_lo <= tp
            hit_sl = ask_hi >= sl
            if hit_tp and hit_sl:
                return _resolve_same_bar(tp_pips, sl_pips, j, cfg)
            if hit_tp:
                return TradeOutcome(tp_pips, 2, j)
            if hit_sl:
                return TradeOutcome(-sl_pips, 0, j)
        exit_ask = bid_closes[-1] + _spread_delta(symbol, future_spreads[min(len(bid_closes) - 1, len(future_spreads) - 1)], cfg)
        return TradeOutcome(pips_from_price_delta(symbol, sell_entry - exit_ask, cfg), 1, len(bid_closes))

    raise ValueError(f'Unsupported side {side!r}; expected BUY or SELL')


def _required_ohlc_columns(df: pd.DataFrame) -> tuple[str, str, str, str]:
    lower = {str(c).lower(): c for c in df.columns}
    close = lower.get('close') or lower.get('close_price') or lower.get('bid_close')
    high = lower.get('high') or lower.get('high_price') or lower.get('bid_high')
    low = lower.get('low') or lower.get('low_price') or lower.get('bid_low')
    open_ = lower.get('open') or lower.get('open_price') or lower.get('bid_open') or close
    if not close or not high or not low:
        raise ValueError('Processed CSV must contain close/high/low columns to generate BUY/SELL outcome targets.')
    return open_, high, low, close



def _positive_filter_cfg(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {'enabled': bool(value)}


def _discard_positive_rows(out: pd.DataFrame, indices: list[int], *, mode: str, reason: str) -> int:
    if not indices:
        return 0
    mode_norm = str(mode or 'no_trade').strip().lower()
    if mode_norm in {'ignore', 'ignored', 'exclude', 'skip'}:
        replacement = -1
    elif mode_norm in {'drop'}:
        # Keep row positions stable for sequence generation. A physical row drop
        # can create artificial sequence jumps, so treat drop as ignore.
        replacement = -1
    else:
        replacement = 1
    out.loc[indices, 'direction_target'] = int(replacement)
    if 'label_filter_status' in out.columns:
        out.loc[indices, 'label_filter_status'] = reason
    return int(len(indices))


def _side_strength_column(side: str) -> str:
    return f'{str(side).lower()}_candidate_strength_score'


def _best_index_by_strength(out: pd.DataFrame, indices: list[int], side: str) -> int:
    if len(indices) == 1:
        return int(indices[0])
    col = _side_strength_column(side)
    if col not in out.columns:
        return int(indices[0])
    values = pd.to_numeric(out.loc[indices, col], errors='coerce').fillna(-1.0e18)
    return int(values.idxmax())


def _apply_positive_deduplication(out: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep only one strong positive label from each nearby setup cluster.

    A single market setup often creates many consecutive positive rows. Keeping
    all of them makes the side models learn broad regions rather than the
    strongest, most identifiable entry point. This filter keeps the strongest
    row in each side-specific cluster and turns the other positive rows into
    NO_TRADE or IGNORE according to discarded_positive_mode.
    """
    lcfg = _label_cfg(cfg)
    dcfg = _positive_filter_cfg(lcfg.get('positive_label_deduplication', {}))
    enabled = bool(dcfg.get('enabled', False))
    min_gap_bars = int(dcfg.get('min_gap_bars', 0) or 0)
    mode = str(dcfg.get('discarded_positive_mode', lcfg.get('discarded_positive_mode', 'no_trade')) or 'no_trade')
    info: dict[str, Any] = {
        'enabled': enabled,
        'mode': str(dcfg.get('mode', 'best_per_cluster')),
        'min_gap_bars': int(min_gap_bars),
        'discarded_positive_mode': mode,
        'buy_removed': 0,
        'sell_removed': 0,
        'buy_clusters': 0,
        'sell_clusters': 0,
    }
    if not enabled or min_gap_bars <= 0 or 'direction_target' not in out.columns:
        return out, info

    result = out.copy()
    cluster_days = None
    if 'time_utc' in result.columns:
        parsed_times = pd.to_datetime(result['time_utc'], utc=True, errors='coerce')
        if parsed_times.notna().any():
            cluster_days = parsed_times.dt.floor('D')

    for side, direction_idx in (('buy', 2), ('sell', 0)):
        positive_indices = [int(x) for x in result.index[pd.to_numeric(result['direction_target'], errors='coerce') == direction_idx].tolist()]
        if not positive_indices:
            continue
        clusters: list[list[int]] = []
        current = [positive_indices[0]]
        for idx in positive_indices[1:]:
            same_day = True
            if cluster_days is not None:
                same_day = bool(cluster_days.iloc[idx] == cluster_days.iloc[current[-1]])
            if same_day and idx - current[-1] <= min_gap_bars:
                current.append(idx)
            else:
                clusters.append(current)
                current = [idx]
        clusters.append(current)
        info[f'{side}_clusters'] = int(len(clusters))

        discard: list[int] = []
        for cluster in clusters:
            keep = _best_index_by_strength(result, cluster, side)
            discard.extend([int(x) for x in cluster if int(x) != keep])
        removed = _discard_positive_rows(result, discard, mode=mode, reason=f'{side}_deduplicated')
        info[f'{side}_removed'] = int(removed)
    return result, info


def _apply_daily_positive_cap(out: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Limit retained positive labels to the strongest K setups per day.

    The cap is side-specific. It runs after de-duplication so a day with many
    overlapping BUY labels does not dominate the training set. Discarded labels
    can be converted to NO_TRADE or IGNORE.
    """
    lcfg = _label_cfg(cfg)
    ccfg = _positive_filter_cfg(lcfg.get('max_positive_setups_per_day', {}))
    enabled = bool(ccfg.get('enabled', False))
    mode = str(ccfg.get('discarded_positive_mode', lcfg.get('discarded_positive_mode', 'no_trade')) or 'no_trade')
    info: dict[str, Any] = {
        'enabled': enabled,
        'buy_cap': ccfg.get('buy', ccfg.get('per_side', ccfg.get('total'))),
        'sell_cap': ccfg.get('sell', ccfg.get('per_side', ccfg.get('total'))),
        'discarded_positive_mode': mode,
        'buy_removed': 0,
        'sell_removed': 0,
        'days_seen': 0,
        'warning': None,
    }
    if not enabled or 'direction_target' not in out.columns:
        return out, info
    if 'time_utc' not in out.columns:
        info['warning'] = 'time_utc missing; daily positive cap skipped'
        return out, info

    result = out.copy()
    times = pd.to_datetime(result['time_utc'], utc=True, errors='coerce')
    valid_days = times.dt.floor('D')
    if valid_days.isna().all():
        info['warning'] = 'time_utc could not be parsed; daily positive cap skipped'
        return result, info
    result['_label_day_utc'] = valid_days
    info['days_seen'] = int(result['_label_day_utc'].dropna().nunique())

    for side, direction_idx, cap_key in (('buy', 2, 'buy_cap'), ('sell', 0, 'sell_cap')):
        raw_cap = info.get(cap_key)
        if raw_cap in (None, '', 0, '0'):
            continue
        cap = int(raw_cap)
        if cap <= 0:
            continue
        strength_col = _side_strength_column(side)
        discard: list[int] = []
        side_mask = pd.to_numeric(result['direction_target'], errors='coerce') == direction_idx
        for _, group in result.loc[side_mask & result['_label_day_utc'].notna()].groupby('_label_day_utc', sort=False):
            if len(group) <= cap:
                continue
            if strength_col in group.columns:
                scores = pd.to_numeric(group[strength_col], errors='coerce').fillna(-1.0e18)
                keep = set(scores.nlargest(cap).index.astype(int).tolist())
            else:
                keep = set(group.index[:cap].astype(int).tolist())
            discard.extend([int(i) for i in group.index if int(i) not in keep])
        removed = _discard_positive_rows(result, discard, mode=mode, reason=f'{side}_daily_cap')
        info[f'{side}_removed'] = int(removed)

    result = result.drop(columns=['_label_day_utc'], errors='ignore')
    return result, info


def _candidate_strength(side_net: float, other_net: float, bars_to_outcome: int, horizon: int) -> float:
    """Rank clean candidates by strength, edge and speed.

    This score is only used for label filtering/ranking and is dropped before
    training data is saved, so it cannot leak future information into the model.
    """
    if not np.isfinite(side_net):
        return float('-inf')
    edge = max(0.0, float(side_net) - float(other_net)) if np.isfinite(other_net) else 0.0
    speed_bonus = 0.0
    if horizon > 0 and bars_to_outcome > 0:
        speed_bonus = max(0.0, float(horizon - bars_to_outcome + 1) / float(horizon))
    return float(side_net + edge + 0.25 * speed_bonus)


def _generate_barrier_direction_targets(df: pd.DataFrame, symbol: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Generate BUY/SELL/NO_TRADE direction targets.

    For every bar, both candidate trades are simulated over the configured horizon.
    In legacy mode the simulation uses the row close and future bid-like high/low
    candles, then subtracts spread afterwards.

    When labels.use_live_bid_ask_simulation is true, labels are generated much
    closer to live execution:
      - the model decides from closed bar i
      - entry can be taken at the next bar open
      - BUY enters at ask and exits against bid prices
      - SELL enters at bid and exits against ask prices
      - spread/slippage affect barrier hits, not just final net pips

    Optional label-quality filters can then keep only the strongest, de-duplicated
    positive setup rows. Discarded positives may be converted to NO_TRADE or
    IGNORE (-1). The training array builder skips IGNORE rows as supervised
    labels while still allowing surrounding bars to exist in sequence context.
    """
    df = df.copy()
    lcfg = _label_cfg(cfg)
    horizon = int(lcfg.get('horizon_bars', 18))
    spread_col = str(lcfg.get('spread_column', 'spread_points'))
    default_spread_points = symbol_default_spread_points(cfg, symbol, default=2.0)
    min_clean_win_net_pips = float(lcfg.get('min_clean_win_net_pips', 0.0))
    min_side_edge_pips = float(lcfg.get('min_side_edge_pips', lcfg.get('min_ev_edge_pips', 0.0)))
    use_live_bid_ask = bool(lcfg.get('use_live_bid_ask_simulation', False))
    entry_on_next_bar_open = bool(lcfg.get('entry_on_next_bar_open', use_live_bid_ask))
    max_spread_pips = symbol_max_spread_pips(cfg, symbol)

    open_col, high_col, low_col, close_col = _required_ohlc_columns(df)
    opens = pd.to_numeric(df[open_col], errors='coerce').to_numpy(float)
    highs = pd.to_numeric(df[high_col], errors='coerce').to_numpy(float)
    lows = pd.to_numeric(df[low_col], errors='coerce').to_numpy(float)
    closes = pd.to_numeric(df[close_col], errors='coerce').to_numpy(float)
    if spread_col in df.columns:
        spreads = pd.to_numeric(df[spread_col], errors='coerce').fillna(default_spread_points).to_numpy(float)
    else:
        spreads = np.full(len(df), default_spread_points, dtype=float)

    n = len(df)

    # Final direction class: 0=SELL, 1=NO_TRADE, 2=BUY. -1=IGNORE when
    # optional strong-setup filters discard a clustered/marginal positive row.
    direction = np.full(n, 1, dtype=np.int64)
    buy_net = np.full(n, np.nan, dtype=float)
    sell_net = np.full(n, np.nan, dtype=float)
    buy_outcome = np.full(n, -1, dtype=np.int64)
    sell_outcome = np.full(n, -1, dtype=np.int64)
    buy_bars = np.full(n, 0, dtype=np.int64)
    sell_bars = np.full(n, 0, dtype=np.int64)
    buy_strength = np.full(n, np.nan, dtype=float)
    sell_strength = np.full(n, np.nan, dtype=float)

    last_i = max(0, n - horizon - 1)
    for i in range(last_i):
        entry_idx = i + 1 if entry_on_next_bar_open else i
        if entry_idx >= n:
            continue
        entry_spread_points = spreads[entry_idx] if use_live_bid_ask else spreads[i]
        if max_spread_pips is not None and _spread_cost_pips(entry_spread_points, cfg) > max_spread_pips:
            continue

        if use_live_bid_ask:
            entry_bid = opens[entry_idx]
            if not np.isfinite(entry_bid):
                continue
            # Include the entry bar in the future path: a live order placed at
            # next-bar open can hit TP/SL within that same candle.
            future_start = entry_idx
            future_end = min(n, future_start + horizon)
            future_hi = highs[future_start:future_end]
            future_lo = lows[future_start:future_end]
            future_close = closes[future_start:future_end]
            future_spreads = spreads[future_start:future_end]
            if len(future_close) == 0:
                continue
            buy = _first_hit_outcome_live_bidask(
                symbol,
                entry_bid,
                future_hi,
                future_lo,
                future_close,
                future_spreads,
                'BUY',
                cfg,
                entry_spread_points=entry_spread_points,
            )
            sell = _first_hit_outcome_live_bidask(
                symbol,
                entry_bid,
                future_hi,
                future_lo,
                future_close,
                future_spreads,
                'SELL',
                cfg,
                entry_spread_points=entry_spread_points,
            )
            # In live bid/ask mode spread and slippage are already embedded in
            # entry/exit prices and barrier reachability. Do not subtract spread
            # again.
            bnet = float(buy.pips)
            snet = float(sell.pips)
        else:
            entry = closes[i]
            if not np.isfinite(entry):
                continue
            future_hi = highs[i + 1:i + 1 + horizon]
            future_lo = lows[i + 1:i + 1 + horizon]
            future_close = closes[i + 1:i + 1 + horizon]
            if len(future_close) == 0:
                continue
            buy = _first_hit_outcome(symbol, entry, future_hi, future_lo, future_close, 'BUY', cfg)
            sell = _first_hit_outcome(symbol, entry, future_hi, future_lo, future_close, 'SELL', cfg)
            cost = _spread_cost_pips(spreads[i], cfg)
            bnet = float(buy.pips - cost)
            snet = float(sell.pips - cost)

        buy_net[i] = bnet
        sell_net[i] = snet
        buy_outcome[i] = int(buy.outcome)
        sell_outcome[i] = int(sell.outcome)
        buy_bars[i] = int(buy.bars_to_outcome)
        sell_bars[i] = int(sell.bars_to_outcome)

        buy_is_clean_win = buy.outcome == 2 and bnet >= min_clean_win_net_pips
        sell_is_clean_win = sell.outcome == 2 and snet >= min_clean_win_net_pips
        if buy_is_clean_win:
            buy_strength[i] = _candidate_strength(bnet, snet, buy.bars_to_outcome, horizon)
        if sell_is_clean_win:
            sell_strength[i] = _candidate_strength(snet, bnet, sell.bars_to_outcome, horizon)

        # Choose the clean winner with better realised net pips.
        if buy_is_clean_win and (not sell_is_clean_win or bnet >= snet + min_side_edge_pips):
            direction[i] = 2
        elif sell_is_clean_win and (not buy_is_clean_win or snet >= bnet + min_side_edge_pips):
            direction[i] = 0

    out = df.iloc[:last_i].copy().reset_index(drop=True)
    keep = len(out)
    out['direction_target'] = direction[:keep]
    out['buy_candidate_net_pips'] = buy_net[:keep]
    out['sell_candidate_net_pips'] = sell_net[:keep]
    # Optional regression targets for the hierarchical edge/pips head. These are
    # future-derived labels, not live features. The feature selector blocks all
    # *_target columns, so they cannot leak into model inputs.
    out['buy_edge_pips_target'] = buy_net[:keep]
    out['sell_edge_pips_target'] = sell_net[:keep]
    out['buy_candidate_outcome'] = buy_outcome[:keep]
    out['sell_candidate_outcome'] = sell_outcome[:keep]
    out['buy_candidate_bars_to_outcome'] = buy_bars[:keep]
    out['sell_candidate_bars_to_outcome'] = sell_bars[:keep]
    out['buy_candidate_strength_score'] = buy_strength[:keep]
    out['sell_candidate_strength_score'] = sell_strength[:keep]
    out['candidate_strength_score'] = np.where(
        out['direction_target'].to_numpy(dtype=int) == 2,
        out['buy_candidate_strength_score'].to_numpy(dtype=float),
        np.where(
            out['direction_target'].to_numpy(dtype=int) == 0,
            out['sell_candidate_strength_score'].to_numpy(dtype=float),
            np.nan,
        ),
    )
    out['label_filter_status'] = 'kept'

    out, dedup_info = _apply_positive_deduplication(out, cfg)
    out, daily_cap_info = _apply_daily_positive_cap(out, cfg)
    positive_filter_info = {
        'deduplication': dedup_info,
        'daily_cap': daily_cap_info,
        'ignored_rows_after_filters': int((pd.to_numeric(out['direction_target'], errors='coerce') < 0).sum()),
    }

    # These future-derived diagnostics are used only inside label generation to
    # rank strong setup candidates. Drop them before returning so on-the-fly
    # training cannot accidentally use them as model features.
    auxiliary_cols = [
        c for c in out.columns
        if (
            'candidate_' in str(c).lower()
            or str(c).lower() in {'candidate_strength_score', 'label_filter_status'}
        )
    ]
    out = out.drop(columns=auxiliary_cols, errors='ignore')
    out.attrs['positive_label_filters'] = positive_filter_info
    return out

def generate_direction_targets(df: pd.DataFrame, symbol: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Generate the BUY/SELL/NO_TRADE direction label used by the simple model.

    Internally this uses the same live-style barrier simulation that replay uses, then returns a single direction_target column.
    """
    return _generate_barrier_direction_targets(df, symbol, cfg)

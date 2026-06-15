from __future__ import annotations

from typing import Iterable

# Default 20-symbol FX universe used by the direction policy project.
# Kept deliberately to spot FX pairs only; no indices, metals, crypto, or CFDs.
DEFAULT_FOREX_SYMBOLS = (
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY',
    'EURGBP', 'EURCHF', 'GBPCHF', 'GBPAUD', 'GBPCAD', 'EURAUD', 'EURCAD', 'AUDNZD','AUDCAD', 'AUDCHF', 'CADCHF', 'EURNZD', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDJPY', 'USDSGD', 'EURSGD', 'GBPSGD', 'AUDSGD', 'NZDSGD', 'SGDJPY', 'USDNOK', 'EURNOK', 'USDSEK', 'EURSEK', 'USDMXN', 'USDZAR',
)

VALID_FOREX = set(DEFAULT_FOREX_SYMBOLS)


def validate_forex_symbols(symbols: Iterable[str]) -> list[str]:
    clean = [str(s).upper().strip() for s in symbols]
    bad = [s for s in clean if s not in VALID_FOREX]
    if bad:
        raise ValueError(f'Unsupported/unknown forex symbols: {bad}')
    return clean


def point_for_symbol(symbol: str, cfg: dict) -> float:
    symbol = symbol.upper()
    overrides = ((cfg.get('trading') or {}).get('point_overrides') or {})
    if symbol in overrides:
        return float(overrides[symbol])
    return 0.001 if symbol.endswith('JPY') else 0.00001


def pips_from_price_delta(symbol: str, price_delta: float, cfg: dict) -> float:
    point = point_for_symbol(symbol, cfg)
    # For 5-digit non-JPY pairs, one pip = 10 points.
    # For 3-digit JPY pairs, one pip = 10 points.
    return price_delta / (point * 10.0)


def price_delta_from_pips(symbol: str, pips: float, cfg: dict) -> float:
    point = point_for_symbol(symbol, cfg)
    return pips * point * 10.0


def pip_size(symbol: str, cfg: dict | None = None) -> float:
    """Return one pip in price units for a forex symbol."""
    if cfg is not None:
        return point_for_symbol(symbol, cfg) * 10.0
    return 0.01 if str(symbol).upper().endswith('JPY') else 0.0001

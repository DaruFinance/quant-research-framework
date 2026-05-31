"""Panel-level strategy primitives (Phase 2 #8 and beyond).

Strategies in this subpackage take a ``PanelData`` + a rebalance bar
index and emit a per-asset weight vector. They compose alpha
computation + ranking + position construction (via the neutralize /
sizing helpers).
"""
from __future__ import annotations

from .long_short import LongShortBasket, momentum_alpha

__all__ = ["LongShortBasket", "momentum_alpha"]

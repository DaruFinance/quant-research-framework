"""Spread-aware stop-loss families (item #12, Phase 3).

Three stop primitives for pairs strategies:

- ``z_multiple_stop``: exit when the spread's rolling z-score
  exceeds a multiple ``z_mult``.
- ``half_life_multiple_stop``: exit if the position has been open for
  more than ``hl_mult`` half-lives without converging.
- ``breakdown_trigger_stop``: exit if the β estimate has moved by
  more than ``beta_jump`` between consecutive refits (the
  cointegration relationship is breaking down).

Pure functions; the caller decides whether and how to fire each
primitive. They read only the spread history and the current bar
state, never future data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np


class StopReason(Enum):
    Z_MULTIPLE = auto()
    HALF_LIFE_MULTIPLE = auto()
    BREAKDOWN = auto()


@dataclass(frozen=True)
class StopDecision:
    fired: bool
    reason: Optional[StopReason] = None
    detail: str = ""


def z_multiple_stop(
    spread: np.ndarray,
    t_idx: int,
    *,
    window: int = 60,
    z_mult: float = 3.0,
) -> StopDecision:
    """Fire if ``|z|`` exceeds ``z_mult`` at bar ``t_idx``.
    ``z`` is computed on the trailing ``window`` bars (excluding the
    current bar from the mean/std calculation to keep the test
    legitimately predictive)."""
    if t_idx < window:
        return StopDecision(fired=False)
    seg = spread[t_idx - window : t_idx]
    seg = seg[~np.isnan(seg)]
    if len(seg) < 2:
        return StopDecision(fired=False)
    mean = seg.mean()
    std = seg.std(ddof=1) + 1e-12
    z = (spread[t_idx] - mean) / std
    if abs(z) > z_mult:
        return StopDecision(
            fired=True, reason=StopReason.Z_MULTIPLE,
            detail=f"|z|={abs(z):.3f} > {z_mult}",
        )
    return StopDecision(fired=False)


def half_life_multiple_stop(
    entry_idx: int,
    t_idx: int,
    half_life: float,
    *,
    hl_mult: float = 5.0,
) -> StopDecision:
    """Fire if the position has been open for ``hl_mult`` half-lives.
    Uses bar count: a 1h-bar half-life of 8 implies the cap at
    hl_mult=5 is reached after 40 bars."""
    if half_life <= 0 or not np.isfinite(half_life):
        return StopDecision(fired=False)
    held_bars = t_idx - entry_idx
    if held_bars >= hl_mult * half_life:
        return StopDecision(
            fired=True, reason=StopReason.HALF_LIFE_MULTIPLE,
            detail=f"held={held_bars} >= {hl_mult}*hl={hl_mult * half_life:.1f}",
        )
    return StopDecision(fired=False)


def breakdown_trigger_stop(
    beta_prev: float,
    beta_new: float,
    *,
    beta_jump: float = 0.5,
) -> StopDecision:
    """Fire if the relative β change between two consecutive refits
    exceeds ``beta_jump`` (default 50%)."""
    if beta_prev == 0:
        return StopDecision(fired=False)
    rel = abs(beta_new - beta_prev) / abs(beta_prev)
    if rel > beta_jump:
        return StopDecision(
            fired=True, reason=StopReason.BREAKDOWN,
            detail=f"|Δβ|/|β|={rel:.3f} > {beta_jump}",
        )
    return StopDecision(fired=False)


__all__ = [
    "StopReason", "StopDecision",
    "z_multiple_stop", "half_life_multiple_stop", "breakdown_trigger_stop",
]

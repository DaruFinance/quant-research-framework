"""Funding-flip and basis-blowout triggers (item #39s, Phase 3).

These are pure point-in-time predicates over the loaded funding /
basis frames.  They consume only data at indices ``<= t`` and emit
``TriggerEvent`` records at every fire point.  The :class:`StatefulRunner`
shape (load → state → step → emit) lets the scheduler (#42) hold
trigger state across the backtest without re-reading the entire frame
per bar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TriggerEvent:
    time_s: int
    kind: str           # "funding_flip" | "basis_blowout"
    direction: int      # +1 = positive jump, -1 = negative jump
    prev: float
    curr: float
    extras: Optional[dict] = None


class FundingFlipTrigger:
    """Fires whenever ``sign(rate)`` changes across consecutive events.

    Parameters
    ----------
    min_magnitude: ignore tiny flips whose absolute current rate is
        below this threshold (default 0).  Helps avoid noise-driven
        firings on a feed that hovers around zero.
    """

    def __init__(self, min_magnitude: float = 0.0):
        self.min_magnitude = float(min_magnitude)

    def run(self, df: pd.DataFrame) -> List[TriggerEvent]:
        events: List[TriggerEvent] = []
        if len(df) < 2:
            return events
        rates = df["rate"].values
        times = df["time"].values
        prev_sign = np.sign(rates[0])
        for i in range(1, len(df)):
            curr = float(rates[i])
            curr_sign = np.sign(curr)
            if (curr_sign != prev_sign
                    and curr_sign != 0
                    and abs(curr) >= self.min_magnitude):
                events.append(TriggerEvent(
                    time_s=int(times[i]),
                    kind="funding_flip",
                    direction=int(curr_sign),
                    prev=float(rates[i - 1]),
                    curr=curr,
                ))
                prev_sign = curr_sign
            elif curr_sign != 0:
                prev_sign = curr_sign
        return events


class BasisBlowoutTrigger:
    """Fires when ``basis_bp`` crosses an N-sigma band relative to a
    trailing window ending at the candidate row (exclusive).

    Window length must be at least 5 to keep the sigma estimate
    meaningful; below the warmup the trigger does not fire.  Each
    candidate row's z-score is computed from rows ``[i-window:i]``
    so the row itself does not contribute to its own threshold.
    """

    def __init__(self, window: int = 60, z_thresh: float = 3.0):
        if window < 5:
            raise ValueError("window must be >= 5 for a usable sigma")
        self.window = int(window)
        self.z_thresh = float(z_thresh)

    def run(self, df: pd.DataFrame) -> List[TriggerEvent]:
        events: List[TriggerEvent] = []
        if "basis_bp" not in df.columns:
            raise ValueError("BasisBlowoutTrigger needs a basis_bp column")
        basis = df["basis_bp"].values
        times = df["time"].values
        for i in range(self.window, len(df)):
            window = basis[i - self.window:i]
            mu = float(window.mean())
            sd = float(window.std())
            if sd == 0.0 or not np.isfinite(sd):
                continue
            z = (float(basis[i]) - mu) / sd
            if abs(z) >= self.z_thresh:
                events.append(TriggerEvent(
                    time_s=int(times[i]),
                    kind="basis_blowout",
                    direction=int(np.sign(z)),
                    prev=mu,
                    curr=float(basis[i]),
                    extras={"z": float(z), "sigma": sd},
                ))
        return events

"""Persistent-funding-sign carry model (Phase 3 item #43)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .base import SignalEmission


class PersistentFundingSignModel:
    """Long carry on a sustained-positive funding regime (i.e. shorts
    pay longs → we are long the underlying); short carry on sustained-
    negative.  Below the minimum streak we emit a flat signal.

    Only events with ``time_s <= t_s`` enter the streak count; the
    function never peeks at future funding rows.
    """

    def __init__(self, min_streak: int = 3):
        if min_streak < 1:
            raise ValueError("min_streak must be >= 1")
        self.min_streak = int(min_streak)

    def signal_at(
        self,
        funding_df: pd.DataFrame,
        t_s: int,
    ) -> SignalEmission:
        mask = funding_df["time"].values <= t_s
        if not mask.any():
            return SignalEmission(time_s=int(t_s), direction=0, strength=0.0,
                                    model="persistent_sign")
        slc = funding_df.loc[mask, "rate"].values
        last_sign = int(np.sign(slc[-1]))
        if last_sign == 0:
            return SignalEmission(time_s=int(t_s), direction=0, strength=0.0,
                                    model="persistent_sign")
        streak = 1
        for v in slc[-2::-1]:
            if int(np.sign(v)) == last_sign:
                streak += 1
            else:
                break
        if streak < self.min_streak:
            return SignalEmission(
                time_s=int(t_s), direction=0,
                strength=float(streak) / float(self.min_streak),
                inputs={"streak": float(streak),
                          "last_rate": float(slc[-1])},
                model="persistent_sign",
            )
        return SignalEmission(
            time_s=int(t_s),
            direction=int(-last_sign),  # carry-collecting side
            strength=float(streak),
            inputs={"streak": float(streak),
                      "last_rate": float(slc[-1])},
            model="persistent_sign",
        )

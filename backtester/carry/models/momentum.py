"""Funding-momentum carry model (Phase 3 item #43)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import SignalEmission


class FundingMomentumModel:
    """Z-score of funding rate vs. its trailing ``window`` events.

    Beyond ``|z| >= z_thresh`` we emit a signal in the carry-collecting
    direction (short the payer side).  Strength is ``|z|``.

    All inputs come from rows with ``time <= t_s``; the trailing
    window does not include the current row's own sign in its own
    estimate (so polluting `> t` cannot flip the decision at `t`).
    """

    def __init__(self, window: int = 20, z_thresh: float = 1.5):
        if window < 5:
            raise ValueError("window must be >= 5 for a usable z-score")
        self.window = int(window)
        self.z_thresh = float(z_thresh)

    def signal_at(
        self,
        funding_df: pd.DataFrame,
        t_s: int,
    ) -> SignalEmission:
        mask = funding_df["time"].values <= t_s
        if not mask.any():
            return SignalEmission(time_s=int(t_s), direction=0, strength=0.0,
                                    model="funding_momentum")
        rates = funding_df.loc[mask, "rate"].values
        if len(rates) <= self.window:
            return SignalEmission(time_s=int(t_s), direction=0, strength=0.0,
                                    model="funding_momentum")
        window = rates[-(self.window + 1):-1]
        mu = float(window.mean())
        sd = float(window.std())
        if sd == 0.0 or not np.isfinite(sd):
            return SignalEmission(time_s=int(t_s), direction=0, strength=0.0,
                                    model="funding_momentum")
        z = (float(rates[-1]) - mu) / sd
        if abs(z) < self.z_thresh:
            return SignalEmission(
                time_s=int(t_s), direction=0, strength=float(abs(z)),
                inputs={"z": float(z), "mu": mu, "sd": sd},
                model="funding_momentum",
            )
        return SignalEmission(
            time_s=int(t_s),
            direction=int(-np.sign(z)),
            strength=float(abs(z)),
            inputs={"z": float(z), "mu": mu, "sd": sd,
                      "rate": float(rates[-1])},
            model="funding_momentum",
        )

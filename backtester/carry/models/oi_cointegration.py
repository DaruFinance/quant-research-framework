"""Funding × OI joint-move model (Phase 3 item #43)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import SignalEmission


class FundingOICointegrationModel:
    """Joint-move detector: funding and OI moving same direction
    (relative to their trailing means) reinforce the signal; opposite
    direction cancels.  Both inputs are resampled to the funding
    cadence at lookup time; OI values strictly at or before the most
    recent funding event used.
    """

    def __init__(self, window: int = 20, scale: float = 1.0):
        if window < 5:
            raise ValueError("window must be >= 5")
        self.window = int(window)
        self.scale = float(scale)

    def signal_at(
        self,
        funding_df: pd.DataFrame,
        oi_df: pd.DataFrame,
        t_s: int,
    ) -> SignalEmission:
        f_mask = funding_df["time"].values <= t_s
        if not f_mask.any() or f_mask.sum() <= self.window:
            return SignalEmission(time_s=int(t_s), direction=0, strength=0.0,
                                    model="funding_oi_coint")

        f_times = funding_df.loc[f_mask, "time"].values
        f_rates = funding_df.loc[f_mask, "rate"].values
        recent_f_times = f_times[-(self.window + 1):]
        recent_f_rates = f_rates[-(self.window + 1):]

        oi_values: list[float] = []
        oi_arr = oi_df["time"].values
        oi_vals = oi_df["open_interest"].values
        for ft in recent_f_times:
            mask = oi_arr <= ft
            if not mask.any():
                return SignalEmission(time_s=int(t_s), direction=0,
                                        strength=0.0,
                                        model="funding_oi_coint")
            oi_values.append(float(oi_vals[np.flatnonzero(mask)[-1]]))

        f_arr = np.asarray(recent_f_rates, dtype=float)
        o_arr = np.asarray(oi_values, dtype=float)
        f_curr = float(f_arr[-1])
        o_curr = float(o_arr[-1])
        f_base = f_arr[:-1]
        o_base = o_arr[:-1]
        f_mu, f_sd = float(f_base.mean()), float(f_base.std())
        o_mu, o_sd = float(o_base.mean()), float(o_base.std())
        if f_sd == 0.0 or o_sd == 0.0:
            return SignalEmission(time_s=int(t_s), direction=0, strength=0.0,
                                    model="funding_oi_coint")
        z_f = (f_curr - f_mu) / f_sd
        z_o = (o_curr - o_mu) / o_sd
        joint = float(np.sign(z_f) * np.sign(z_o)
                       * min(abs(z_f), abs(z_o)) * self.scale)
        if joint == 0.0:
            return SignalEmission(time_s=int(t_s), direction=0,
                                    strength=0.0,
                                    inputs={"z_f": z_f, "z_o": z_o},
                                    model="funding_oi_coint")
        return SignalEmission(
            time_s=int(t_s),
            direction=int(-np.sign(z_f)) if joint > 0 else 0,
            strength=float(abs(joint)),
            inputs={"z_f": float(z_f), "z_o": float(z_o), "joint": joint},
            model="funding_oi_coint",
        )

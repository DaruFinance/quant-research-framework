"""Funding-signal model library (item #43, Phase 3).

Each model exposes a ``signal_at(funding_df, t_s, **kw) -> SignalEmission``
contract.  The emission is a directional decision (``+1``/``0``/``-1``)
plus enough metadata for the trade-audit harness to recompute the
decision at ``t_s`` from the inputs available at that time.

Three models ship at launch (more can be added later):

- :class:`PersistentFundingSignModel` — if the funding rate has held
  one sign for at least ``min_streak`` consecutive events, emit a
  signal in the *carry-collecting* direction (i.e. short the side
  that pays).
- :class:`FundingMomentumModel` — z-score of the funding rate against
  its trailing ``window`` events.  Strong-positive → long carry;
  strong-negative → short.
- :class:`FundingOICointegrationModel` — detects co-movement between
  funding sign and OI sign over a trailing window; signal scales
  with the magnitude of the joint move.
"""
from __future__ import annotations

from .base import SignalEmission
from .persistent_sign import PersistentFundingSignModel
from .momentum import FundingMomentumModel
from .oi_cointegration import FundingOICointegrationModel

__all__ = [
    "SignalEmission",
    "PersistentFundingSignModel",
    "FundingMomentumModel",
    "FundingOICointegrationModel",
]

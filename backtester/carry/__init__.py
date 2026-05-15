"""Carry / basis / funding plugin (Phase 3 T6 — items #38..#43).

This subpackage hosts the loaders, triggers, scheduler, and signal
models for the T6 strategy tree (funding, basis, OI, on-chain).
Stays optional behind the ``[pairs]`` / ``[carry]`` extras of the
top-level package — see ``pyproject.toml``.

Public surface re-exported here for convenience. Individual modules
keep their lazy import shape so importing ``backtester.carry`` does
not pull statsmodels / scipy unless a user actually uses a function
that needs them.
"""
from __future__ import annotations

from .basis import BasisRecord, load_basis
from .funding import FundingEvent, load_funding, next_funding_time
from .models import (
    FundingMomentumModel,
    FundingOICointegrationModel,
    PersistentFundingSignModel,
    SignalEmission,
)
from .oi import OIRecord, load_oi
from .onchain import OnChainSnapshot, load_onchain
from .scheduler import EventDrivenScheduler, ScheduledRebalance
from .triggers import (
    BasisBlowoutTrigger,
    FundingFlipTrigger,
    TriggerEvent,
)

__all__ = [
    # #38
    "FundingEvent", "load_funding", "next_funding_time",
    # #39
    "BasisRecord", "load_basis",
    # #39s
    "FundingFlipTrigger", "BasisBlowoutTrigger", "TriggerEvent",
    # #40
    "OIRecord", "load_oi",
    # #41
    "OnChainSnapshot", "load_onchain",
    # #42
    "EventDrivenScheduler", "ScheduledRebalance",
    # #43
    "PersistentFundingSignModel", "FundingMomentumModel",
    "FundingOICointegrationModel", "SignalEmission",
]

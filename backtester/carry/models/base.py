"""Common signal-emission contract shared by all carry models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class SignalEmission:
    time_s: int
    direction: int       # +1 long carry / -1 short carry / 0 flat
    strength: float      # magnitude in [0, +inf), model-specific
    inputs: Dict[str, float] = field(default_factory=dict)
    model: str = ""

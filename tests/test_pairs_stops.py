"""Tests for spread-aware stop-loss families (item #12, Phase 3)."""
from __future__ import annotations

import numpy as np
import pytest

from backtester.pairs import (
    StopReason,
    breakdown_trigger_stop,
    half_life_multiple_stop,
    z_multiple_stop,
)


def test_z_stop_does_not_fire_within_warmup():
    rng = np.random.default_rng(0)
    spread = rng.normal(0, 1, size=200)
    out = z_multiple_stop(spread, t_idx=30, window=60, z_mult=3.0)
    assert not out.fired


def test_z_stop_fires_on_extreme_value():
    spread = np.zeros(200)
    spread[150] = 10.0  # extreme outlier
    out = z_multiple_stop(spread, t_idx=150, window=60, z_mult=3.0)
    assert out.fired
    assert out.reason == StopReason.Z_MULTIPLE


def test_z_stop_does_not_fire_on_calm_series():
    rng = np.random.default_rng(0)
    spread = rng.normal(0, 0.001, size=200)
    out = z_multiple_stop(spread, t_idx=150, window=60, z_mult=3.0)
    assert not out.fired


def test_half_life_stop_does_not_fire_before_cap():
    out = half_life_multiple_stop(entry_idx=100, t_idx=120,
                                    half_life=10.0, hl_mult=5.0)
    assert not out.fired  # 20 bars < 50 cap


def test_half_life_stop_fires_after_cap():
    out = half_life_multiple_stop(entry_idx=100, t_idx=160,
                                    half_life=10.0, hl_mult=5.0)
    assert out.fired
    assert out.reason == StopReason.HALF_LIFE_MULTIPLE


def test_half_life_stop_no_fire_on_invalid_hl():
    for bad in (0, -1, float("inf"), float("nan")):
        out = half_life_multiple_stop(entry_idx=100, t_idx=200,
                                        half_life=bad, hl_mult=5.0)
        assert not out.fired


def test_breakdown_stop_fires_on_large_beta_jump():
    out = breakdown_trigger_stop(beta_prev=1.0, beta_new=1.7, beta_jump=0.5)
    assert out.fired
    assert out.reason == StopReason.BREAKDOWN


def test_breakdown_stop_does_not_fire_on_small_beta_drift():
    out = breakdown_trigger_stop(beta_prev=1.0, beta_new=1.2, beta_jump=0.5)
    assert not out.fired


def test_breakdown_stop_zero_beta_safe():
    out = breakdown_trigger_stop(beta_prev=0.0, beta_new=1.0, beta_jump=0.5)
    assert not out.fired

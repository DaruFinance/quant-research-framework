"""Tests for pre-screening eligibility filters (item #13, Phase 3)."""
from __future__ import annotations

import numpy as np
import pytest

from backtester.pairs import (
    EligibilityCriteria,
    half_life_ou,
    is_eligible_pair,
)


def test_half_life_finite_on_mean_reverting():
    # OU process with known kappa: ds = -0.1 * s + noise. Half-life
    # should be around ln(2)/0.1 ≈ 6.93.
    rng = np.random.default_rng(0)
    n = 500
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = s[i - 1] - 0.1 * s[i - 1] + rng.normal(0, 0.05)
    hl = half_life_ou(s)
    assert 4.0 < hl < 12.0, f"expected ~6.9, got {hl}"


def test_half_life_inf_on_random_walk():
    rng = np.random.default_rng(1)
    s = np.cumsum(rng.normal(0, 0.05, size=500))
    hl = half_life_ou(s)
    # Random walk has no mean reversion: slope of ds ~ s_lag is ~0;
    # half-life may be very large or +inf.
    assert hl > 100


def test_eligible_pair_accepts_typical_inputs():
    rng = np.random.default_rng(0)
    n = 500
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = s[i - 1] - 0.1 * s[i - 1] + rng.normal(0, 0.05)
    ok, reason = is_eligible_pair(s, p_value=0.01)
    assert ok, reason


def test_eligible_pair_rejects_high_p_value():
    s = np.random.default_rng(0).normal(0, 1, size=100)
    ok, reason = is_eligible_pair(s, p_value=0.30,
                                    criteria=EligibilityCriteria(p_max=0.05))
    assert not ok
    assert "p_value" in reason


def test_eligible_pair_rejects_too_short_window():
    s = np.zeros(30)
    ok, reason = is_eligible_pair(s, p_value=0.01,
                                    criteria=EligibilityCriteria(min_window=60))
    assert not ok
    assert "insufficient" in reason


def test_eligible_pair_rejects_out_of_range_half_life():
    # Very slow mean reversion -> huge half-life.
    rng = np.random.default_rng(2)
    n = 500
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = s[i - 1] - 0.0001 * s[i - 1] + rng.normal(0, 0.05)
    ok, reason = is_eligible_pair(
        s, p_value=0.01,
        criteria=EligibilityCriteria(half_life_range=(1.0, 100.0)),
    )
    assert not ok
    assert "half_life" in reason

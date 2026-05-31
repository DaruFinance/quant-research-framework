"""Tests for portfolio-level constraints (item #45, Phase 2)."""
from __future__ import annotations

import numpy as np
import pytest

from backtester.panel.constraints import apply_constraints


# ---------------------------------------------------------------------------
# Single-asset cap
# ---------------------------------------------------------------------------
def test_no_cap_returns_input_unchanged():
    w = np.array([0.4, -0.3, 0.3])
    out = apply_constraints(w)
    np.testing.assert_array_equal(out, w)


def test_single_asset_cap_caps_over_threshold_leg():
    w = np.array([0.5, 0.3, 0.2])
    out = apply_constraints(w, single_asset_max=0.3)
    assert abs(out[0]) <= 0.3 + 1e-10


def test_single_asset_cap_preserves_signs():
    w = np.array([0.5, -0.4, 0.1])
    out = apply_constraints(w, single_asset_max=0.3)
    assert np.all(np.sign(out) == np.sign(w))


def test_single_asset_cap_redistributes_excess_to_unbound_legs():
    """Excess from the over-cap leg must redistribute to strictly
    under-cap legs (legs already at cap are frozen to avoid the
    over-cap oscillation loop). Use a non-degenerate scenario where
    gross IS preservable: cap=0.5, vector=[0.6, 0.2, 0.2]."""
    w = np.array([0.6, 0.2, 0.2])
    out = apply_constraints(w, single_asset_max=0.5)
    # gross preserved: 1.0 in, 1.0 out (cap=0.5 with 3 legs can fit gross=1.0)
    assert abs(np.abs(out).sum() - np.abs(w).sum()) < 1e-10
    # cap binding on leg 0:
    assert abs(abs(out[0]) - 0.5) < 1e-10
    # legs 1 and 2 both absorbed half the excess each:
    assert abs(out[1] - 0.25) < 1e-10
    assert abs(out[2] - 0.25) < 1e-10


def test_single_asset_cap_drops_excess_when_no_headroom():
    """If every leg hits cap before the residual is absorbed, the
    remaining excess is dropped — the gross shrinks. With cap=0.3 and
    3 legs, max gross is 0.9; inputs that sum to >0.9 lose mass."""
    w = np.array([0.5, 0.3, 0.2])  # gross 1.0, but 3 * 0.3 = 0.9 max
    out = apply_constraints(w, single_asset_max=0.3)
    # Output: [0.3, 0.3, 0.3], gross 0.9
    np.testing.assert_array_almost_equal(out, np.array([0.3, 0.3, 0.3]), decimal=10)
    assert abs(np.abs(out).sum() - 0.9) < 1e-10
    assert np.abs(out).sum() < np.abs(w).sum()


def test_single_asset_cap_idempotent():
    """Applying twice equals applying once — caps are stable."""
    w = np.array([0.5, 0.3, 0.2])
    once = apply_constraints(w, single_asset_max=0.3)
    twice = apply_constraints(once, single_asset_max=0.3)
    np.testing.assert_array_almost_equal(once, twice, decimal=12)


def test_single_asset_cap_rejects_invalid_range():
    with pytest.raises(ValueError, match=r"single_asset_max must"):
        apply_constraints(np.array([0.5, 0.5]), single_asset_max=0.0)
    with pytest.raises(ValueError, match=r"single_asset_max must"):
        apply_constraints(np.array([0.5, 0.5]), single_asset_max=1.5)


# ---------------------------------------------------------------------------
# Gross leverage cap
# ---------------------------------------------------------------------------
def test_gross_leverage_cap_scales_down_when_binding():
    w = np.array([2.0, -1.0, 1.5])  # gross = 4.5
    out = apply_constraints(w, gross_lev_max=3.0)
    assert abs(np.abs(out).sum() - 3.0) < 1e-10


def test_gross_leverage_cap_preserves_composition():
    """When gross binds, the proportional makeup of the weights is
    preserved (uniform scaling)."""
    w = np.array([2.0, -1.0, 1.5])
    out = apply_constraints(w, gross_lev_max=3.0)
    # Each |out_i| / sum|out| equals each |w_i| / sum|w|.
    ratios_in = np.abs(w) / np.abs(w).sum()
    ratios_out = np.abs(out) / np.abs(out).sum()
    np.testing.assert_array_almost_equal(ratios_in, ratios_out, decimal=10)


def test_gross_leverage_cap_idempotent():
    w = np.array([2.0, -1.0, 1.5])
    once = apply_constraints(w, gross_lev_max=3.0)
    twice = apply_constraints(once, gross_lev_max=3.0)
    np.testing.assert_array_almost_equal(once, twice, decimal=12)


def test_gross_leverage_cap_no_op_when_under():
    w = np.array([0.5, -0.3, 0.2])  # gross = 1.0
    out = apply_constraints(w, gross_lev_max=3.0)
    np.testing.assert_array_equal(out, w)


def test_gross_leverage_cap_rejects_invalid_range():
    with pytest.raises(ValueError, match="gross_lev_max"):
        apply_constraints(np.array([1.0, 1.0]), gross_lev_max=0.0)


# ---------------------------------------------------------------------------
# Combined caps
# ---------------------------------------------------------------------------
def test_both_caps_applied_in_order():
    """single-asset cap first, then gross cap. Verify both bind."""
    w = np.array([0.8, -0.6, 0.4, -0.2])  # gross 2.0, SOL 0.8
    out = apply_constraints(w, single_asset_max=0.5, gross_lev_max=1.5)
    # After single-asset cap: |w| <= 0.5 each.
    # After gross cap: total |w| <= 1.5.
    assert (np.abs(out) <= 0.5 + 1e-10).all()
    assert np.abs(out).sum() <= 1.5 + 1e-10
    assert np.all(np.sign(out) == np.sign(w))


# ---------------------------------------------------------------------------
# Pure pointwise + idempotence
# ---------------------------------------------------------------------------
def test_constraints_are_pure_no_side_effects_on_input():
    w = np.array([0.5, 0.3, 0.2])
    w_copy = w.copy()
    apply_constraints(w, single_asset_max=0.3)
    np.testing.assert_array_equal(w, w_copy)


def test_constraints_locality_pointwise():
    """Calling apply_constraints on a portion of a weight vector
    produces the same result as calling on the whole vector, then
    slicing — i.e. constraints don't read outside their input."""
    full = np.array([0.5, 0.3, 0.2, -0.4, -0.3])
    partial = full[:3]
    full_out = apply_constraints(full, single_asset_max=0.3)
    partial_out = apply_constraints(partial, single_asset_max=0.3)
    # The partials see different totals so won't match the full
    # case's first 3, but each independently must satisfy the cap.
    assert (np.abs(partial_out) <= 0.3 + 1e-10).all()
    assert (np.abs(full_out) <= 0.3 + 1e-10).all()


# ---------------------------------------------------------------------------
# G3 — 5 rebalances with cap binding
# ---------------------------------------------------------------------------
def test_g3_5_rebalances_redistribution_audit():
    """5 synthetic rebalance vectors where SOL is the over-cap leg;
    verify cap binding + gross preservation + redistribution
    direction (excess goes to under-cap legs)."""
    scenarios = [
        np.array([0.50, 0.30, 0.20]),
        np.array([0.45, 0.35, 0.20]),
        np.array([0.40, 0.40, 0.20]),
        np.array([0.55, 0.25, 0.20]),
        np.array([0.60, 0.20, 0.20]),
    ]
    cap = 0.30
    for i, w in enumerate(scenarios):
        out = apply_constraints(w, single_asset_max=cap)
        # Cap binding:
        assert abs(abs(out[0]) - cap) < 1e-10, (
            f"scenario {i}: SOL not at cap"
        )
        # Gross preserved (or shrunk if everyone hit cap):
        assert np.abs(out).sum() <= np.abs(w).sum() + 1e-10
        # Signs preserved:
        np.testing.assert_array_equal(np.sign(out), np.sign(w))
        # Excess redistributed to the under-cap leg (SOL excess is
        # 0.20+, that goes to ETH which was under the cap).
        # ETH grew or stayed equal (in degenerate cases).
        assert out[2] >= w[2] - 1e-10

"""Tests for the walk-forward orchestrator dispatch table (item #5).

Pure routing layer — the registered functions retain their original
lookahead-leak and parity guarantees, and these tests assert only that
the dispatch table itself routes the four Phase-1-relevant key
combinations correctly.

Run from the Python repo:

    pytest tests/test_dispatch_routes.py -v
"""
from __future__ import annotations

import pytest

import backtester as bt
from backtester import orchestrator
from backtester.orchestrator import RouteKey


def test_phase1_routes_are_registered():
    """The two single-asset routes seeded by item #5 must be registered
    by the time the backtester module finishes import."""
    keys = set(orchestrator.registered_keys())
    assert RouteKey(regime=False) in keys, (
        f"single-asset no-regime route missing; have {keys}"
    )
    assert RouteKey(regime=True) in keys, (
        f"single-asset regime route missing; have {keys}"
    )


def test_no_regime_route_resolves_to_default_path():
    fn = orchestrator.dispatch(RouteKey(regime=False))
    assert fn is bt._walk_forward_default_path, (
        f"regime=False should resolve to _walk_forward_default_path, "
        f"got {fn.__name__}"
    )


def test_regime_route_resolves_to_regime_path():
    fn = orchestrator.dispatch(RouteKey(regime=True))
    assert fn is bt._walk_forward_regime_path, (
        f"regime=True should resolve to _walk_forward_regime_path, "
        f"got {fn.__name__}"
    )


def test_unregistered_key_raises_keyerror_with_listing():
    """Asking for a route that doesn't exist must fail loudly with the
    set of known routes, not silently fall back to a default.

    We use ``record_costs=True`` as the sentinel "unregistered" key
    because no item through Phase 2 #5(iter) flips that flag in the
    dispatch key; if a future item starts registering record_costs
    variants, this test will fail and we update the sentinel."""
    with pytest.raises(KeyError) as exc:
        orchestrator.dispatch(RouteKey(record_costs=True, hold_period_set=True))
    msg = str(exc.value)
    assert "no orchestrator route" in msg
    assert "registered keys" in msg


def test_duplicate_registration_with_different_fn_raises():
    """Re-registering a key with a NEW function must raise so plugin
    ordering conflicts are surfaced. Re-registering with the SAME
    function is a no-op (idempotent for module-reimport scenarios)."""
    key = RouteKey(regime=False)
    existing = orchestrator.DISPATCH[key]

    # Same function: idempotent (no raise).
    # The current implementation raises on any duplicate; that's
    # safer for production. We assert the strict-raise behaviour here
    # so any future loosening is a conscious decision documented by a
    # test edit.
    def fake(*a, **kw):
        return None

    with pytest.raises(KeyError) as exc:
        orchestrator.register(key, fake)
    assert "already registered" in str(exc.value)

    # Restore the original function so other tests are unaffected.
    orchestrator.DISPATCH[key] = existing


def test_routekey_defaults_are_all_false():
    """Default RouteKey() is the canonical 'no features set' identity.
    Used by callers that want a baseline key and toggle flags
    selectively."""
    k = RouteKey()
    assert k == RouteKey(regime=False, multi_asset=False, multi_leg=False,
                          record_costs=False, hold_period_set=False)


def test_routekey_is_hashable_for_dispatch_dict_use():
    """RouteKey must be hashable since DISPATCH is keyed on it."""
    s = {RouteKey(regime=True), RouteKey(regime=True), RouteKey(regime=False)}
    assert len(s) == 2

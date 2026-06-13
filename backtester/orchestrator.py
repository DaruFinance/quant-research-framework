"""Walk-forward orchestrator dispatch table (item #5).

Replaces the if/else inside ``_walk_forward_impl`` with a registry keyed
on a 5-bool ``RouteKey``. Phase 1 registers two entries:

- ``RouteKey(regime=False)``   → ``_walk_forward_default_path``
- ``RouteKey(regime=True)``    → ``_walk_forward_regime_path``

Subsequent phases plug additional routes:

- Phase 2 #5(iter) registers ``RouteKey(multi_asset=True, ...)``
  variants that iterate the same global WFO window grid over an
  asset panel.
- Phase 3 plugs pair-as-unit and cohort routes via #5(iter) too.
- Item #28 / #34 flip ``multi_leg=True``; item #3 flips
  ``record_costs=True`` (today only gates stdout exposure, not
  routing); item #46 flips ``hold_period_set=True``.

The dispatch is **pure routing**: each registered function preserves
the existing no-look-ahead and parity guarantees of the path it wraps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class RouteKey:
    """Composite key for orchestrator dispatch. All flags default to
    False so call sites can construct partial keys (``RouteKey(regime=True)``
    is enough to pick the regime+WFO route on a single-asset run)."""
    regime: bool = False
    multi_asset: bool = False
    multi_leg: bool = False
    record_costs: bool = False
    hold_period_set: bool = False


# Process-global registry. Population happens at module-import time in
# `backtester/__init__.py` after the path implementations are defined.
DISPATCH: Dict[RouteKey, Callable[..., Any]] = {}


def register(key: RouteKey, fn: Callable[..., Any]) -> None:
    """Register an orchestrator implementation for ``key``.

    Duplicate registration raises ``KeyError`` to surface conflicting
    plugin orderings rather than silently last-write-wins.
    """
    if key in DISPATCH:
        raise KeyError(f"orchestrator route {key} is already registered")
    DISPATCH[key] = fn


def dispatch(key: RouteKey) -> Callable[..., Any]:
    """Resolve a dispatch key to its registered function.

    Raises ``KeyError`` with the set of currently-registered keys when
    ``key`` is missing — surface "no route for this feature combination
    yet" loudly rather than fall back to a default that hides a bug.
    """
    if key not in DISPATCH:
        registered = sorted(DISPATCH.keys(), key=repr)
        raise KeyError(
            f"no orchestrator route for {key}; "
            f"registered keys: {registered}"
        )
    return DISPATCH[key]


def registered_keys() -> list[RouteKey]:
    """Snapshot of currently registered keys. Used by tests to verify
    the routing surface."""
    return list(DISPATCH.keys())


def clear_registry_for_test() -> None:
    """Test-only escape hatch. Production code never calls this."""
    DISPATCH.clear()

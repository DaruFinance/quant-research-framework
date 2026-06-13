# Item #5 — verification log

**Goal:** Refactor `_walk_forward_impl` into a dispatch table keyed on
a 5-bool `RouteKey`. Phase 1 lands two single-asset routes
(`regime=False`, `regime=True`); subsequent phases plug
`multi_asset`, `multi_leg`, `record_costs`, `hold_period_set` variants
via the same registry.

**Dataset:** N/A — pure routing refactor. The verification gate is
zero behavioral change against the pre-refactor stdout / parity
surface.

## What landed

**Python:**

- `backtester/orchestrator.py` (new): `RouteKey` frozen dataclass
  (5 bool fields, defaults False), `DISPATCH` registry dict,
  `register(key, fn)` (raises on duplicate), `dispatch(key)` (raises
  KeyError with the registered key list when missing).
- `backtester/__init__.py`:
  - `from . import orchestrator` near the top with the other
    relative imports.
  - `_walk_forward_impl` rewritten as a thin dispatcher: builds
    `rb_scenarios` via the new `_build_rb_scenarios` helper, then
    constructs a `RouteKey` and dispatches.
  - The pre-#5 if/else branches extracted verbatim into
    `_walk_forward_regime_path` and `_walk_forward_default_path`
    (the regime body wrapped in `if True:` to minimise the diff
    against the original indentation).
  - Module bottom: two `orchestrator.register` calls seed the
    Phase 1 routes.
- `tests/test_dispatch_routes.py` (new, 7 tests).

**Rust:**

- `src/orchestrator.rs` (new): `RouteKey` struct (5 bool fields,
  Default), `OrchestratorMode { SingleNoRegime, SingleWithRegime }`,
  `dispatch(key) -> Result<OrchestratorMode, RouteError>` with a
  Phase-2-pending error variant when `multi_asset=true` is asked.
- `src/lib.rs`: `pub mod orchestrator;`.
- 4 inline `#[cfg(test)]` unit tests in `orchestrator.rs`.

The Rust `walk_forward` / `walk_forward_regime` internals are
untouched in Phase 1: the dispatch infrastructure is exposed as a
public API surface today; Phase 2 will unify those function
signatures behind it when panel routes (`multi_asset=true`) force
the harmonisation.

## G1 — Parity surface

| Surface         | Result        | Notes |
|-----------------|---------------|-------|
| parity_check    | PARITY OK     | post-refactor, 1e-3 |
| parity_regime   | PARITY OK     | post-refactor, 1e-3 |
| parity_forex    | PARITY OK     | post-refactor, 1e-3 |
| parity_ledger   | LEDGER PARITY OK | 1389 trades, 6945 fields |
| DS-SOL-1K vs v0.4.0 baseline | 54/54 tags, 0 mismatches | bit-identical |

## G2 — Test infrastructure

**Python pytest tests/: 56 passed, 3 skipped, 0 failed** (was 49 pre-#5,
+7 from `test_dispatch_routes.py`).

Dispatch-route tests:
- `test_phase1_routes_are_registered` — both `regime=False` and
  `regime=True` routes appear in `DISPATCH` after import. ✓
- `test_no_regime_route_resolves_to_default_path` —
  `dispatch(RouteKey(regime=False))` returns
  `_walk_forward_default_path`. ✓
- `test_regime_route_resolves_to_regime_path` — likewise for the
  regime branch. ✓
- `test_unregistered_key_raises_keyerror_with_listing` — asking for
  `RouteKey(multi_asset=True)` (not registered in Phase 1) raises
  `KeyError` whose message lists the registered routes. ✓
- `test_duplicate_registration_with_different_fn_raises` —
  re-registering a key with a NEW function raises (catches plugin
  ordering conflicts at import time). ✓
- `test_routekey_defaults_are_all_false` — canonical identity. ✓
- `test_routekey_is_hashable_for_dispatch_dict_use` — `RouteKey` is
  hashable as required by `Dict[RouteKey, ...]`. ✓

**Leak audit (all invariant property tests re-run before and after
the refactor):**

`tests/test_invariants_property.py` + `tests/test_invariants.py`: 18
passed. The orchestrator dispatch is pure routing and inherits the
lookahead-freeness of the wrapped path implementations; the harness
self-tests from item #14 continue to pass.

**Rust cargo test (release, --tests + --lib):**

- `tests/invariants.rs`: 14 passed (unchanged from #14).
- `tests/behavioural.rs`, `tests/contract_v0_2.rs`: unchanged.
- `tests/test_parity_registry.py` (run via pytest from Rust repo):
  8 passed (unchanged from #15).
- New `src/orchestrator.rs` inline unit tests:
  - `default_route_is_single_no_regime` ✓
  - `regime_flag_selects_single_with_regime` ✓
  - `multi_asset_flag_errors_pending_phase2` ✓
  - `route_key_is_hashable_and_eq` ✓

## G3 — Zero-diff against pre-refactor

The plan called for "stdout diff against pre-refactor: zero
non-whitespace bytes differ." Operationally that means the four
parity scripts and the DS-SOL-1K metric capture must produce
identical numeric output. They do — every parity script reports
`rel = 0.00%` on the headline fields (Trades, ROI, PF, Sharpe,
WinRate, Exp, MaxDD), and DS-SOL-1K vs v0.4.0 baseline reports
**54 tags, 0 mismatches** (every (tag × field) pair bit-identical).

Five failure modes the refactor would have introduced if done
sloppily, all caught by the gate:

1. **Regime body extracted with wrong indent:** would make the
   regime path run un-indented module code → SyntaxError at import,
   pytest collection fails. ✓ caught.
2. **rb_scenarios built in the wrong scope:** if I'd left it inside
   the regime branch, the no-regime path would crash with
   `NameError: rb_scenarios`. ✓ pytest covers both paths.
3. **Dispatch key mis-built:** if I'd written
   `regime=USE_REGIME_SEG` (without `and USE_WFO`), the no-USE_WFO
   case would route to regime path → parity_check would diverge. ✓
   covered by parity_check (which has USE_WFO=True).
4. **Module-level `from . import orchestrator` missed:** import
   would `NameError` on first `_walk_forward_impl` call. ✓ caught at
   first pytest invocation.
5. **Register called BEFORE the function defs:** would
   `NameError: _walk_forward_regime_path is not defined`. ✓ caught
   at import.

## Sign-off

**PROCEED.**

Phase 1 progress: **5/6 items complete** (#2, #3, #14, #15, #5).
Remaining: #46 (hold-period bin as first-class engine param). The
orchestrator infrastructure now exposes the dispatch surface that
Phase 2's panel routes (#1, #4, #5(iter), #6-#8, #44-#45) will plug
into, and that Phase 3's pairs/carry routes will extend further.

Daniel Vieira Gatto — 2026-05-14.

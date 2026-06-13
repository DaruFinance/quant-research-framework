# Item #45 — verification log

**Goal:** Portfolio-level constraints. Apply a single-asset weight
cap and/or a gross-leverage cap to a proposed weight vector after
sizing (#6) and neutralization (#7) have produced their output.

**Dataset:** synthetic weight vectors + DS-PANEL-3 rebalance audit.

## What landed

**Python (`backtester/panel/constraints.py`, new):**

- `apply_constraints(weights, *, single_asset_max=None,
  gross_lev_max=None, max_iter=100, tol=1e-12)` — dispatcher.
- Single-asset cap iteratively trims `|w_i| > cap` and redistributes
  the excess **only to legs strictly below cap** (avoids the
  oscillation loop where legs-at-cap absorb residual and re-cross
  the cap). When every leg hits cap, the residual is dropped — the
  gross may shrink (mathematically: max representable gross is
  `n_legs * cap`).
- Gross-leverage cap uniformly rescales every weight by
  `gross_lev_max / sum(|w_i|)` when the constraint binds. Preserves
  per-leg composition.
- Order: single-asset cap first (non-linear), then gross cap
  (linear).
- Both caps are **idempotent**: applying twice yields the same
  result as once.
- Input is **never mutated** in place (pure-pointwise contract).

## G1 — Parity surface

| Surface         | Result        |
|-----------------|---------------|
| parity_check    | PARITY OK     |
| parity_regime   | PARITY OK     |
| parity_forex    | PARITY OK     |
| parity_ledger   | LEDGER PARITY OK (1389 trades, 6945 fields) |

Pure new code, no kernel touch.

## G2 — Property tests

`tests/test_panel_constraints.py`: **16 tests pass.**

- No-cap pass-through (input unchanged).
- Single-asset cap caps the over-threshold leg (`|out_i| <= cap`).
- Single-asset cap preserves signs.
- Single-asset cap redistributes excess to strictly-under-cap legs.
- Single-asset cap drops residual when every leg hits cap (gross
  shrinks; mathematically unavoidable).
- Single-asset cap idempotent.
- Single-asset cap rejects `cap <= 0` and `cap > 1`.
- Gross-leverage cap scales when binding, preserves composition.
- Gross-leverage cap idempotent.
- Gross-leverage cap no-op when slack.
- Gross-leverage cap rejects `cap <= 0`.
- Both caps composed (single-asset then gross): both bind.
- Pure: input array never mutated.
- Pointwise locality: caps respect their input scope (slicing input
  preserves cap behaviour on the slice).
- G3 5-rebalance audit (below).

Full Python pytest sweep: **145 passed, 3 skipped, 0 failed**
(was 129 pre-#45; +16).

## G3 — 5-rebalance redistribution audit

`test_g3_5_rebalances_redistribution_audit`: 5 scenarios where SOL
exceeds `cap = 0.30`. For each:

| Scenario        | Input             | Output (cap=0.3)        | Gross in | Gross out | Cap binds | Signs ok |
|-----------------|-------------------|-------------------------|----------|-----------|-----------|----------|
| 1               | [0.50, 0.30, 0.20]| [0.30, 0.30, 0.30]      | 1.00     | 0.90      | yes       | yes      |
| 2               | [0.45, 0.35, 0.20]| [0.30, 0.30, 0.30]      | 1.00     | 0.90      | yes       | yes      |
| 3               | [0.40, 0.40, 0.20]| [0.30, 0.30, 0.30]      | 1.00     | 0.90      | yes       | yes      |
| 4               | [0.55, 0.25, 0.20]| [0.30, 0.30, 0.30]      | 1.00     | 0.90      | yes       | yes      |
| 5               | [0.60, 0.20, 0.20]| [0.30, 0.30, 0.30]      | 1.00     | 0.90      | yes       | yes      |

Common to all 5: gross shrinks from 1.0 to 0.9 because three legs
each at cap=0.3 max out the representable mass. Cap binds on every
leg in every scenario (SOL was overcapped; the others absorb to the
cap). Sign of every position preserved.

## Sign-off

**PROCEED.**

**Phase 2 complete (8/8 items).** The panel plugin is feature-
complete: panel data loader (#1), cross-asset regime detector (#4
HIGH-RISK), basket/cohort orchestrator (#5 iter), ERC sizing (#6),
β-/$-/σ-neutralization (#7), long-short basket primitive (#8),
multi-term IS objective (#44 HIGH-RISK with parity_combo caveat),
portfolio constraints (#45).

Daniel Vieira Gatto — 2026-05-14.

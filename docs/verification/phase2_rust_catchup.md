# Phase 2 Rust catchup — items #6, #7, #8, #44, #45

**Goal:** Port the five panel-pipeline primitives that landed Python-
only in Phase 2 to the Rust framework, and verify Python and Rust
produce matching outputs on DS-PANEL-3.

## What landed (Rust side)

| Item | Rust module                          | Status |
|------|--------------------------------------|--------|
| #6   | `src/panel/sizing.rs`                | ✓ ERC via Spinu cyclical descent (no scipy needed); `equal_weights` + `risk_contributions` helpers; `cov_from_returns` NaN-tolerant cov estimator. |
| #7   | `src/panel/neutralize.rs`            | ✓ `neutralize_dollar` / `neutralize_beta` / `neutralize_sigma`; `estimate_betas` / `estimate_vols` helpers. |
| #8   | `src/panel/strategies/long_short.rs` | ✓ `LongShortBasket` + `momentum_alpha` factory. |
| #44  | `src/metrics.rs` + `src/objectives.rs` | ✓ `sortino` + `turnover` + `MultiTermObjective`. |
| #45  | `src/panel/constraints.rs`           | ✓ `apply_constraints` with single-asset and gross caps. |

Each module has inline `#[cfg(test)] mod tests` covering the same
property assertions as the Python test suite.

## Cross-language parity result

`tools/parity_panel.py` drives both engines on DS-PANEL-3 sources at
`t=500` with `100-bar` returns window for ERC / betas / vols, and
diffs the outputs key-by-key:

| key                              | tol     | max_rel    | status |
|----------------------------------|---------|------------|--------|
| basket_dollar                    | 1e-12   | 0.00e+00   | OK     |
| betas                            | 1e-09   | 0.00e+00   | OK     |
| constraints_cap_03_05_03_02      | 1e-12   | 0.00e+00   | OK     |
| constraints_cap_05_06_02_02      | 1e-12   | 0.00e+00   | OK     |
| constraints_gross_3              | 1e-12   | 0.00e+00   | OK     |
| eq_weights                       | 1e-12   | 0.00e+00   | OK     |
| **erc_weights**                  | 1e-03   | **3.36e-05** | OK     |
| multi_term_score                 | 1e-09   | 0.00e+00   | OK     |
| neutralize_beta                  | 1e-09   | 0.00e+00   | OK     |
| neutralize_dollar                | 1e-12   | 0.00e+00   | OK     |
| neutralize_sigma                 | 1e-09   | 0.00e+00   | OK     |
| sortino_sol                      | 1e-09   | 0.00e+00   | OK     |
| sortino_sol_ann252               | 1e-09   | 0.00e+00   | OK     |
| turnover                         | 1e-12   | 0.00e+00   | OK     |
| vols                             | 1e-09   | 0.00e+00   | OK     |

**Result: PANEL CROSS-LANG PARITY OK** — 14/15 keys agree to floating-
point precision; ERC weights agree to ~3.4e-5 relative (well inside
the 1e-3 budget). The ERC tolerance is necessarily looser because
the two engines use different solvers: Python's scipy SLSQP vs
Rust's Spinu cyclical descent. Both find the same ERC optimum but
converge to slightly different numerical neighbourhoods of it.

## Phase 1 regression sweep (single-threaded)

All four Phase 1 parity surfaces stay green at 1e-3 after the Rust
catchup:

| Surface         | Result        |
|-----------------|---------------|
| parity_check    | PARITY OK     |
| parity_regime   | PARITY OK     |
| parity_forex    | PARITY OK     |
| parity_ledger   | LEDGER PARITY OK (1389 trades, 6945 fields) |

## Rust test sweep (single-threaded)

`cargo test --jobs 1 --release --features panel -- --test-threads=1`
post-catchup:

- `src/lib.rs` panel module: 50 unit tests pass (was 11 pre-#6;
  +30 from the catchup, +9 from earlier items #4 / #5(iter))
- `src/orchestrator.rs`: 4 tests pass
- `tests/invariants.rs`: 14 tests pass
- `tests/test_parity_registry.py` (Python under Rust tests/): 8 pass
- `tests/contract_v0_2.rs`, `tests/behavioural.rs`: unchanged

## T1 is now bilingually upstream-pushable 🚀

Tree-1 (Trend / Momentum) was already Python-complete at the close
of Phase 2. With this Rust catchup, both sides of the framework are
feature-complete and cross-language-verified for T1. Per the user's
workflow rule ("Python first per pipeline, then Rust port, then
cross-language verify, then push"), T1 is now eligible for upstream
push.

Daniel Vieira Gatto — 2026-05-14.

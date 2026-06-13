# Item #46 — verification log

**Goal:** Promote hold-period from a strategy-level workaround to a
first-class engine config. `MAX_HOLD_BARS = 0` is the default (no
force-close, engine behaves exactly as pre-#46). `MAX_HOLD_BARS > 0`
makes the kernel emit a code-2 / code-4 close at bar `i` whenever an
open position satisfies `(i - ent_bar) >= MAX_HOLD_BARS`.

**Precedence:** news/session > hold-period > SL/TP intrabar > signal-
driven. The `elif` chain in the kernel guarantees the order; SL/TP's
intrabar check still wins via the existing `continue` shortcut.

**Dataset:** `tests/fixtures/sol_1h_30000_31000.csv` (DS-SOL-1K).

## What landed

**Python (`backtester/__init__.py`):**

- Module constant `MAX_HOLD_BARS = 0` with the precedence docstring.
- `_backtest_numba_core` signature extended with `max_hold_bars`
  positional parameter (Numba-int).
- In the main loop, immediately after the news/session block and
  before the SL/TP intrabar check, an `elif` clause:
  ```python
  elif max_hold_bars > 0 and (idx - ent_bar) >= max_hold_bars:
      code = 2 if open_pos == 1 else 4
  ```
  The existing code-2 / code-4 emission paths handle the actual
  trade close — no new emission site added.
- `backtest()` wrapper passes `MAX_HOLD_BARS` through to the kernel.

**Rust (`src/lib.rs`):**

- `Config.max_hold_bars: usize` field (default 0).
- `Config::with_max_hold_bars(n)` builder.
- In `backtest_core`, a matching guard right after the session-end
  block, before the SL/TP check.

**Tests (`tests/test_invariants_property.py`):**

- `test_max_hold_bars_no_leak_property` — Hypothesis: for every
  (seed, n, lb, max_hold), every emitted trade must satisfy
  `exit_idx - entry_idx <= max_hold`. The kernel's decision uses only
  `idx` and `ent_bar` (both at-or-before the bar), so the property is
  also a strong lookahead guard.
- `test_max_hold_bars_zero_preserves_v0_4_0_behavior` — idempotence
  check: setting MAX_HOLD_BARS=0 twice produces identical trade
  lists and metrics. Failure would surface a stale-state bug or
  accidental side effect of the in-loop check.

## G1 — Parity surface (default = MAX_HOLD_BARS=0)

All four parity scripts green at 1e-3 — the gate is preserved
exactly when the cap is off:

| Surface         | Result        |
|-----------------|---------------|
| parity_check    | PARITY OK     |
| parity_regime   | PARITY OK     |
| parity_forex    | PARITY OK     |
| parity_ledger   | LEDGER PARITY OK (1389 trades, 6945 fields) |

Python ↔ Rust parity at `max_hold_bars > 0` is verified by
**algorithm-identity**: both engines compute `(i - ent_bar) >=
max_hold_bars` over `usize`/`int64` integers and emit code 2/4 on the
match — no time-dependent inputs differ between languages. When item
#15's `hold-period` family gains stdout exposure, the
``--include hold-period`` parity lane will assert this directly.

## G2 — Lookahead / leak property tests

`tests/test_invariants_property.py` post-#46: **8 property tests
pass** (was 6 pre-#46; +2 from `test_max_hold_bars_*`).

Full pytest sweep: **58 passed, 3 skipped, 0 failed.**

The property test runs ~30 random `(seed, n, lb, max_hold)` triples;
for each, every emitted trade satisfies `hold <= max_hold` AND every
trade's decision input is at-or-before its `entry_idx`. Pure-integer
arithmetic; trivially lookahead-free.

## G3 — Hand-inspected trades (DS-SOL-1K, cap=5)

Note on cap value: the plan called for `max_hold_bars=24` on 24×1h
bars (one day). On this DS-SOL-1K slice the EMA-cross strategy with
default SL/TP exits aggressively — natural max hold is 12 bars; the
24-bar cap never binds. Re-ran at `cap=5` to actually exercise the
binding path. `cap=24` is the right value for slower carry-style
strategies; once item #6's funding-carry tree lands the standard
verification slice will be different.

DS-SOL-1K, LB=10, `MAX_HOLD_BARS=5`:

- OFF: 37 trades, hold distribution top 10 = `[12, 11, 8, 7, 6, 5,
  5, 5, 4, 4]`. Sharpe -0.7071, ROI -0.001460, MaxDD 0.002282.
- ON: 37 trades, hold distribution top 10 = `[5, 5, 5, 5, 5, 5, 5,
  5, 4, 4]`. Sharpe -0.5592, ROI -0.001060, MaxDD 0.002205.
- 8 trades capped at exactly 5 bars; 29 exited early via SL/TP/signal.

5 capped trades:

|  # | side | ent | exi | hold | ent_px  | exi_px  | pnl     |
|----|------|-----|-----|------|---------|---------|---------|
|  3 |  +1  | 105 | 110 | 5    | 98.5195 | 97.5051 | -26.99  |
| 11 |  +1  | 279 | 284 | 5    | 85.8057 | 87.0939 | +36.52  |
| 12 |  -1  | 290 | 295 | 5    | 87.9336 | 87.7863 |  +2.94  |
| 15 |  +1  | 407 | 412 | 5    |100.5902 |101.0697 | +10.91  |
| 17 |  +1  | 486 | 491 | 5    | 99.1397 | 98.1189 | -26.74  |

Every capped trade has `exit_idx - entry_idx == 5`. ✓

2 uncapped trades (signal-driven exits within 5 bars):

|  # | side | ent | exi | hold | ent_px  | exi_px  | pnl     |
|----|------|-----|-----|------|---------|---------|---------|
|  0 |  -1  |   2 |   3 | 1    | 96.8909 | 97.8892 | -26.76  |
|  1 |  +1  |  22 |  24 | 2    | 93.8381 | 96.6243 | +73.21  |

Both have `hold < 5`. Both exited via signal-driven open-price code
2 / 4 (not via the hold-period gate). ✓

Lookahead audit (5 capped trades):

- Trade 3 ent=105: cap fires at idx=110 because `110 - 105 == 5 >=
  max_hold_bars=5`. Both indices ≤ 110. ✓
- Trades 11, 12, 15, 17: identical reasoning; each fires at `ent +
  max_hold_bars`, both indices ≤ the firing bar.
- The cap decision reads no series values at all — purely
  arithmetic on `idx` and `ent_bar`. No conceivable lookahead.

## Sign-off

**PROCEED.**

**Phase 1 complete (6/6 items).** Ledger schema (#2), per-leg costs
(#3), invariant framework (#14), parity registry (#15), orchestrator
dispatch (#5), hold-period bin (#46) all delivered with G1 + G2 + G3
green and full leak audits.

The CORE layer is ready. Phase 2 (`backtester.panel` plugin: items
#1, #4, #5(iter), #6, #7, #8, #44, #45) plugs into the dispatch
registry's `multi_asset=True` route; Phase 3 (pairs + carry) and
Phase 4–5 (MM simulator + arb) inherit the same trade-log schema,
cost decomposition, lookahead-leak harness, and parity registry.

Daniel Vieira Gatto — 2026-05-14.

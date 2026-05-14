# Verification logs

Every numbered item in the multi-month extension plan
(`~/.claude/plans/enchanted-strolling-hopcroft.md`) lands one Markdown
log in this directory: `item02.md`, `item03.md`, …, `item46.md`.

A log is created **only after** all three gates pass. Code merges to the
v2 working branch wait on the log.

## The three gates (G1 / G2 / G3)

Every item passes by satisfying all three before the next item starts.

### G1 — Parity

The cross-language parity surface (Python ↔ Rust) at `1e-3` relative
tolerance stays green:

- `tools/parity_check.py` — default config surface
- `tools/parity_regime.py` — regime + WFO surface
- `tools/parity_forex.py` — forex-mode surface
- `tools/parity_ledger.py` — per-trade ledger row-by-row diff
- `tools/parity_combo.py` — four-way combo (known diff until item #44)

The default expectation is "no metric line moved when the new feature
flag is off." When an item legitimately adds metric lines, the
`MetricRegistry` from item #15 keeps them gated behind `--include` flags
so the default lane stays at the baseline count.

### G2 — Lookahead / leak property test

Every new state-bearing function gets a pollute-and-verify property
test in `tests/test_invariants_property.py`. The pattern:

1. Pick a random index `T`.
2. Pollute the input data at positions `> T`.
3. Call the function with `T` as the decision boundary.
4. Assert the output for indices `≤ T` equals the un-polluted call.

High-risk items (#4, #11, #9, #18, #32, #44) run with 50 random `T`
points; standard items run with 20.

### G3 — Hand-inspected trades

5 trades from the canonical real-data backtest are walked through by
hand. For each: trace the entry signal back to its driving bar, confirm
every index consulted is `≤ entry_idx`, reproduce the entry/exit math.

Trades to inspect: typically trades #1, ⌊N/4⌋, ⌊N/2⌋, ⌊3N/4⌋, #N.

## Log template

Use this template for every `itemNN.md`:

```
## Item #N — verification log

- Dataset: <path>
- Config: <key params copy-pasted from Config repr>
- Trade table (5 trades):
  | # | entry_idx | entry_time | exit_idx | exit_time | side | qty | entry_px | exit_px | fee | slippage | funding | pnl |
  |---|-----------|------------|----------|-----------|------|-----|----------|---------|-----|----------|---------|-----|

- For each trade: trace its origin.
  - Which bar's indicator value crossed the threshold?
  - Which regime label / spread-beta value / funding-rate sign / book-state was used?
  - Show the actual numbers. Confirm every index consulted is <= entry_idx.
- Parity surface result: <x>/<n> green.
- Lookahead property tests: <list> all pass.
- Sign-off: PROCEED / HOLD. <signature, date>.
```

No implementation commit lands on the v2 working branch until the log
file is present and the sign-off line reads PROCEED.

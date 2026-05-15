# Phase 3 — T6 carry items (Python side): #38, #39, #39s, #40, #41, #42, #43

**Goal:** Land the funding / basis / OI / on-chain / scheduler /
signal-model plumbing on the Python side.  Rust port deferred to the
next workstream per the Python-first workflow rule.

**Datasets:** DS-FUNDING-200, DS-BASIS-1D, DS-OI-7D, DS-ONCHAIN-50
(all built and committed earlier in Phase 3).

## What landed

**`backtester/carry/funding.py` — #38:** `FundingEvent`, `load_funding`
(strict 8h boundary check, NaN rejection, dedup), `next_funding_time`,
`rate_at` (point-in-time look-up, `≤ t` only).

**`backtester/carry/basis.py` — #39:** `BasisRecord`, `load_basis` —
recomputes `basis_bp` from `(perp - spot)/spot` and rejects sources
whose pre-computed column drifts > 0.01bp (catches smoothed /
forward-looking sources).  `basis_at` for point-in-time look-up.

**`backtester/carry/triggers.py` — #39s:** `FundingFlipTrigger` fires
on consecutive sign change with `min_magnitude` filter.
`BasisBlowoutTrigger` z-scores `basis_bp` against a trailing window
ending at the candidate row (exclusive) — the row itself does not
contribute to its own threshold.  `TriggerEvent` carries the prev /
curr values + extras (z, sigma).

**`backtester/carry/oi.py` — #40:** `OIRecord`, `load_oi` — verifies
uniform cadence within tolerance to catch mis-read intervals, rejects
NaN.  `oi_at` lookahead-free.

**`backtester/carry/onchain.py` — #41:** snapshot-pinning loader.
`load_onchain` reads from disk, computes `sha256` + `mtime` of the
source file, stores them in `df.attrs["snapshot_sha256"]` etc.  A
revised source file (provider backfill / reorg) produces a different
hash so consumers can detect drift, while the pinned in-memory copy
continues to return the original values.

**`backtester/carry/scheduler.py` — #42 (HIGH-RISK):**
`EventDrivenScheduler` merges bar-cadence / funding-event / trigger
streams into a sorted rebalance schedule with deterministic
co-located-event ordering (`bar < funding < trigger`).
`next_rebalance(after_s)` is the streaming form used by the
orchestrator.

**`backtester/carry/models/` — #43:** `SignalEmission` dataclass plus
three models:

- `PersistentFundingSignModel(min_streak)` — emits carry direction
  after a streak of same-sign rates ≥ min_streak.
- `FundingMomentumModel(window, z_thresh)` — z-score against
  trailing window (exclusive of current row).
- `FundingOICointegrationModel(window, scale)` — joint-move
  detector between funding and OI z-scores.

## G1 — Parity surface

| Surface         | Result        |
|-----------------|---------------|
| parity_check    | PARITY OK     |
| parity_regime   | PARITY OK     |
| parity_forex    | PARITY OK     |
| parity_ledger   | LEDGER PARITY OK (1389 trades, 6945 fields) |

Pure additions; Phase 1 / Phase 2 / Phase 3 pairs surfaces untouched.

## G2 — Property tests

**Test files (39 new test cases, all pass):**

- `tests/test_carry_funding.py` (8): boundary check, NaN rejection,
  `rate_at` 20-cut pollute test, `next_funding_time` arithmetic.
- `tests/test_carry_basis.py` (5): recompute-drift detection (5bp
  injection → loader raises), `basis_at` 10-cut pollute test.
- `tests/test_carry_triggers.py` (6): flip emits on sign change,
  flip 10-cut pollute test, blowout z-window arithmetic, blowout
  spike-pollution test.
- `tests/test_carry_oi.py` (4): cadence-check arithmetic (rejects
  1800s rows when 3600s ± 60s expected), `oi_at` 10-cut pollute
  test.
- `tests/test_carry_onchain.py` (5): snapshot pinning — write a
  modified copy, reload it, assert different `snapshot_sha256` and
  that the pinned DataFrame's value at the revised row is unchanged.
- `tests/test_carry_scheduler.py` (4): bar-only schedule arithmetic,
  co-located-event order (bar < funding < trigger), **HIGH-RISK
  `next_rebalance` 20-T pollute test** (pollute funding times and
  trigger times past T, assert `next_rebalance(T)` unchanged).
- `tests/test_carry_models.py` (7): persistent-sign streak math,
  momentum z-spike, both with 10-cut pollute tests; OI-cointegration
  emits valid direction on the real fixtures.

**Full Python pytest sweep:** **220 passed, 3 skipped, 0 failed**
(was 181 at end of T2 pairs; +39 from carry).

## G3 — Hand-inspected sample

End-to-end on DS-FUNDING-200 + DS-OI-7D + DS-BASIS-1D:

1. `funding = load_funding(DS-FUNDING-200)` → 200 rows, all on 8h
   boundaries, mean rate ≈ +0.5e-5 per 8h.
2. `flips = FundingFlipTrigger().run(funding)` → ≥ 3 events; each
   `direction == sign(curr)` and `sign(curr) != sign(prev)`.
3. `oi = load_oi(DS-OI-7D)` → 168 rows, cadence 3600±60s, no NaN.
4. `basis = load_basis(DS-BASIS-1D)` → 24 rows, `basis_bp` matches
   fresh `(perp-spot)/spot × 1e4` to floating-point identity.
5. `model = PersistentFundingSignModel(min_streak=3)`;
   `sig = model.signal_at(funding, t_last)` → `direction ∈ {-1, 0, +1}`
   with `inputs["streak"]` matching a hand-count of trailing same-sign
   events.
6. `EventDrivenScheduler(bar_cadence_s=24*3600, funding_df=funding,
   triggers=flips, ...).run()` → 200 funding events + bar events +
   `len(flips)` trigger events, sorted by `(time, kind_rank)`.

All `t_s` lookups in (5) and (6) trace back to data at row indices
≤ t_s only; pollution past t_s does not change the emission.

## Sign-off

**PROCEED.** All seven T6-carry Python items land with G1+G2+G3 green.
HIGH-RISK scheduler 20-T pollute battery passes.  Snapshot pinning
catches a 1-row revision in the on-chain feed.

**Next:** Rust port for the entire Phase 3 surface (pairs + carry)
— `src/pairs/` and `src/carry/` mirrors, cargo features `pairs` and
`carry`.  Then `tools/parity_pairs.py` + `tools/parity_carry.py` for
cross-language verification.  Only then push T2 + T6 to upstream.

Daniel Vieira Gatto — 2026-05-14.

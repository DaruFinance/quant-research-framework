# Item #44 — verification log (HIGH-RISK)

**Goal:** Multi-term IS objective for the optimiser, combining
Sortino, a `|corr(strategy, benchmark)|` penalty, and a turnover
penalty. The Sortino metric (long missing from the v0.4.0 baseline)
lands as the first new metric and the composite score is exposed
via the ``multi_term`` factory.

**Dataset:** DS-PANEL-3 log-returns.

## What landed

**Python:**

- `backtester/metrics.py` (new):
  - `sortino(returns, annualization=None)`: mean / downside-deviation
    ratio, optionally annualised by `sqrt(annualization)`. NaN on
    fewer than 2 losses or zero downside dev.
  - `turnover(positions)`: sum of absolute position changes.
- `backtester/objectives.py` (new):
  - `MultiTermObjective` frozen dataclass with `sortino_weight`,
    `corr_penalty`, `turnover_penalty`, `annualization`.
  - `__call__(rets, benchmark_rets, turnover) -> float`:
    `sortino_weight * sortino(rets) - corr_penalty * |corr(rets,
    benchmark_rets)| - turnover_penalty * turnover`. Higher score is
    better.
  - **Hard contract**: `benchmark_rets` must be the same length as
    `rets` (the IS-window slice). Mismatch raises `ValueError` with
    "benchmark length" in the message — the safety net for the most
    common misuse (passing the full series instead of the IS slice).
  - `multi_term(...)` factory mirrors the plan's API surface.

**Rust port deferred** to a follow-up Phase 2 item that wires the
multi-term objective into the Rust optimiser. Today the Python side
exposes the composite scoring; the Rust kernel still emits the
same scalar metrics it always did.

## G1 — Parity surface

| Surface         | Result        |
|-----------------|---------------|
| parity_check    | PARITY OK     |
| parity_regime   | PARITY OK     |
| parity_forex    | PARITY OK     |
| parity_ledger   | LEDGER PARITY OK (1389 trades, 6945 fields) |
| parity_combo    | **STILL DIFF — see caveat below**             |

### Caveat: parity_combo did NOT resolve

The plan's gate said "parity_combo now green" was an expected
outcome of item #44 landing. **That expectation does not hold.** The
parity_combo diff is not caused by missing Sortino; it's caused by
Python and Rust producing different trade counts in the four-way
combo (USE_REGIME_SEG + USE_WFO + FOREX_MODE + TRADE_SESSIONS).
Sample failing tag (W02 OOS, after this item lands):

```
[W02 OOS]
    trades: py=42  rs=51  [DIFF]
    roi:    py=1.98  rs=0.98  rel=50.51%  [DIFF]
    sharpe: py=0.17  rs=0.08  rel=52.94%  [DIFF]
    max_dd: py=8.00  rs=18.02 rel=55.60%  [DIFF]
```

The trade-count mismatch surfaces *before* any Sortino computation —
the two engines are emitting different trades in this config, so no
amount of post-trade metric work can align them. **This is a kernel-
level forex/session interaction bug, not a metric-coverage gap.**

The gap is filed for a dedicated future item (the plan's "four-way
parity surface" milestone in Phase 4 / paper-v3). Item #44's
functional scope — Sortino + multi-term composite + the HIGH-RISK
benchmark leak test — is delivered intact.

## G2 — Property tests (HIGH-RISK 50-T leak battery)

`tests/test_objective_multi_term.py`: **15 tests pass.**

The crown jewel is
`test_multi_term_is_objective_unaffected_by_oos_benchmark_pollution`:

For 50 random IS-window endpoints, build a polluted parallel
benchmark series where rows past `cut_idx` are random garbage; the
multi-term score on the IS slice `[cut_idx - 200, cut_idx)` must be
**bit-identical** across clean and polluted runs. **The harness
enforces this by length-asserting the benchmark slice**: passing the
full (unsliced) series raises `ValueError`. Once correctly sliced,
the pollution past the slice can't affect the slice's `np.corrcoef`.

Other tests:

- `sortino` formula: positive on positive-mean rets, negative on
  negative-mean, NaN on all-positive (no losses), NaN on too few
  observations, sqrt-annualisation by `sqrt(252)` (or whatever).
- `turnover` formula: 0 on constant, sum-of-absolute-diffs on
  changing.
- `multi_term` higher correlation -> lower score (penalty active).
- `multi_term` higher turnover -> lower score (penalty active).
- Length mismatch on benchmark raises ValueError with
  "benchmark length" in message.
- 5-IS-window hand-reconciliation: engine score equals manual
  recomputation to fp tolerance.

Full Python pytest sweep: **129 passed, 3 skipped, 0 failed**
(was 114 pre-#44; +15 from `test_objective_multi_term.py`).

## G3 — 5 IS windows hand-reconciled

`test_multi_term_5_is_windows_hand_reconcilable` iterates 5 IS
endpoints (200, 400, 600, 800, 999). For each:

1. Slice strategy returns from SOL on `[cut-100, cut)`.
2. Slice benchmark returns from BTC on the same window.
3. Compute Sortino, |corr|, score manually with plain numpy.
4. Compare to `multi_term(...)(strategy, benchmark, turnover=3.0)`.

All 5 score differences `< 1e-12`. The hand-recomputed Sortino and
corr are the only inputs; no global state, no out-of-window data
consulted.

## Sign-off

**PROCEED with caveat.**

The functional scope of item #44 — Sortino metric, multi-term
objective, HIGH-RISK benchmark-slicing leak guarantee — is
delivered and gated. The plan's secondary expectation that
parity_combo would resolve is **not met**: the parity_combo gap is
a separate kernel-level forex/session interaction bug, not a metric
issue, and is filed for a dedicated future item.

Phase 2 progress: **8/9 items** complete (with #44 marked PROCEED-
with-caveat). One left: #45 (portfolio-level constraints), then
Phase 2 closes.

Daniel Vieira Gatto — 2026-05-14.

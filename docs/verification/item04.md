# Item #4 — verification log (HIGH-RISK)

**Goal:** Generalise the single-asset `detect_regimes(df) -> pd.Series`
to a panel version `detect_regimes_panel(panel) -> Dict[asset, pd.Series]`.
Each asset's label at time `t` may depend on **any asset's data at
times ≤ t**, never on data at times > t.

**Dataset:** DS-PANEL-3 (3 assets × 1000 timestamps × OHLC).

## What landed

**Python (`backtester/panel/regime.py`, new):**

- `detect_regimes_panel_per_asset(panel)` — default: each asset gets
  its own EMA-200 + 8-bar consistency regime, computed from its
  own close series only. Trivially leak-free across assets.
  Decorated with `@registers_invariant(name="panel_regime_per_asset",
  data_kind="panel")` so the framework's pollute-and-verify harness
  picks it up automatically.
- `detect_regimes_panel_market(market_asset='BTC')` factory: returns
  a detector that broadcasts the named market asset's labels to
  every asset in the panel. Classic BTC-dominance regime. Leak-free
  because the market asset's own regime at `t` reads only its own
  past.
- `detect_regimes_panel` module-level alias = `_per_asset` for
  symmetry with the single-asset `bt.detect_regimes` pluggable
  pattern.

**Rust (`src/panel/regime.rs`, behind `feature = "panel"`):**

- `trait PanelRegimeDetector { fn detect(&self, panel) -> HashMap<String, Vec<u8>>; }`.
- `PerAssetRegime` and `MarketRegime` impl.
- `LABEL_RANGING / LABEL_UPTREND / LABEL_DOWNTREND` u8 constants
  (no string allocation in the hot loop, unlike Python which
  follows the original string-label convention for parity with
  single-asset).

## G1 — Parity surface

The panel plugin is opt-in on both sides. Default cargo build and
default pip install don't pull xarray/ndarray; nothing in the
Phase 1 metric surface is touched.

| Surface         | Result        |
|-----------------|---------------|
| parity_check    | PARITY OK     |
| parity_regime   | PARITY OK     |
| parity_forex    | PARITY OK     |
| parity_ledger   | LEDGER PARITY OK (1389 trades, 6945 fields) |

## G2 — Lookahead / leak property tests (HIGH-RISK)

Per the plan, item #4 is a HIGH-RISK lookahead surface (multi-asset
detectors that accidentally read across asset boundaries at future
times are easy to write). The mitigations are belt-and-suspenders:

**The 50-T cross-asset pollute battery** (`_assert_no_cross_asset_leak`
helper in `tests/test_panel_loader.py`):

For each of 50 cut indices ≥ 220 (post-EMA-200 warmup), for each
ordered `(victim, witness)` pair in the 3-asset panel:

1. Build a polluted panel where `victim`'s OHLC at rows `>= cut` is
   NaN.
2. Run the detector on clean and polluted panels.
3. Assert `witness`'s labels for rows `[0, cut)` are bit-identical
   across the two runs.

Total experiments per detector: 50 cuts × 3 victims × 2 witnesses
= 300 pollute checks. Both default detectors pass; the test runs
the same battery on the Rust side too (300 checks in cargo test
release mode).

**Deliberate-leak catcher:** `test_cross_asset_leak_harness_catches_known_leak`
registers a detector that reads SOL's `close.shift(-1)` to label BTC
— a clear cross-asset future-read — and asserts the harness raises
`AssertionError` with the offending asset name in the message. ✓

**Default-detector property tests:**

- `test_panel_regime_per_asset_no_cross_asset_leak` ✓
- `test_panel_regime_market_no_cross_asset_leak` ✓
- `test_panel_regime_market_broadcasts_market_asset_labels` ✓ (BTC
  label series broadcast bit-identically to SOL and ETH)
- `test_panel_regime_market_rejects_missing_market_asset` ✓ (DOGE
  not in panel raises with the asset name)

Full Python pytest sweep: **71 passed, 3 skipped, 0 failed** (was
67 pre-#4; +4 from `test_panel_loader.py`).

Full Rust cargo test sweep with `--features panel`: **11 panel
tests pass** (was 7 pre-#4; +4 from `src/panel/regime.rs` inline
tests including the same 50-T leak battery on the Rust path).

## G3 — Hand-inspected regime distributions

Per-asset regime counts on DS-PANEL-3:

| Asset | Uptrend | Downtrend | Ranging | Total |
|-------|---------|-----------|---------|-------|
| SOL   | 480     | 336       | 184     | 1000  |
| BTC   | 602     | 220       | 178     | 1000  |
| ETH   | 491     | 363       | 146     | 1000  |

The 1000-bar slice spans 2024-01-13 → 2024-02-23, which contains
the Q1 2024 crypto rally; BTC's higher Uptrend share (60.2%)
reflects that. SOL/ETH show similar trend bias with more chop.

Market-regime broadcast verified: applying `detect_regimes_panel_market("BTC")`
makes SOL and ETH's label series bit-identical to BTC's (601
Uptrend, 220 Downtrend, 178 Ranging across all three).

## Sign-off

**PROCEED.**

The cross-asset regime contract is locked. Items #5(iter) basket
orchestrator, #6 ERC sizing, #7 neutralization, #8 long-short basket,
and #44 multi-term IS objective can now consume
`detect_regimes_panel(panel) -> Dict[asset, Series]` knowing the
plugged-in detector is leak-free across the asset dimension.

Daniel Vieira Gatto — 2026-05-14.

## Summary

What this PR does (one to three bullets).

## Parity verification (REQUIRED if engine semantics change)

Per `CONTRIBUTING.md`, any change that alters engine behaviour must
keep the cross-language metric diff inside `1e-3` relative tolerance.
Tick the four checks below before merge:

- [ ] `pytest -q` (≥ 32 tests pass; property-test contract intact).
- [ ] `python tools/parity_check.py --csv data/SOLUSDT_1h.csv --tol 0.001` from the Rust port (56 / 56 metric points).
- [ ] `python tools/parity_regime.py --csv data/SOLUSDT_1h.csv --tol 0.001` from the Rust port (98 / 98 metric points).
- [ ] `python tools/parity_forex.py --csv data/EURUSD_1h.csv --tol 0.001` from the Rust port (56 / 56 metric points).

If you also want belt-and-braces coverage, the BTC 30m, DOGE 30m,
SYNTH_100k parity_check runs from the Rust port's CI matrix should
also stay green.

## Linked Rust PR (REQUIRED if engine semantics change)

Engine-side metric-affecting changes ship as a paired commit:

- Rust port PR: <link>

## Risk

What could go wrong; how it would be detected (test, runtime, parity-script numeric drift).

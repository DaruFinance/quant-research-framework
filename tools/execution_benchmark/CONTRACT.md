# Matched signal-driven execution workload

This directory is a separate execution-kernel comparison. It is not the QRF
walk-forward benchmark and its timings must not be divided into the existing
full-WFO timings.

- Input: the real 150,000-bar BTCUSDT 30-minute file used by the QRF audit.
- Strategies: the same five QRF signal families and two fixed lookbacks per
  family. `write_events.py` freezes one deterministic event CSV before any
  engine process. Every engine consumes that exact file.
  Existing strategy names are retained for traceability, but their embedded TP
  labels are inactive here because take-profit is disabled.
- Causality: each raw signal is shifted one bar. An event at index `i` executes
  at `open[i]` and depends only on data through `i-1`. This is long-only spot:
  positive enters long, negative closes long, and zero holds.
- Boundary: no first-bar trade. Any final-bar flip is suppressed; an existing
  position is explicitly closed at the final bar's open. This gives all engines
  a representable finalization rule.
- Positioning: a single long position. Competitors use one BTC unit.
  QRF's fixed-notional ledger must be normalized by its recorded quantity for
  per-unit comparison.
- Execution charge: 7 bp per fill, implemented as commission with zero price
  offset. This combines the 5 bp taker fee and a 2 bp slippage proxy. It is not
  a claim that price slippage was modeled.
- Disabled: WFO, parameter optimization, SL, TP, funding and robustness runs.
- Timed region: only the ten serial engine calls, including an engine's
  per-run constructor when it has one. OHLC and frozen-event CSV
  loading and normalized ledger writing are outside internal engine time but
  remain inside externally measured process wall/user/system/RSS. Signal
  generation itself is outside every measured engine process.
- Required validation: event hashes and trade count, side, entry/exit indices,
  entry/exit prices, per-unit execution charge, and per-unit net PnL. Report
  tolerances; do not substitute headline metric agreement for ledger agreement.

Backtesting.py has no public switch to suppress its progress iterator. Its
adapter replaces only the module-local `_tqdm` wrapper with the identity
iterator so terminal rendering is not part of engine time; trading code is
unchanged.

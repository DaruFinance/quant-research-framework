# v0.4.0 baselines

Frozen reference outputs captured at the v0.4.0 head (Phase 0 of the
extension plan). Every Phase 1+ change must reproduce these when the
new feature flag is off. If anything below moves, the change is not
backwards-compatible and the corresponding `docs/verification/itemNN.md`
gate cannot pass.

## Parity surface

| File | Surface | Result |
|------|---------|--------|
| `v0.4.0_parity_check_stdout.txt`  | default config (IS/OOS baseline, smart-opt, WFO, 4 robustness overlays) | PARITY OK at 1e-3 |
| `v0.4.0_parity_regime_stdout.txt` | `USE_REGIME_SEG=True` + `USE_WFO=True` | PARITY OK at 1e-3 |
| `v0.4.0_parity_forex_stdout.txt`  | `FOREX_MODE=True` on EURUSD 1h | PARITY OK at 1e-3 |

All three were run via:

```
QRF_PY_DIR=/home/daru/quant-research-framework-v2 \
  python3 tools/<script>.py --tol 0.001
```

from the Rust v2 repo. Each script regex-extracts metric lines from
both engines' stdout and diffs the 8 representative tags
(`IS-raw`, `OOS-raw`, `IS-opt`, `OOS-opt`, `Baseline IS`, `Baseline OOS`,
`W01 IS`, `W01 OOS`) × 7 fields. Max observed relative deviation: 0.00%.

## DS-SOL-1K

| File | Description |
|------|-------------|
| `v0.4.0_ds_sol_1k.json`     | Parsed metrics-by-tag from running the bundled default strategy on the 1000-bar SOL slice |
| `v0.4.0_ds_sol_1k_raw.txt`  | Full stdout from that run |

The bundled defaults (`BACKTEST_CANDLES=10000`, `OOS_CANDLES=90000`,
`WFO_TRIGGER_VAL=5000`) do not fit a 1000-bar dataset, so the baseline
uses the smallest configuration that produces meaningful WFO activity:

```
BACKTEST_CANDLES = 300
OOS_CANDLES      = 600
ORIGINAL_OOS     = 600
WFO_TRIGGER_VAL  = 150
DEFAULT_LB       = 20
USE_MONTE_CARLO  = False
```

The same config is the verification target for item #2 (multi-leg
trade ledger schema) and item #3 (per-leg costs). Multi-leg flag off →
parsed metrics must match this JSON file to floating-point identity.

The bundled default strategy is the EMA(20) vs EMA(lb) crossover
(`bt.create_raw_signals`), not an SMA crossover as the original plan
called it.

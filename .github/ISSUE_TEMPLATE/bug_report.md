---
name: Bug report
about: Report a defect in the engine, parity surface, or test contract.
title: "[bug] "
labels: bug
assignees: ''
---

### Summary

A clear, one-sentence description of the bug.

### Reproduction

1. Dataset / CSV used (path or download command).
2. Module flags / env vars (`USE_OOS2`, `FOREX_MODE`, `BT_CSV`, etc.).
3. Exact command run (`python -m backtester`, a `pytest -k …` selector, or a parity script).
4. Expected output.
5. Observed output (paste a stdout snippet or attach `trade_list.csv`).

### Environment

- Framework version (`pyproject.toml`):
- Python version:
- OS:
- numpy / pandas / numba versions (paste `pip freeze | grep -E "numpy|pandas|numba|scipy"`):

### Parity impact (if applicable)

- Does any of the four parity scripts (`parity_check.py`, `parity_regime.py`, `parity_forex.py`, `parity_ledger.py` from the Rust port) flag a mismatch after this bug? If yes, attach the script's stdout.

### Additional context

Anything else the reviewer needs (logs, screenshots, equity-curve plots).

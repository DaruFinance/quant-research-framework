# Convenience targets for the Python reference.
#
#   make repro       — the <5-minute reproduction: the no-look-ahead property
#                      suite (Hypothesis) + the look-ahead leak demo.
#   make leak        — just the look-ahead leak demo
#   make invariants  — just the Hypothesis no-look-ahead property tests
#   make test        — the full pytest suite
#
# CROSS-LANGUAGE PARITY (Python vs Rust) is driven from the Rust port repo,
# which holds the parity harnesses:
#   cd ../quant-research-framework-rs && make parity
export OPENBLAS_NUM_THREADS = 1   # single-threaded: deterministic + polite on shared boxes

PY := python3

.PHONY: repro leak invariants test

leak:
	$(PY) tools/leak_demo.py

invariants:
	$(PY) -m pytest tests/test_invariants_property.py tests/test_invariants.py -q

test:
	$(PY) -m pytest -q

repro: invariants leak
	@echo
	@echo "repro OK — the no-look-ahead property holds over the generated input"
	@echo "space, and the leak demo caught the planted forward-peek bug."
	@echo "For cross-engine parity vs the Rust port: cd ../quant-research-framework-rs && make parity"

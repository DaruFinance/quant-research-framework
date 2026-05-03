---
name: Feature request
about: Suggest a new feature or change in semantics.
title: "[feature] "
labels: enhancement
assignees: ''
---

### Problem

What gap or limitation are you trying to address?

### Proposed solution

A concrete API or behaviour change. Reference the relevant module (`backtester/__init__.py` line ranges) where useful.

### Parity considerations

Does the proposed change alter any IS/OOS/baseline/optimised/WFO metric? If yes:
- Which parity script(s) will need to change in lockstep on the Rust port (`DaruFinance/quant-research-framework-rs`)?
- Are you also able to land the Rust mirror, or is this a Python-only proposal?

(If the change is engine-side and metric-affecting, the parity invariant in CONTRIBUTING.md applies: a green PR must include a paired Rust commit and an updated parity-script tolerance band.)

### Alternatives considered

Other ways to solve the same problem, and why this proposal is better.

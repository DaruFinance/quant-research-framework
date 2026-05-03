# Sphinx docstring cleanup TODO

The Sphinx build (`docs.yml`) currently runs without `-W` because six
docstrings under `backtester/` use pandas/NumPy-style indentation that
docutils flags as "Unexpected indentation" or "Block quote ends without
a blank line". Output is still rendered, just noisily.

Convert each of these to clean RST (use `Parameters`/`Returns`
napoleon-style sections with a blank line after the section header and
no leading two-space indentation on body paragraphs), then drop the
`-W` opt-out from `.github/workflows/docs.yml` and re-enable
warnings-as-errors.

Tracked warnings (run `sphinx-build -b html . _build/html` from
`docs/` to reproduce):

1. `backtester/__init__.py` :: `evaluate_filters` — line 4 of
   docstring: `ERROR: Unexpected indentation. [docutils]`.
2. `backtester/__init__.py` :: `optimize_regimes_sequential` — line 6:
   `ERROR: Unexpected indentation.`; line 7: `WARNING: Block quote
   ends without a blank line; unexpected unindent.`.
3. `backtester/dsr.py` :: module docstring — line 23: `ERROR:
   Unexpected indentation.`; line 24: `WARNING: Block quote ends
   without a blank line; unexpected unindent.`; line 31: `WARNING:
   Definition list ends without a blank line; unexpected unindent.`.

None of these affect the rendered HTML enough to misrepresent the API,
which is why this is a `docs/TODO.md` cleanup item rather than a
release-blocker.

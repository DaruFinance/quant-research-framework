# Releasing

This document describes the release process for the Python framework.
The Rust port has its own `RELEASING.md`; the two coordinate via
matching version-suffix tags (e.g. Python `v0.3.1` ↔ Rust `v0.3.3`).

## One-time PyPI trusted-publishing setup

1. On `pypi.org`, go to *Account → Publishing* and add a new pending
   publisher: project name `quant-research-framework`, owner
   `DaruFinance`, repo `quant-research-framework`, workflow
   `publish-pypi.yml`, environment `pypi`.
2. In this repo on GitHub: *Settings → Environments → New environment*
   named `pypi`, no secrets needed (trusted publishing replaces tokens).
3. The `.github/workflows/publish-pypi.yml` workflow will run on every
   `v*` tag push and publish to PyPI without any token in repo secrets.

## Cutting a release

```bash
# 1. Bump the version in pyproject.toml and CITATION.cff.
$EDITOR pyproject.toml CITATION.cff

# 2. Add a CHANGELOG.md entry with the date and a summary of changes
#    (Keep-A-Changelog format: Added / Changed / Fixed / Removed / Deprecated).
$EDITOR CHANGELOG.md

# 3. Commit and tag.
git add pyproject.toml CITATION.cff CHANGELOG.md
git commit -m "release: vX.Y.Z"
git tag vX.Y.Z
git push origin main vX.Y.Z

# 4. The publish workflow will trigger automatically and publish to PyPI
#    via trusted publishing. Verify on https://pypi.org/project/quant-research-framework/
```

## Coordinating with the Rust port

If the release changes engine semantics (anything that affects the
metric output of the cross-language parity surfaces), the Rust port
must land a matching change in the same release window. Coordinate
the two tags so they agree on a `paper-vN` retag.

| Python tag | Rust tag | Notes |
|---|---|---|
| `v0.3.0`   | `v0.3.2` | paper-v2 |
| `v0.3.1`   | `v0.3.3` | paper-v2 polish |

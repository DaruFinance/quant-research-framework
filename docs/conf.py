"""Sphinx configuration for the quant-research-framework docs."""
import os
import sys
from pathlib import Path

# Add the package to the path so autodoc can import it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# So `import backtester` doesn't blow up on the missing CSV check at
# docs-build time.
os.environ.setdefault("BT_CSV", "data_SOLUSDT_1h.csv")

project = "quant-research-framework"
author = "Daniel Vieira Gatto"
copyright = "2026, Daniel Vieira Gatto"
release = "0.4.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "quant-research-framework"

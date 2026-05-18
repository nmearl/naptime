"""Sphinx configuration for the NAPTIME documentation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# ── Project metadata ──────────────────────────────────────────────────────────
project = "NAPTIME"
author = "Nicholas Earl"
copyright = "2025, Nicholas Earl"
release = "0.1.0"

# ── Extensions ────────────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

# ── Autodoc ───────────────────────────────────────────────────────────────────
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# ── Napoleon (NumPy/Google docstrings) ────────────────────────────────────────
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

# ── Intersphinx ───────────────────────────────────────────────────────────────
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# ── Theme ─────────────────────────────────────────────────────────────────────
html_theme = "furo"
html_title = "NAPTIME"

# ── Source ────────────────────────────────────────────────────────────────────
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

"""Sphinx configuration for the AtomBench documentation."""
from __future__ import annotations

import os
import sys

# Make the `atombench` package importable for autodoc without installing it.
sys.path.insert(0, os.path.abspath(".."))

# ── Project information ───────────────────────────────────────────────────────
project = "AtomBench"
author = "AtomBench developers"
copyright = ""  # not copyrighted — hides the footer copyright line


def _read_version() -> str:
    try:
        import tomllib  # Python 3.11+

        with open(os.path.join(os.path.dirname(__file__), "..", "pyproject.toml"), "rb") as fh:
            return tomllib.load(fh)["project"]["version"]
    except Exception:
        return "0.1.0"


release = _read_version()
version = release

# ── General configuration ────────────────────────────────────────────────────
extensions = [
    "myst_parser",            # write docs in Markdown
    "sphinx.ext.autodoc",     # pull docstrings from the package
    "sphinx.ext.napoleon",    # Google/NumPy-style docstrings
    "sphinx.ext.viewcode",    # links to highlighted source
    "sphinx.ext.intersphinx", # cross-link to Python docs
    "sphinx_copybutton",      # copy button on code blocks
    "sphinx_design",          # cards, grids, buttons on the landing page
]

# autodoc imports the package; mock heavy/native third-party deps so the docs
# build needs only Sphinx + MyST + furo (no pymatgen/amd/jarvis install in CI).
autodoc_mock_imports = [
    "amd", "pymatgen", "jarvis", "numpy", "pandas", "scipy",
    "sklearn", "matplotlib", "click", "requests",
]
autodoc_typehints = "description"
autodoc_member_order = "bysource"

napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "requirements.txt"]

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# Strip interactive prompts so "Copy" yields runnable text.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False

# ── HTML output ──────────────────────────────────────────────────────────────
html_theme = "furo"
html_title = "AtomBench"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"
html_show_sphinx = False

# Syntax-highlighting themes (furo reads pygments_dark_style for dark mode).
pygments_style = "friendly"
pygments_dark_style = "material"

_GITHUB = "https://github.com/atomgptlab/atombench"

html_theme_options = {
    "sidebar_hide_name": True,  # the logo already shows the name
    "top_of_page_buttons": ["view"],
    "source_repository": f"{_GITHUB}/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#e23744",
        "color-brand-content": "#cf2e3a",
        "color-brand-visited": "#cf2e3a",
        "font-stack": (
            "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, "
            "Arial, sans-serif"
        ),
        "font-stack--monospace": (
            "'SFMono-Regular', Menlo, Consolas, 'Liberation Mono', monospace"
        ),
    },
    "dark_css_variables": {
        "color-brand-primary": "#ff8a8a",
        "color-brand-content": "#ffa0a0",
        "color-brand-visited": "#ffa0a0",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": _GITHUB,
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 '
                "3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 "
                "0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-"
                ".82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 "
                "1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 "
                "0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 "
                "2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 "
                "2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 "
                "3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 "
                '2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z">'
                "</path></svg>"
            ),
            "class": "",
        },
    ],
}

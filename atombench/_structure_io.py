"""
atombench._structure_io — format-agnostic crystal structure parsing.

Benchmark CSV cells (target/prediction columns) may contain crystal structures
in any of the following representations — the format is auto-detected:

  poscar-escaped  POSCAR with literal \\n escape sequences  (current default)
  poscar          POSCAR file text with real newlines
  cif             Crystallographic Information File
  xsf             XCrysDen Structure Format (CRYSTAL / PRIMVEC blocks)
  yaml            pymatgen Structure serialised as YAML
  json            pymatgen Structure serialised as JSON
  filepath        Path to a structure file in any of the above formats

Public API
----------
KNOWN_FORMATS       frozenset of recognised format strings
detect_format(s)    return the format key for a cell string
parse_structure(s)  parse a cell string → pymatgen Structure (auto-detect)
assert_valid_cell(s, *, context="")
                    assert that a cell string is valid; raises AssertionError
                    with a descriptive message if not
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from pymatgen.core import Structure

# ── Supported formats ─────────────────────────────────────────────────────────
KNOWN_FORMATS: frozenset[str] = frozenset({
    "poscar",
    "poscar-escaped",
    "cif",
    "xsf",
    "yaml",
    "json",
    "filepath",
})

# Regex: CIF data keywords that appear inside the file body (not just the header)
_CIF_BODY_RE = re.compile(
    r"_cell_length_a|_cell_angle_alpha|_atom_site_|loop_\s*\n\s*_",
    re.IGNORECASE,
)

# XSF block keywords (first non-blank line only)
_XSF_KEYWORDS = frozenset({
    "CRYSTAL", "ATOMS", "ANIMSTEPS", "PRIMVEC", "PRIMCOORD",
    "CONVVEC", "CONVCOORD", "FORCES",
})


# ── Format detection ──────────────────────────────────────────────────────────
def detect_format(cell_str: str) -> str:
    """
    Auto-detect the crystal structure format of *cell_str*.

    Returns one of the strings in KNOWN_FORMATS.

    Detection order (first match wins):
      1. filepath        — no newlines, string is a path to an existing file
      2. json            — starts with '{'
      3. cif             — starts with 'data_' or contains CIF data keywords
      4. xsf             — first non-blank line is a recognised XSF keyword
      5. yaml            — starts with '@class: Structure'
      6. poscar-escaped  — contains literal \\n escape sequences
      7. poscar          — default (POSCAR with real newlines)
    """
    assert cell_str and cell_str.strip(), \
        "Cell string must not be empty or whitespace-only."

    stripped = cell_str.strip()

    # 1. File path — no newlines of any kind, and the path resolves to a file
    if "\n" not in stripped and "\\n" not in stripped:
        try:
            if Path(stripped).is_file():
                return "filepath"
        except (OSError, ValueError):
            pass

    # 2. JSON
    if stripped.startswith("{"):
        return "json"

    # 3. CIF — 'data_' header or CIF body keywords
    if stripped.lower().startswith("data_") or _CIF_BODY_RE.search(stripped):
        return "cif"

    # For line-based checks, work on unescaped content so we see real lines
    if "\\n" in cell_str and "\n" not in cell_str:
        unescaped = cell_str.replace("\\n", "\n").replace("\\t", "\t").strip()
        is_escaped = True
    else:
        unescaped = stripped
        is_escaped = False

    non_blank_lines = [ln for ln in unescaped.splitlines() if ln.strip()]
    first_line = non_blank_lines[0].strip() if non_blank_lines else ""

    # 4. XSF — first non-blank line is an XSF block keyword (case-insensitive)
    if first_line.upper() in _XSF_KEYWORDS:
        return "xsf"

    # 5. YAML — pymatgen YAML Structure starts with '@class: Structure'
    if first_line.startswith("@class: Structure"):
        return "yaml"

    # 6. Escaped POSCAR
    if is_escaped or "\\n" in cell_str:
        return "poscar-escaped"

    # 7. Default: real-newline POSCAR
    return "poscar"


# ── Parsing ───────────────────────────────────────────────────────────────────
def parse_structure(cell_str: str) -> Structure:
    """
    Parse *cell_str* into a pymatgen Structure, auto-detecting the format.

    Asserts
    -------
    - *cell_str* is non-empty.
    - The detected format is one of KNOWN_FORMATS.
    - The resulting Structure has at least one site.

    Raises AssertionError or any pymatgen/stdlib exception on failure.
    """
    assert cell_str and cell_str.strip(), \
        "Cell string must be a non-empty structure representation."

    fmt = detect_format(cell_str)

    # detect_format's return value is always in KNOWN_FORMATS by construction,
    # but assert here to make the invariant explicit and catch future regressions.
    assert fmt in KNOWN_FORMATS, \
        f"Internal error: detect_format returned unrecognised format {fmt!r}."

    s = _parse_with_fmt(cell_str, fmt)

    assert s is not None and len(s) > 0, \
        f"Parsed structure is empty or has no sites (format detected: {fmt!r})."

    return s


def _parse_with_fmt(cell_str: str, fmt: str) -> Structure:
    """Dispatch to the appropriate pymatgen parser for *fmt*."""
    if fmt == "filepath":
        p = Path(cell_str.strip())
        assert p.is_file(), \
            f"Structure file path does not exist: {p}"
        text = p.read_text(encoding="utf-8", errors="replace")
        # Recurse: let the file content be format-detected on its own
        return parse_structure(text)

    if fmt == "poscar-escaped":
        text = cell_str.replace("\\n", "\n").replace("\\t", " ").strip()
        return Structure.from_str(text, fmt="poscar")

    if fmt == "json":
        return Structure.from_str(cell_str.strip(), fmt="json")

    if fmt in ("poscar", "cif", "xsf", "yaml"):
        return Structure.from_str(cell_str.strip(), fmt=fmt)

    # Should never reach here given the KNOWN_FORMATS assertion in parse_structure
    raise ValueError(f"No parser implemented for format {fmt!r}.")


# ── Explicit validation ───────────────────────────────────────────────────────
def assert_valid_cell(cell_str: str, *, context: str = "") -> None:
    """
    Assert that *cell_str* is a parseable crystal structure in a supported format.

    *context* is prepended to error messages (e.g. a row ID or column name)
    to help pinpoint the failing cell.

    Raises AssertionError with a descriptive message on any failure.
    """
    prefix = f"{context}: " if context else ""
    assert cell_str and cell_str.strip(), \
        f"{prefix}cell string is empty or whitespace-only."

    try:
        fmt = detect_format(cell_str)
    except AssertionError as exc:
        raise AssertionError(f"{prefix}{exc}") from exc

    assert fmt in KNOWN_FORMATS, \
        f"{prefix}unrecognised format (detect_format returned {fmt!r})."

    try:
        s = parse_structure(cell_str)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(
            f"{prefix}could not parse structure (format: {fmt!r}): {exc}"
        ) from exc

    assert len(s) > 0, \
        f"{prefix}parsed structure has no sites (format: {fmt!r})."

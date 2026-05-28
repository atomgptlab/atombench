"""
atombench._common — shared discovery and validation utilities.

All three CLI entry-points (atombench, atombench-plots, atombench-tables) accept
exactly one kind of input: a benchmark CSV file or a directory that contains
benchmark CSV files (directly or one per subdirectory).  The functions here
enforce that contract with assert statements so violations are caught early.

A benchmark CSV is any CSV whose header contains the three columns:
    id, target, prediction
"""
from __future__ import annotations

import csv as csv_mod
from pathlib import Path
from typing import List, Optional, Tuple


# ── Column contract ───────────────────────────────────────────────────────────
REQUIRED_COLUMNS = frozenset({"id", "target", "prediction"})


def _csv_columns(path: Path) -> Optional[frozenset]:
    """Return the lowercased header fields of *path*, or None on read error."""
    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv_mod.DictReader(fh)
            fields = reader.fieldnames or []
            return frozenset(f.strip().lower() for f in fields)
    except Exception:
        return None


def is_benchmark_csv(path: Path) -> bool:
    """Return True iff *path* is a readable CSV with the required columns."""
    if not (path.is_file() and path.suffix.lower() == ".csv"):
        return False
    cols = _csv_columns(path)
    return cols is not None and REQUIRED_COLUMNS.issubset(cols)


def assert_benchmark_csv(path: Path) -> None:
    """Assert that *path* is a valid benchmark CSV; raise AssertionError if not."""
    assert path.exists(),          f"Path does not exist: {path}"
    assert path.is_file(),         f"Expected a file, got a directory: {path}"
    assert path.suffix.lower() == ".csv", \
        f"Input file must be a .csv, got: {path.name}"
    cols = _csv_columns(path)
    assert cols is not None, f"Could not read CSV: {path}"
    assert REQUIRED_COLUMNS.issubset(cols), (
        f"Benchmark CSV {path.name} is missing required columns "
        f"{REQUIRED_COLUMNS - cols}; found: {cols}"
    )


def _newest_benchmark_csv_in_dir(d: Path) -> Optional[Path]:
    """Walk *d* and return the newest benchmark CSV found, or None."""
    best: Optional[Tuple[float, Path]] = None
    for child in d.rglob("*.csv"):
        if not child.is_file():
            continue
        cols = _csv_columns(child)
        if cols is None or not REQUIRED_COLUMNS.issubset(cols):
            continue
        mt = child.stat().st_mtime
        if best is None or mt > best[0]:
            best = (mt, child)
    return best[1] if best else None


def discover_benchmark_csvs(path: Path) -> List[Tuple[str, Path]]:
    """
    Return (bench_name, csv_path) pairs discovered from *path*.

    Accepted inputs
    ---------------
    - A single benchmark CSV file.
    - A flat directory containing one or more benchmark CSVs directly.
    - A structured directory whose immediate subdirectories each contain a
      benchmark CSV (anywhere inside them).

    Asserts
    -------
    - *path* must exist.
    - *path* must be a file or a directory (not a symlink to something else, etc.).
    - If *path* is a file it must be a valid benchmark CSV.
    - If *path* is a directory it must contain at least one benchmark CSV.
    """
    assert path.exists(), f"Input path does not exist: {path}"
    assert path.is_file() or path.is_dir(), \
        f"Input must be a CSV file or a directory, got: {path}"

    if path.is_file():
        assert_benchmark_csv(path)
        return [(path.stem, path)]

    # Directory: try flat layout first (CSVs directly inside)
    direct = sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv" and is_benchmark_csv(p)
    )
    if direct:
        return [(p.stem, p) for p in direct]

    # Structured layout: one benchmark CSV per immediate subdirectory
    results: List[Tuple[str, Path]] = []
    for entry in sorted(path.iterdir()):
        if not entry.is_dir():
            continue
        csv = _newest_benchmark_csv_in_dir(entry)
        if csv is not None:
            results.append((entry.name, csv))

    assert results, (
        f"No benchmark CSV files found under {path}. "
        f"Expected either CSV files directly in the directory or one per subdirectory. "
        f"A benchmark CSV must have columns: {sorted(REQUIRED_COLUMNS)}."
    )
    return results

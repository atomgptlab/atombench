#!/usr/bin/env python3
"""
atombench.verify — cross-benchmark consistency checks.

Entry point:  atombench-verify <path>

Checks that all benchmark CSVs for the same dataset (identified by 'alex' or
'jarvis' in the directory name) share an identical test-set ID list.  Also
confirms that TARGET structures are consistent across benchmarks (warning only).
Missing IDs that are listed in a sibling *.misses.csv are treated as expected
and do not cause failure.

Exits 0 on success, 1 if unaccounted ID mismatches are found.
"""
from __future__ import annotations

import csv as csv_mod
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import click
import numpy as np
from pymatgen.core import Structure

from atombench._common import discover_benchmark_csvs
from atombench._structure_io import parse_structure


# ── Structure helpers ─────────────────────────────────────────────────────────
def _parse_structure(cell_str: str) -> Optional[Structure]:
    try:
        return parse_structure(cell_str)
    except Exception:
        return None


def _structure_signature(cell_str: Optional[str], decimals: int = 1) -> tuple:
    """
    Order-insensitive structural signature rounded to *decimals* decimal places.
    Returns a hashable tuple; special sentinel strings on failure.
    """
    if not cell_str or not str(cell_str).strip():
        return ("__EMPTY__",)
    s = _parse_structure(str(cell_str))
    if s is None:
        return ("__PARSE_FAILED__",)

    lat = np.round(s.lattice.matrix, decimals)
    lat_sig = tuple(lat.reshape(-1).tolist())

    buckets: Dict[str, List[Tuple]] = {}
    for site in s:
        sp = str(site.species_string)
        coord = tuple(np.round(site.frac_coords, decimals).tolist())
        buckets.setdefault(sp, []).append(coord)
    for sp in buckets:
        buckets[sp].sort()

    species_sig = tuple(sorted((sp, tuple(coords)) for sp, coords in buckets.items()))
    return ("OK", lat_sig, species_sig)


# ── CSV loading ───────────────────────────────────────────────────────────────
def _load_id_to_target_sig(csv_path: Path) -> Dict[str, tuple]:
    mapping: Dict[str, tuple] = {}
    with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv_mod.DictReader(fh)
        for row in reader:
            _id = str(row.get("id", "")).strip()
            if not _id:
                continue
            sig = _structure_signature(row.get("target"))
            if _id in mapping and mapping[_id] != sig:
                click.echo(
                    f"[WARN] Duplicate ID with differing TARGET in {csv_path.name}: {_id}",
                    err=True,
                )
                continue
            mapping.setdefault(_id, sig)
    return mapping


def _load_misses_ids(csv_path: Path) -> Set[str]:
    misses = csv_path.with_name(f"{csv_path.stem}.misses{csv_path.suffix}")
    if not misses.exists():
        return set()
    ids: Set[str] = set()
    with misses.open("r", newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv_mod.DictReader(fh)
        keymap = {(fn or "").strip().lower(): fn for fn in (reader.fieldnames or [])}
        id_key = keymap.get("id")
        if id_key is None:
            return ids
        for row in reader:
            v = str(row.get(id_key, "")).strip()
            if v:
                ids.add(v)
    return ids


# ── Dataset grouping ──────────────────────────────────────────────────────────
def _dataset_tag(name: str) -> Optional[str]:
    nl = name.lower()
    if "jarvis" in nl:
        return "jarvis"
    if "alex" in nl or "alexandria" in nl:
        return "alex"
    return None


def group_by_dataset(
    pairs: List[Tuple[str, Path]]
) -> Dict[str, List[Tuple[str, Path]]]:
    """Group ``(name, csv_path)`` pairs by dataset tag (``alex`` / ``jarvis``)."""
    groups: Dict[str, List[Tuple[str, Path]]] = {}
    ungrouped = []
    for name, csv_path in pairs:
        tag = _dataset_tag(name)
        if tag:
            groups.setdefault(tag, []).append((name, csv_path))
        else:
            ungrouped.append(name)
    if ungrouped:
        click.echo(
            f"[INFO] {len(ungrouped)} benchmark(s) not grouped (no 'alex'/'jarvis' tag): "
            + ", ".join(ungrouped),
            err=True,
        )
    return groups


# ── Comparison logic ──────────────────────────────────────────────────────────
def compare_group(
    dataset_key: str,
    items: List[Tuple[str, Path]],
    show_diff: int,
) -> bool:
    """
    Return True if ID sets are consistent (or differences are fully accounted for
    by ``*.misses.csv`` files); False on unaccounted mismatches.
    """
    pretty = "Alexandria" if dataset_key == "alex" else "JARVIS"
    if len(items) < 2:
        click.echo(f"[INFO] Only 1 benchmark for {pretty} — skipping consensus check.")
        return True

    loaded = []
    for name, csv_path in items:
        try:
            mapping   = _load_id_to_target_sig(csv_path)
            misses    = _load_misses_ids(csv_path)
            loaded.append((name, csv_path, mapping, misses))
        except Exception as e:
            click.echo(f"[WARN] Could not read {csv_path}: {e}", err=True)
            return True   # read failure is not an ID mismatch

    id_sets  = [set(m) for _, _, m, _ in loaded]
    union    = set().union(*id_sets)

    unaccounted = False
    accounted   = False
    for name, csv_path, mapping, misses in loaded:
        missing   = union - set(mapping)
        uncovered = missing - misses
        covered   = missing & misses
        if covered:
            accounted = True
        if uncovered:
            unaccounted = True

    if unaccounted:
        click.echo(f"[MISMATCH] ID sets differ for {pretty}.")
        for name, csv_path, mapping, misses in loaded:
            missing   = union - set(mapping)
            uncovered = missing - misses
            covered   = missing & misses
            if uncovered:
                sample = ", ".join(sorted(uncovered)[:show_diff])
                click.echo(f"  {name}: {len(uncovered)} IDs missing, not in misses CSV"
                           f" (e.g., {sample})")
            if covered:
                click.echo(f"  {name}: {len(covered)} IDs missing, accounted for by"
                           f" {csv_path.stem}.misses.csv")
        return False

    if accounted:
        click.echo(f"[OK] {pretty}: ID differences fully accounted for by *.misses.csv files.")
    else:
        # Also verify TARGET structures on common IDs
        common  = set.intersection(*id_sets)
        base_name, _, base_map, _ = loaded[0]
        mismatches = []
        for _id in sorted(common):
            for name, _, mapping, _ in loaded[1:]:
                if mapping.get(_id) != base_map.get(_id):
                    mismatches.append((_id, base_name, name))
                    if len(mismatches) >= show_diff:
                        break
            if len(mismatches) >= show_diff:
                break
        if mismatches:
            click.echo(
                f"[WARN] {pretty}: TARGET structures differ on {len(mismatches)} IDs "
                f"(rounded, species-order-insensitive):",
                err=True,
            )
            for _id, a, b in mismatches:
                click.echo(f"  ID {_id}: {a} ≠ {b}", err=True)
        else:
            click.echo(f"[OK] {pretty}: IDs and TARGET structures match across all benchmarks.")

    return True


# ── CLI ───────────────────────────────────────────────────────────────────────
@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--show-diff", default=5, show_default=True, type=int,
              help="Max example differences to display per mismatch.")
def main(path: Path, show_diff: int) -> None:
    """Verify cross-benchmark consistency for a set of benchmark CSV files.

    \b
    PATH can be:
      • a single benchmark CSV file
      • a directory of benchmark CSV files (flat or one per subdirectory)

    Benchmarks are grouped by dataset tag ('alex'/'jarvis' in the directory
    name).  Within each group, ID sets must match (modulo *.misses.csv).
    TARGET structure consistency is checked and reported as a warning.

    Exits 0 on success, 1 if unaccounted ID mismatches are found.
    """
    pairs  = discover_benchmark_csvs(path.resolve())
    groups = group_by_dataset(pairs)

    if not groups:
        raise click.ClickException("No dataset groups found (need 'alex' or 'jarvis' in names).")

    any_fail = False
    for dataset_key, items in sorted(groups.items()):
        items.sort(key=lambda t: t[0].lower())
        if not compare_group(dataset_key, items, show_diff):
            any_fail = True

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()

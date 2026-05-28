#!/usr/bin/env python3
"""
atombench — compute metrics, generate plots, and write summary tables for
crystal structure reconstruction benchmarks.

Usage:
    atombench path/to/benchmark.csv  output/dir/
    atombench path/to/directory/     output/dir/
"""
from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import mean_absolute_error

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import amd

from atombench._common import discover_benchmark_csvs
from atombench._structure_io import parse_structure

from atombench.plots import (
    plot_kld_bar_chart,
    plot_mae_abc_bar_chart,
    plot_mae_angles_bar_chart,
    plot_rmse_bar_chart,
    plot_ccrmsd_bar_chart,
    plot_match_rate_bar_chart,
    plot_crystal_system_mae_from_json,
    plot_reconstruction_grids,
)

from atombench.tables import extract_metrics, build_metrics_tex

# ── Constants ─────────────────────────────────────────────────────────────────
CRYSYS_ALL = [
    "triclinic", "monoclinic", "orthorhombic",
    "tetragonal", "trigonal", "hexagonal", "cubic",
]
CRYSYS_PLOT_ORDER = [
    "cubic", "hexagonal", "trigonal",
    "tetragonal", "orthorhombic", "monoclinic",
]
PARAMS_ALL = ("a", "b", "c", "alpha", "beta", "gamma")


# ── Pymatgen / Niggli helpers ──────────────────────────────────────────────────
@lru_cache(maxsize=20000)
def _reduced_struct(cell_str: str) -> Structure:
    s = parse_structure(cell_str)
    s = s.get_primitive_structure()
    return s.get_reduced_structure(reduction_algo="niggli")


@lru_cache(maxsize=20000)
def _niggli_params(cell_str: str) -> Tuple[float, ...]:
    s = _reduced_struct(cell_str)
    return (*s.lattice.abc, *s.lattice.angles)


# ── Statistical helpers ────────────────────────────────────────────────────────
def _kld(p_vals, q_vals) -> float:
    p = np.asarray(p_vals, dtype=np.float64)
    q = np.asarray(q_vals, dtype=np.float64)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    p /= p.sum()
    q /= q.sum()
    return float(scipy_entropy(p, q))


def _mae(x, y) -> float:
    return float(mean_absolute_error(x, y)) if len(x) > 0 else float("nan")


# ── Metrics computation ────────────────────────────────────────────────────────
def _extract_niggli_pairs(df: pd.DataFrame) -> Tuple[Dict[str, list], Dict[str, list]]:
    xs = {k: [] for k in PARAMS_ALL}
    ys = {k: [] for k in PARAMS_ALL}
    for _, row in df.iterrows():
        try:
            tp = _niggli_params(str(row["target"]))
            pp = _niggli_params(str(row["prediction"]))
            for i, k in enumerate(PARAMS_ALL):
                xs[k].append(tp[i])
                ys[k].append(pp[i])
        except Exception:
            continue
    return xs, ys


def _compute_atomgen_rmse(df: pd.DataFrame) -> dict:
    STOL = 0.5
    matcher = StructureMatcher(stol=STOL, angle_tol=10, ltol=0.3)
    norm_rms, rms_ang = [], []
    n_total = n_matched = 0

    for _, row in df.iterrows():
        try:
            st = _reduced_struct(str(row["target"]))
            sp = _reduced_struct(str(row["prediction"]))
            n_total += 1
            rms = matcher.get_rms_dist(sp, st)
            if rms is not None:
                r   = float(rms[0])
                vol = float(st.lattice.volume)
                if np.isfinite(vol) and vol > 0:
                    sc = float(np.cbrt(vol))
                    if np.isfinite(sc) and sc > 0:
                        norm_rms.append(r / sc)
                        rms_ang.append(r)
                        n_matched += 1
        except Exception:
            continue

    base = {"stol": STOL, "n_matched": n_matched, "n_total": n_total}
    if n_total == 0 or n_matched == 0:
        return {
            "mean_normalized_cartesian_rms": float("nan"),
            "mean_cartesian_rms_angstrom":   float("nan"),
            "match_rate": float("nan") if n_total == 0 else round(n_matched / n_total, 6),
            **base,
        }
    return {
        "mean_normalized_cartesian_rms": round(float(np.mean(norm_rms)), 6),
        "mean_cartesian_rms_angstrom":   round(float(np.mean(rms_ang)), 6),
        "match_rate":                    round(n_matched / n_total, 6),
        **base,
    }


@lru_cache(maxsize=20000)
def _amd_vec(cell_str: str, k: int) -> tuple:
    s  = _reduced_struct(cell_str)
    ps = amd.periodicset_from_pymatgen_structure(s)
    return tuple(np.asarray(amd.AMD(ps, int(k)), dtype=np.float64).tolist())


def _compute_ccrmse(df: pd.DataFrame, k: int) -> Tuple[float, int]:
    s2, n = 0.0, 0
    for _, row in df.iterrows():
        try:
            vt = np.asarray(_amd_vec(str(row["target"]),     k), dtype=np.float64)
            vp = np.asarray(_amd_vec(str(row["prediction"]), k), dtype=np.float64)
            d  = float(np.max(np.abs(vp - vt)))
            if not np.isfinite(d) or d < 0:
                continue
            s2 += d * d
            n  += 1
        except Exception:
            continue
    return (float(np.sqrt(s2 / n)), n) if n > 0 else (float("nan"), 0)


def _crystal_system(s: Structure, symprec: float) -> Optional[str]:
    try:
        sga  = SpacegroupAnalyzer(s, symprec=symprec)
        conv = sga.get_conventional_standard_structure()
        cs   = SpacegroupAnalyzer(conv, symprec=symprec).get_crystal_system()
        cs   = cs.lower() if isinstance(cs, str) else None
        return cs if cs in CRYSYS_ALL else None
    except Exception:
        return None


def _compute_crystal_system_mae(df: pd.DataFrame, symprec: float, kmin: int) -> dict:
    errors: Dict[str, list] = defaultdict(list)
    for _, r in df.iterrows():
        try:
            t  = str(r["target"])
            p  = str(r["prediction"])
            st = _reduced_struct(t)
            cs = _crystal_system(st, symprec)
            if cs is None:
                continue
            tp = _niggli_params(t)
            pp = _niggli_params(p)
            errors[cs].append({k: abs(pp[i] - tp[i]) for i, k in enumerate(PARAMS_ALL)})
        except Exception:
            continue

    by_system = []
    for cs in CRYSYS_PLOT_ORDER:
        rows = errors.get(cs, [])
        if len(rows) < kmin:
            continue
        by_system.append({
            "crystal_system":    cs,
            "n_reconstructions": len(rows),
            "mae": {
                k: float(round(float(np.mean([rw[k] for rw in rows])), 6))
                for k in PARAMS_ALL
            },
        })
    return {"by_system": by_system, "symprec": symprec, "kmin": kmin}


def compute_metrics(df: pd.DataFrame, bench_name: str, *,
                    amd_k: int, symprec: float, kmin: int) -> dict:
    """Compute all reconstruction metrics for one benchmark DataFrame."""
    click.echo("    extracting Niggli params …")
    xs, ys = _extract_niggli_pairs(df)

    mae_vals = {k: _mae(xs[k], ys[k])                             for k in PARAMS_ALL}
    kld_vals = {k: _kld(xs[k], ys[k]) if xs[k] else float("nan") for k in PARAMS_ALL}

    click.echo("    StructureMatcher RMSE …")
    rmse = _compute_atomgen_rmse(df)

    click.echo(f"    ccRMSD/AMD (k={amd_k}) …")
    ccrmsd_val, n_ccrmsd = _compute_ccrmse(df, amd_k)

    click.echo(f"    crystal-system MAE (symprec={symprec}, kmin={kmin}) …")
    crysys = _compute_crystal_system_mae(df, symprec, kmin)

    return {
        "benchmark_name": bench_name,
        "KLD": {k: kld_vals[k] for k in PARAMS_ALL},
        "MAE": {"average_mae": {k: mae_vals[k] for k in PARAMS_ALL}},
        "RMSE": {"AtomGen": rmse},
        "ccRMSD": {"value": ccrmsd_val, "amd_k": amd_k, "n_eval": n_ccrmsd},
        "crystal_system_mae": crysys,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────
@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.argument("outdir", type=click.Path(path_type=Path))
@click.option("--amd-k", default=100, show_default=True, type=int,
              help="AMD vector length k.")
@click.option("--symprec", default=0.1, show_default=True, type=float,
              help="Symmetry tolerance for SpacegroupAnalyzer (Å).")
@click.option("--kmin", default=10, show_default=True, type=int,
              help="Minimum structures per crystal system for crystal-system MAE charts.")
def main(
    path: Path,
    outdir: Path,
    amd_k: int,
    symprec: float,
    kmin: int,
) -> None:
    """Compute metrics, generate plots, and write summary tables.

    \b
    PATH can be:
      • a single benchmark CSV file  (columns: id, target, prediction)
      • a directory of CSV files     (one benchmark per CSV or subdirectory)

    OUTDIR is the output directory:
      • figures/               — all plots (PNG)
      • numerical_calculations/ — metrics_table.{json,tex}, epic_metrics.csv

    Per-benchmark metrics.json files are written beside each CSV for caching.
    """
    outdir     = Path(outdir).resolve()
    figures_dir = outdir / "figures"
    num_dir     = outdir / "numerical_calculations"
    figures_dir.mkdir(parents=True, exist_ok=True)
    num_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = discover_benchmark_csvs(path.resolve())

    click.echo(f"Found {len(benchmarks)} benchmark(s).")

    all_results: List[Tuple[str, Path, dict]] = []

    for bench_name, csv_path in benchmarks:
        click.echo(f"\n── {bench_name}")
        metrics_path = csv_path.parent / "metrics.json"

        df = pd.read_csv(csv_path)
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})
        missing = [c for c in ("target", "prediction") if c not in df.columns]
        if missing:
            click.echo(f"  ⚠ missing columns {missing} — skipping", err=True)
            continue
        click.echo(f"  {len(df)} rows from {csv_path.name}")
        metrics = compute_metrics(df, bench_name, amd_k=amd_k, symprec=symprec, kmin=kmin)
        with metrics_path.open("w") as fh:
            json.dump(metrics, fh, indent=2)
        click.echo(f"  ✓ metrics.json → {metrics_path.parent.name}/")

        all_results.append((bench_name, csv_path, metrics))

    if not all_results:
        raise click.ClickException("No benchmarks were successfully processed.")

    # ── Plots ──────────────────────────────────────────────────────────────────
    click.echo("\n── Plots")

    all_metrics = [m for _, _, m in all_results]
    rows = [pd.json_normalize(m, sep=".", max_level=3).iloc[0].to_dict() for m in all_metrics]
    df_metrics = pd.DataFrame(rows)

    plot_kld_bar_chart(df_metrics, figures_dir)
    plot_mae_abc_bar_chart(df_metrics, figures_dir)
    plot_mae_angles_bar_chart(df_metrics, figures_dir)
    plot_rmse_bar_chart(df_metrics, figures_dir)
    plot_ccrmsd_bar_chart(df_metrics, figures_dir)
    plot_match_rate_bar_chart(df_metrics, figures_dir)

    if any("crystal_system_mae" in m for m in all_metrics):
        try:
            plot_crystal_system_mae_from_json(all_metrics, figures_dir, kmin=kmin)
        except Exception as e:
            click.echo(f"  ⚠ crystal-system MAE charts failed — {e}", err=True)

    bench_roots = {csv_path.parent.parent for _, csv_path, _ in all_results}
    grid_root = bench_roots.pop() if len(bench_roots) == 1 else all_results[0][1].parent.parent
    try:
        plot_reconstruction_grids(grid_root, figures_dir)
    except Exception as e:
        click.echo(f"  ⚠ reconstruction grids failed — {e}", err=True)

    # ── Tables ─────────────────────────────────────────────────────────────────
    click.echo("\n── Tables")

    table_results = {name: extract_metrics(m) for name, _, m in all_results}

    out_json = num_dir / "metrics_table.json"
    out_tex  = num_dir / "metrics_table.tex"
    out_csv  = num_dir / "epic_metrics.csv"

    out_json.write_text(json.dumps(table_results, indent=2) + "\n")
    out_tex.write_text(build_metrics_tex(table_results))

    raw_records = []
    for name, e in table_results.items():
        rec = {"benchmark_name": name}
        rec.update(pd.json_normalize(e, sep=".").iloc[0].to_dict())
        raw_records.append(rec)
    df_flat = pd.DataFrame(raw_records)
    cols = ["benchmark_name"] + [c for c in df_flat.columns if c != "benchmark_name"]
    df_flat[cols].to_csv(out_csv, index=False)

    click.echo(f"  ✓ metrics_table.json")
    click.echo(f"  ✓ metrics_table.tex")
    click.echo(f"  ✓ epic_metrics.csv")

    click.echo(f"\nAll done. Output: {outdir}")


if __name__ == "__main__":
    main()

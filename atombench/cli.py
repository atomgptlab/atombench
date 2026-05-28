#!/usr/bin/env python3
"""
atombench CLI — compute metrics and generate plots for crystal structure reconstruction benchmarks.

Usage:
    atombench path/to/benchmark.csv
    atombench path/to/directory/of/benchmarks/
"""
from __future__ import annotations

import csv as csv_mod
import json
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click

# matplotlib must be configured before pyplot is imported
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({
    "font.family": "serif",
    "axes.linewidth": 0.8,
    "patch.linewidth": 0.0,
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif", "STIX"],
})

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import mean_absolute_error

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import amd

# ── Constants ─────────────────────────────────────────────────────────────────
CRYSYS_ALL = [
    "triclinic", "monoclinic", "orthorhombic",
    "tetragonal", "trigonal", "hexagonal", "cubic",
]
CRYSYS_PLOT_ORDER = [
    "cubic", "hexagonal", "trigonal",
    "tetragonal", "orthorhombic", "monoclinic",
]
AX_LABEL_MAP = {
    "a": r"$a$", "b": r"$b$", "c": r"$c$",
    "alpha": r"$\alpha$", "beta": r"$\beta$", "gamma": r"$\gamma$",
}
PARAMS_ALL = ("a", "b", "c", "alpha", "beta", "gamma")

WVU_BLUE    = "#002855"
WVU_GOLD    = "#EEAA00"
LEN_GRANITE = ["#4A6272", "#89A9BC", "#D6E3EC"]
ANG_GRANITE = ["#6A5560", "#B08A97", "#E7D6DC"]


# ── Pymatgen / Niggli helpers ──────────────────────────────────────────────────
@lru_cache(maxsize=20000)
def _reduced_struct(poscar_text: str) -> Structure:
    s = Structure.from_str(poscar_text.replace("\\n", "\n"), fmt="poscar")
    s = s.get_primitive_structure()
    return s.get_reduced_structure(reduction_algo="niggli")


@lru_cache(maxsize=20000)
def _niggli_params(poscar_text: str) -> Tuple[float, ...]:
    s = _reduced_struct(poscar_text)
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
            "mean_cartesian_rms_angstrom": float("nan"),
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
def _amd_vec(poscar_text: str, k: int) -> tuple:
    s     = _reduced_struct(poscar_text)
    motif = np.asarray(s.cart_coords, dtype=np.float64)
    cell  = np.asarray(s.lattice.matrix, dtype=np.float64)
    return tuple(np.asarray(amd.AMD((motif, cell), int(k)), dtype=np.float64).tolist())


def _compute_ccrmse(df: pd.DataFrame, k: int, tau: float) -> Tuple[float, int]:
    if tau <= 0:
        return float("nan"), 0
    s2, n = 0.0, 0
    for _, row in df.iterrows():
        try:
            vt = np.asarray(_amd_vec(str(row["target"]),     k), dtype=np.float64)
            vp = np.asarray(_amd_vec(str(row["prediction"]), k), dtype=np.float64)
            d  = float(np.max(np.abs(vp - vt)))
            if not np.isfinite(d) or d < 0:
                continue
            dc  = d if d <= tau else tau
            s2 += dc * dc
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
                    tau: float, amd_k: int, symprec: float, kmin: int) -> dict:
    """Compute all reconstruction metrics for one benchmark DataFrame."""
    click.echo("    extracting Niggli params …")
    xs, ys = _extract_niggli_pairs(df)

    mae_vals = {k: _mae(xs[k], ys[k])                                    for k in PARAMS_ALL}
    kld_vals = {k: _kld(xs[k], ys[k]) if xs[k] else float("nan")         for k in PARAMS_ALL}

    click.echo("    StructureMatcher RMSE …")
    rmse = _compute_atomgen_rmse(df)

    click.echo(f"    ccRMSE/AMD (k={amd_k}, tau={tau}) …")
    ccrmse_val, n_ccrmse = _compute_ccrmse(df, amd_k, tau)

    click.echo(f"    crystal-system MAE (symprec={symprec}, kmin={kmin}) …")
    crysys = _compute_crystal_system_mae(df, symprec, kmin)

    return {
        "benchmark_name": bench_name,
        "KLD": {k: kld_vals[k] for k in PARAMS_ALL},
        "MAE": {"average_mae": {k: mae_vals[k] for k in PARAMS_ALL}},
        "RMSE": {"AtomGen": rmse},
        "ccRMSE": {"value": ccrmse_val, "tau": tau, "amd_k": amd_k, "n_eval": n_ccrmse},
        "crystal_system_mae": crysys,
    }


# ── Discovery helpers ──────────────────────────────────────────────────────────
def _is_benchmark_csv(path: Path) -> bool:
    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv_mod.DictReader(fh)
            fields = {f.strip().lower() for f in (reader.fieldnames or [])}
            return {"id", "target", "prediction"}.issubset(fields)
    except Exception:
        return False


def _find_csv_in_dir(d: Path) -> Optional[Path]:
    """Return the newest benchmark CSV found anywhere under directory d."""
    latest: Optional[Tuple[float, Path]] = None
    for root, _, files in os.walk(d):
        for f in files:
            if not f.lower().endswith(".csv"):
                continue
            path = Path(root) / f
            if _is_benchmark_csv(path):
                mt = path.stat().st_mtime
                if latest is None or mt > latest[0]:
                    latest = (mt, path)
    return latest[1] if latest else None


def discover_benchmarks(path: Path) -> List[Tuple[str, Path]]:
    """
    Return (bench_name, csv_path) pairs from a file or directory.

    - Single CSV file  → one benchmark.
    - Flat directory   → one benchmark per .csv file found directly inside.
    - Structured dir   → one benchmark per subdirectory that contains a CSV.
      (If both direct CSVs and subdirs-with-CSVs exist, direct CSVs take priority.)
    """
    if path.is_file():
        return [(path.stem, path)]

    direct_csvs = [p for p in sorted(path.glob("*.csv")) if _is_benchmark_csv(p)]
    if direct_csvs:
        return [(p.stem, p) for p in direct_csvs]

    subdir_benchmarks = []
    for entry in sorted(path.iterdir()):
        if not entry.is_dir():
            continue
        csv = _find_csv_in_dir(entry)
        if csv:
            subdir_benchmarks.append((entry.name, csv))
    return subdir_benchmarks


# ── Plotting helpers ───────────────────────────────────────────────────────────
def _style_bar_axes(ax, ylabel: str, title: str) -> None:
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=22)
    ax.legend(title="Lattice Parameter", title_fontsize=15, fontsize=15)
    plt.xticks(rotation=30, ha="right", fontsize=13)
    plt.yticks(fontsize=15)
    plt.tight_layout()


# ── Individual plot functions ──────────────────────────────────────────────────
def plot_distribution(bench_name: str, csv_path: Path, outdir: Path) -> None:
    plt.rcParams.update({"font.size": 18})
    df = pd.read_csv(csv_path)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

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

    def _overlay(ax, x, y, bins, xlabel, title):
        wx = np.ones_like(x, dtype=float) / max(1, len(x)) * 100
        wy = np.ones_like(y, dtype=float) / max(1, len(y)) * 100
        ax.hist(x, bins=bins, weights=wx, alpha=0.6, color="tab:blue", label="target")
        ax.hist(y, bins=bins, weights=wy, alpha=0.6, color="plum",     label="predicted")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        return ax

    fig  = plt.figure(figsize=(14, 8))
    grid = GridSpec(2, 3)

    _overlay(plt.subplot(grid[0, 0]),
             xs["a"], ys["a"], np.arange(2, 7, 0.1),
             r"a ($\AA$)", "(a)").set_ylabel("Materials dist.")
    plt.legend()
    _overlay(plt.subplot(grid[0, 1]), xs["c"],     ys["c"],     np.arange(2, 7, 0.1),    r"c ($\AA$)",           "(b)")
    _overlay(plt.subplot(grid[0, 2]), xs["gamma"], ys["gamma"], np.arange(30, 150, 10),  r"$\gamma$ ($^\circ$)", "(c)")

    x_spg, y_spg, x_Z, y_Z, x_lat, y_lat = [], [], [], [], [], []
    lat_order  = ["triclinic", "monoclinic", "orthorhombic",
                  "tetragonal", "trigonal", "hexagonal", "cubic"]
    lat_to_idx = {name: i for i, name in enumerate(lat_order)}

    for _, row in df.iterrows():
        try:
            st    = _reduced_struct(str(row["target"]))
            sp    = _reduced_struct(str(row["prediction"]))
            x_Z.append(st.composition.weight)
            y_Z.append(sp.composition.weight)
            sga_t = SpacegroupAnalyzer(st, symprec=0.1)
            sga_p = SpacegroupAnalyzer(sp, symprec=0.1)
            x_spg.append(sga_t.get_space_group_number())
            y_spg.append(sga_p.get_space_group_number())
            x_lat.append(sga_t.get_crystal_system())
            y_lat.append(sga_p.get_crystal_system())
        except Exception:
            continue

    _overlay(plt.subplot(grid[1, 0]),
             x_spg, y_spg, np.arange(1, 231, 10),
             "Spacegroup number", "(d)").set_ylabel("Materials dist.")

    valid  = [(lx, ly) for lx, ly in zip(x_lat, y_lat) if lx and ly]
    xl, yl = zip(*valid) if valid else ([], [])
    xl_c   = np.bincount([lat_to_idx[l] for l in xl], minlength=len(lat_order))
    yl_c   = np.bincount([lat_to_idx[l] for l in yl], minlength=len(lat_order))
    ax_lat = plt.subplot(grid[1, 1])
    pos    = np.arange(len(lat_order))
    ax_lat.bar(pos, xl_c, width=0.4, alpha=0.6, label="target",    color="tab:blue")
    ax_lat.bar(pos, yl_c, width=0.4, alpha=0.6, label="predicted", color="plum")
    ax_lat.set_xticks(pos)
    ax_lat.set_xticklabels((pos + 1).tolist(), rotation=0, ha="center")
    ax_lat.set_xlabel("Crystal system number")
    ax_lat.set_title("(e)")

    _overlay(plt.subplot(grid[1, 2]), x_Z, y_Z, np.arange(15, 2000, 100), "Weight (AMU)", "(f)")

    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.suptitle(bench_name, fontsize=30)
    out_png = outdir / f"{bench_name}_distribution.png"
    plt.savefig(out_png, format="png", dpi=200)
    plt.close(fig)
    click.echo(f"  ✓ {out_png.name}")


def plot_kld_bar_chart(df: pd.DataFrame, outdir: Path) -> None:
    kld_cols = [f"KLD.{k}" for k in PARAMS_ALL]
    if any(c not in df.columns for c in kld_cols):
        click.echo("  ⚠ Missing KLD columns — skipping KLD bar chart", err=True)
        return
    kld_df = (df.set_index("benchmark_name")[kld_cols]
                .rename(columns=lambda c: AX_LABEL_MAP[c.split(".")[-1]]))
    fig, ax = plt.subplots(figsize=(10, 8))
    kld_df.plot(kind="bar", edgecolor="k", ax=ax)
    _style_bar_axes(ax, "KL Divergence (Nats)",
                    "KL Divergence of Predicted vs. Target\nLattice-Parameter Distributions")
    plt.savefig(outdir / "comparison_bar_chart.png", dpi=300)
    plt.close(fig)
    click.echo("  ✓ comparison_bar_chart.png")


def plot_mae_abc_bar_chart(df: pd.DataFrame, outdir: Path) -> None:
    for cand in ([f"MAE.average_mae.{k}" for k in PARAMS_ALL],
                 [f"MAE.{k}" for k in PARAMS_ALL]):
        if all(c in df.columns for c in cand):
            mae_cols = cand
            break
    else:
        click.echo("  ⚠ Missing MAE columns — skipping MAE bar charts", err=True)
        return
    mae_df = (df.set_index("benchmark_name")[mae_cols]
                .rename(columns=lambda c: AX_LABEL_MAP[c.split(".")[-1]]))
    length_cols = [AX_LABEL_MAP[k] for k in ("a", "b", "c")]
    fig, ax = plt.subplots(figsize=(10, 8))
    mae_df[length_cols].plot(kind="bar", edgecolor="k", ax=ax)
    _style_bar_axes(ax, "Mean Absolute Error (Å)", "Mean Absolute Error – Lattice Lengths (Å)")
    plt.savefig(outdir / "mae_bar_chart_abc.png", dpi=300)
    plt.close(fig)
    click.echo("  ✓ mae_bar_chart_abc.png")


def plot_mae_angles_bar_chart(df: pd.DataFrame, outdir: Path) -> None:
    for cand in ([f"MAE.average_mae.{k}" for k in PARAMS_ALL],
                 [f"MAE.{k}" for k in PARAMS_ALL]):
        if all(c in df.columns for c in cand):
            mae_cols = cand
            break
    else:
        return
    mae_df = (df.set_index("benchmark_name")[mae_cols]
                .rename(columns=lambda c: AX_LABEL_MAP[c.split(".")[-1]]))
    angle_cols = [AX_LABEL_MAP[k] for k in ("alpha", "beta", "gamma")]
    fig, ax = plt.subplots(figsize=(10, 8))
    mae_df[angle_cols].plot(kind="bar", edgecolor="k",
                            color=["red", "purple", "brown"], ax=ax)
    _style_bar_axes(ax, "Mean Absolute Error (°)", "Mean Absolute Error – Lattice Angles (°)")
    plt.savefig(outdir / "mae_bar_chart_angles.png", dpi=300)
    plt.close(fig)
    click.echo("  ✓ mae_bar_chart_angles.png")


def plot_rmse_bar_chart(df: pd.DataFrame, outdir: Path) -> None:
    for cand in ["RMSE.AtomGen.mean_cartesian_rms_angstrom",
                 "RMSE.AtomGen.mean_normalized_cartesian_rms",
                 "RMSE.AtomGen", "RMSE"]:
        if cand in df.columns:
            rmse_col = cand
            break
    else:
        click.echo("  ⚠ No RMSE column found — skipping RMSE bar chart", err=True)
        return

    plot_df = df[["benchmark_name", rmse_col]].rename(columns={rmse_col: "RMSE"}).copy()
    unique_names = list(dict.fromkeys(plot_df["benchmark_name"]))
    palette      = plt.cm.tab10.colors
    color_map    = {n: palette[i % len(palette)] for i, n in enumerate(unique_names)}

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = np.arange(len(plot_df))
    ax.bar(pos, plot_df["RMSE"].astype(float).tolist(),
           width=0.55, edgecolor="k", linewidth=0.8,
           color=[color_map[n] for n in plot_df["benchmark_name"]])
    ax.set_xticks(pos)
    ax.set_xticklabels(plot_df["benchmark_name"].tolist(), rotation=30, ha="right", fontsize=13)
    handles = [mpatches.Patch(color=color_map[n], label=n) for n in unique_names]
    ax.legend(handles=handles, title_fontsize=15, fontsize=15)
    ax.set_ylabel("Average RMSE (Å)", fontsize=16)
    ax.set_title("Average Root Mean Squared Error\nfor Predicted vs. Target Atomic Coordinates",
                 fontsize=22)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(outdir / "rmse_bar_chart.png", dpi=300)
    plt.close(fig)
    click.echo("  ✓ rmse_bar_chart.png")


def plot_match_rate_bar_chart(df: pd.DataFrame, outdir: Path) -> None:
    mr_col = "RMSE.AtomGen.match_rate"
    if mr_col not in df.columns:
        click.echo("  ⚠ No match_rate column — skipping match rate chart", err=True)
        return
    plot_df = df[["benchmark_name", mr_col]].rename(columns={mr_col: "match_rate"}).copy()
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = np.arange(len(plot_df))
    ax.bar(pos, plot_df["match_rate"].astype(float).tolist(),
           width=0.55, edgecolor="k", linewidth=0.8, color=WVU_BLUE)
    ax.set_xticks(pos)
    ax.set_xticklabels(plot_df["benchmark_name"].tolist(), rotation=30, ha="right", fontsize=13)
    ax.set_ylabel("Match Rate", fontsize=16)
    ax.set_title("Structure Matcher Match Rate (STOL=0.5)", fontsize=22)
    ax.set_ylim(0, 1)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(outdir / "match_rate_bar_chart.png", dpi=300)
    plt.close(fig)
    click.echo("  ✓ match_rate_bar_chart.png")


def plot_crystal_system_mae_charts(metrics_dicts: List[dict], outdir: Path, kmin: int) -> None:
    pooled_sum:   Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    pooled_count: Dict[str, int]              = defaultdict(int)

    for m in metrics_dicts:
        for entry in m.get("crystal_system_mae", {}).get("by_system", []):
            cs, n = entry["crystal_system"], int(entry["n_reconstructions"])
            for param, val in entry["mae"].items():
                if np.isfinite(val):
                    pooled_sum[cs][param] += float(val) * n
            pooled_count[cs] += n

    systems = [cs for cs in CRYSYS_PLOT_ORDER if pooled_count.get(cs, 0) >= kmin]
    if not systems:
        click.echo("  ⚠ No crystal systems pass kmin — skipping crystal-system MAE charts",
                   err=True)
        return

    data, counts_arr = {}, []
    for cs in systems:
        n      = pooled_count[cs]
        data[cs] = {p: pooled_sum[cs][p] / n for p in PARAMS_ALL}
        counts_arr.append(n)

    g       = pd.DataFrame(data).T[list(PARAMS_ALL)]
    g.index = [s.capitalize() for s in systems]
    plot_df = g.rename(columns=AX_LABEL_MAP)
    counts_arr = np.array(counts_arr, dtype=int)

    def _add_counts(ax, counts, tops):
        current_ylim = ax.get_ylim()
        ymax = max(current_ylim[1], float(np.max(tops)) * 1.22 if len(tops) else current_ylim[1])
        ax.set_ylim(0, ymax)
        off = 0.02 * ax.get_ylim()[1]
        for i, (n, top) in enumerate(zip(counts, tops)):
            ax.text(i, float(top) + off, f"n={int(n)}", ha="center", va="bottom", fontsize=12)

    def _style_crysys(ax, ylabel, title):
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(title, fontsize=22)
        ax.legend(title="Lattice Parameter", title_fontsize=15, fontsize=15,
                  loc="center left", bbox_to_anchor=(0.02, 0.66), borderaxespad=0.0)
        plt.xticks(rotation=30, ha="right", fontsize=13)
        plt.yticks(fontsize=15)

    length_cols = [AX_LABEL_MAP[k] for k in ("a", "b", "c")]
    angle_cols  = [AX_LABEL_MAP[k] for k in ("alpha", "beta", "gamma")]

    len_tops = g[["a", "b", "c"]].max(axis=1).to_numpy(dtype=float)
    fig, ax  = plt.subplots(figsize=(10, 8))
    plot_df[length_cols].plot(kind="bar", edgecolor="k", ax=ax, color=LEN_GRANITE)
    _style_crysys(ax, "Mean Absolute Error (Å)",
                  "Mean Absolute Error by Crystal System\n"
                  "Results Pooled from All Benchmarks\nLattice Lengths (Å)")
    _add_counts(ax, counts_arr, len_tops)
    fig.tight_layout()
    plt.savefig(outdir / "crystal_system_mae_bar_chart_abc.png", dpi=500, bbox_inches="tight")
    plt.close(fig)
    click.echo("  ✓ crystal_system_mae_bar_chart_abc.png")

    ang_tops = g[["alpha", "beta", "gamma"]].max(axis=1).to_numpy(dtype=float)
    fig, ax  = plt.subplots(figsize=(10, 8))
    plot_df[angle_cols].plot(kind="bar", edgecolor="k", ax=ax, color=ANG_GRANITE)
    _style_crysys(ax, "Mean Absolute Error (°)",
                  "Mean Absolute Error by Crystal System\n"
                  "Results Pooled from All Benchmarks\nLattice Angles (°)")
    _add_counts(ax, counts_arr, ang_tops)
    fig.tight_layout()
    plt.savefig(outdir / "crystal_system_mae_bar_chart_angles.png", dpi=500, bbox_inches="tight")
    plt.close(fig)
    click.echo("  ✓ crystal_system_mae_bar_chart_angles.png")


# ── CLI ────────────────────────────────────────────────────────────────────────
@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--outdir", "-o", default="atombench_output", show_default=True, type=Path,
              help="Output directory for metrics JSON files and plot PNGs.")
@click.option("--name", "-n", default=None,
              help="Override the benchmark name (only meaningful for a single CSV input).")
@click.option("--tau", default=0.5, show_default=True, type=float,
              help="ccRMSE clamp threshold.")
@click.option("--amd-k", default=100, show_default=True, type=int,
              help="AMD vector length k.")
@click.option("--symprec", default=0.1, show_default=True, type=float,
              help="Symmetry tolerance for SpacegroupAnalyzer (Å).")
@click.option("--kmin", default=10, show_default=True, type=int,
              help="Minimum structures per crystal system for the crystal-system MAE charts.")
@click.option("--skip-metrics", is_flag=True,
              help="Re-use an existing metrics JSON if present; skip recomputation.")
@click.option("--metrics-only", is_flag=True,
              help="Compute metrics only; do not generate any plots.")
def main(
    path: Path,
    outdir: Path,
    name: Optional[str],
    tau: float,
    amd_k: int,
    symprec: float,
    kmin: int,
    skip_metrics: bool,
    metrics_only: bool,
) -> None:
    """Compute metrics and generate plots for crystal structure reconstruction benchmarks.

    \b
    PATH can be:
      • a single benchmark CSV file  (columns: id, target, prediction)
      • a directory of CSV files     (one benchmark per CSV)
      • a directory of subdirectories (one benchmark per subdirectory containing a CSV)

    Metrics are written to OUTDIR as <bench_name>_metrics.json.
    Plots are written to OUTDIR as PNG files.
    """
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Discover benchmarks ───────────────────────────────────────────────────
    benchmarks = discover_benchmarks(path.resolve())
    if not benchmarks:
        raise click.ClickException(f"No benchmark CSV files found at: {path}")

    if name is not None and len(benchmarks) == 1:
        benchmarks = [(name, benchmarks[0][1])]

    click.echo(f"Found {len(benchmarks)} benchmark(s).")

    # ── Compute (or load) metrics ─────────────────────────────────────────────
    all_results: List[Tuple[str, Path, dict]] = []   # (name, csv_path, metrics)

    for bench_name, csv_path in benchmarks:
        click.echo(f"\n── {bench_name}")
        metrics_path = outdir / f"{bench_name}_metrics.json"

        if skip_metrics and metrics_path.is_file():
            click.echo(f"  ← {metrics_path.name} (existing)")
            with metrics_path.open() as fh:
                metrics = json.load(fh)
        else:
            df = pd.read_csv(csv_path)
            df = df.rename(columns={c: c.strip().lower() for c in df.columns})
            missing = [c for c in ("target", "prediction") if c not in df.columns]
            if missing:
                click.echo(f"  ⚠ missing columns {missing} — skipping", err=True)
                continue
            click.echo(f"  {len(df)} rows from {csv_path.name}")
            metrics = compute_metrics(df, bench_name,
                                      tau=tau, amd_k=amd_k, symprec=symprec, kmin=kmin)
            with metrics_path.open("w") as fh:
                json.dump(metrics, fh, indent=2)
            click.echo(f"  ✓ {metrics_path.name}")

        all_results.append((bench_name, csv_path, metrics))

    if not all_results:
        raise click.ClickException("No benchmarks were successfully processed.")

    if metrics_only:
        click.echo(f"\nDone (metrics only). Output: {outdir}")
        return

    # ── Generate plots ────────────────────────────────────────────────────────
    click.echo("\n── Plots")

    # Distribution plots — one per benchmark
    for bench_name, csv_path, _ in all_results:
        try:
            plot_distribution(bench_name, csv_path, outdir)
        except Exception as e:
            click.echo(f"  ⚠ {bench_name}: distribution plot failed — {e}", err=True)

    # Aggregate bar charts — all benchmarks on one figure each
    all_metrics = [m for _, _, m in all_results]
    rows = [pd.json_normalize(m, sep=".", max_level=3).iloc[0].to_dict() for m in all_metrics]
    df_metrics = pd.DataFrame(rows)

    plot_kld_bar_chart(df_metrics, outdir)
    plot_mae_abc_bar_chart(df_metrics, outdir)
    plot_mae_angles_bar_chart(df_metrics, outdir)
    plot_rmse_bar_chart(df_metrics, outdir)
    plot_match_rate_bar_chart(df_metrics, outdir)

    # Crystal-system MAE charts
    if any("crystal_system_mae" in m for m in all_metrics):
        try:
            plot_crystal_system_mae_charts(all_metrics, outdir, kmin=kmin)
        except Exception as e:
            click.echo(f"  ⚠ crystal-system MAE charts failed — {e}", err=True)
    else:
        click.echo("  ⚠ no crystal_system_mae data — skipping crystal-system charts", err=True)

    click.echo(f"\nAll done. Output: {outdir}")


if __name__ == "__main__":
    main()

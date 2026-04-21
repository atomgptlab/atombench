#!/usr/bin/env python3
"""
generate_plots.py — generate all reconstruction benchmark plots.

Input : --root <job_runs_dir>   (subdirs each containing a CSV + metrics.json)
        [--config <groups.json>] (optional; required only for N×3 grid charts)
Output: PNGs written to --outdir (default: ./figures)

The script is dataset-agnostic.  No dataset names, model names, or bin values
are hardcoded.  All implementation-specific parameters come from the optional
config file or CLI arguments.  Bar charts, distribution plots, and crystal-
system MAE charts run without a config.  Grid charts require a config.

Config JSON schema:
{
  "display_names": { "dir_name": "Pretty Label", ... },
  "groups": [
    {
      "name": "GroupName",
      "title": "Suptitle for grid chart",
      "benchmarks": { "Row Label": "benchmark_dir_name", ... },
      "bins": {
        "a_c_width": 0.10, "a_c_min": 2.0,  "a_c_max": 10.0,
        "gamma_width": 8.0, "gamma_min": 30.0, "gamma_max": 140.0
      }
    }
  ]
}
"""
from __future__ import annotations

import argparse
import csv as csv_mod
import json
import os
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({
    "font.family": "serif",
    "axes.linewidth": 0.8,
    "patch.linewidth": 0.0,
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif", "STIX"],
})

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ── Visual constants ──────────────────────────────────────────────────────────
WVU_BLUE = "#002855"   # target colour in grid charts
WVU_GOLD = "#EEAA00"   # predicted colour in grid charts
PARAMS       = ["a", "c", "gamma"]
PARAM_LABEL  = {"a": r"$a$ (Å)", "c": r"$c$ (Å)", "gamma": r"$\gamma$ (°)"}
PANEL_LETTERS = [f"({chr(97+i)})" for i in range(26)]

LEN_GRANITE = ["#4A6272", "#89A9BC", "#D6E3EC"]
ANG_GRANITE = ["#6A5560", "#B08A97", "#E7D6DC"]
PNG_DPI_CRYSYS = 500

CRYSYS_PLOT_ORDER = [
    "cubic", "hexagonal", "trigonal",
    "tetragonal", "orthorhombic", "monoclinic",
]

AX_LABEL_MAP = {
    "a": r"$a$", "b": r"$b$", "c": r"$c$",
    "alpha": r"$\alpha$", "beta": r"$\beta$", "gamma": r"$\gamma$",
}

# ── Niggli / pymatgen helpers ─────────────────────────────────────────────────
@lru_cache(maxsize=20000)
def reduced_structure_from_poscar_text(poscar_text: str) -> Structure:
    s = Structure.from_str(poscar_text.replace("\\n", "\n"), fmt="poscar")
    s = s.get_primitive_structure()
    s = s.get_reduced_structure(reduction_algo="niggli")
    return s


@lru_cache(maxsize=20000)
def niggli_params_from_poscar_text(poscar_text: str):
    s = reduced_structure_from_poscar_text(poscar_text)
    a, b, c = s.lattice.abc
    alpha, beta, gamma = s.lattice.angles
    return float(a), float(b), float(c), float(alpha), float(beta), float(gamma)


# ── Discovery helpers ─────────────────────────────────────────────────────────
def find_benchmark_csv(dir_path: Path) -> Optional[Path]:
    """Return the newest CSV under dir_path that has id/target/prediction columns."""
    latest: Optional[Tuple[float, Path]] = None
    for p, _, files in os.walk(dir_path):
        for f in files:
            if not f.lower().endswith(".csv"):
                continue
            path = Path(p) / f
            try:
                with path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
                    reader = csv_mod.DictReader(fh)
                    fields = {fn.strip().lower() for fn in (reader.fieldnames or [])}
                    if {"id", "target", "prediction"}.issubset(fields):
                        mt = path.stat().st_mtime
                        if latest is None or mt > latest[0]:
                            latest = (mt, path)
            except Exception:
                continue
    return latest[1] if latest else None


def discover_benchmarks(root: Path) -> List[Tuple[str, Path, Optional[Path], Optional[dict]]]:
    """
    Return list of (bench_name, bench_dir, csv_path_or_None, metrics_dict_or_None)
    for each subdirectory of root.
    """
    results = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        csv_path = find_benchmark_csv(entry)
        metrics  = None
        mfp      = entry / "metrics.json"
        if mfp.is_file():
            try:
                with mfp.open() as fh:
                    metrics = json.load(fh)
            except Exception as e:
                print(f"⚠️  {entry.name}: could not load metrics.json — {e}", file=sys.stderr)
        results.append((entry.name, entry, csv_path, metrics))
    return results


# ── Distribution plot (2×3 grid per benchmark) ───────────────────────────────
def _overlay_hist(ax, x, y, bins, xlabel, title):
    w_x = np.ones_like(x, dtype=float) / max(1, len(x)) * 100
    w_y = np.ones_like(y, dtype=float) / max(1, len(y)) * 100
    ax.hist(x, bins=bins, weights=w_x, alpha=0.6, color="tab:blue", label="target")
    ax.hist(y, bins=bins, weights=w_y, alpha=0.6, color="plum",     label="predicted")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return ax


def plot_distribution(bench_name: str, csv_path: Path, outdir: Path) -> None:
    plt.rcParams.update({"font.size": 18})
    df = pd.read_csv(csv_path)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    x_a, x_b, x_c, x_alpha, x_beta, x_gamma = [], [], [], [], [], []
    y_a, y_b, y_c, y_alpha, y_beta, y_gamma = [], [], [], [], [], []

    for _, row in df.iterrows():
        try:
            ta, tb, tc, tal, tbe, tga = niggli_params_from_poscar_text(str(row["target"]))
            pa, pb, pc, pal, pbe, pga = niggli_params_from_poscar_text(str(row["prediction"]))
            x_a.append(ta);      y_a.append(pa)
            x_b.append(tb);      y_b.append(pb)
            x_c.append(tc);      y_c.append(pc)
            x_alpha.append(tal); y_alpha.append(pal)
            x_beta.append(tbe);  y_beta.append(pbe)
            x_gamma.append(tga); y_gamma.append(pga)
        except Exception:
            continue

    fig = plt.figure(figsize=(14, 8))
    grid = GridSpec(2, 3)

    _overlay_hist(plt.subplot(grid[0, 0]),
                  x_a, y_a, np.arange(2, 7, 0.1),
                  r"a ($\AA$)", "(a)").set_ylabel("Materials dist.")
    plt.legend()
    _overlay_hist(plt.subplot(grid[0, 1]),
                  x_c, y_c, np.arange(2, 7, 0.1), r"c ($\AA$)", "(b)")
    _overlay_hist(plt.subplot(grid[0, 2]),
                  x_gamma, y_gamma, np.arange(30, 150, 10),
                  r"$\gamma$ ($^\circ$)", "(c)")

    # additional per-structure properties
    x_spg, y_spg, x_Z, y_Z, x_lat, y_lat = [], [], [], [], [], []
    lat_order = ["triclinic", "monoclinic", "orthorhombic",
                 "tetragonal", "trigonal", "hexagonal", "cubic"]
    lat_to_idx = {name: i for i, name in enumerate(lat_order)}

    for _, row in df.iterrows():
        try:
            s_t = reduced_structure_from_poscar_text(str(row["target"]))
            s_p = reduced_structure_from_poscar_text(str(row["prediction"]))
            x_Z.append(s_t.composition.weight)
            y_Z.append(s_p.composition.weight)
            sga_t = SpacegroupAnalyzer(s_t, symprec=0.1)
            sga_p = SpacegroupAnalyzer(s_p, symprec=0.1)
            x_spg.append(sga_t.get_space_group_number())
            y_spg.append(sga_p.get_space_group_number())
            x_lat.append(sga_t.get_crystal_system())
            y_lat.append(sga_p.get_crystal_system())
        except Exception:
            continue

    _overlay_hist(plt.subplot(grid[1, 0]),
                  x_spg, y_spg, np.arange(1, 231, 10),
                  "Spacegroup number", "(d)").set_ylabel("Materials dist.")

    # crystal system counts
    valid_lat = [(lx, ly) for lx, ly in zip(x_lat, y_lat) if lx and ly]
    if valid_lat:
        xl, yl = zip(*valid_lat)
    else:
        xl, yl = [], []
    xl_counts = np.bincount([lat_to_idx[l] for l in xl], minlength=len(lat_order))
    yl_counts = np.bincount([lat_to_idx[l] for l in yl], minlength=len(lat_order))
    ax_lat = plt.subplot(grid[1, 1])
    pos = np.arange(len(lat_order))
    bar_w = 0.4
    ax_lat.bar(pos, xl_counts, width=bar_w, alpha=0.6, label="target",    color="tab:blue")
    ax_lat.bar(pos, yl_counts, width=bar_w, alpha=0.6, label="predicted", color="plum")
    ax_lat.set_xticks(pos)
    ax_lat.set_xticklabels((pos + 1).tolist(), rotation=0, ha="center")
    ax_lat.set_xlabel("Crystal system number")
    ax_lat.set_title("(e)")

    _overlay_hist(plt.subplot(grid[1, 2]),
                  x_Z, y_Z, np.arange(15, 2000, 100), "Weight (AMU)", "(f)")

    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.suptitle(bench_name, fontsize=30)

    out_png = outdir / f"{bench_name}_distribution.png"
    plt.savefig(out_png, format="png")
    plt.close(fig)
    print(f"✓ {out_png.name}")


# ── Bar chart helpers ─────────────────────────────────────────────────────────
def _style_bar_axes(ax, ylabel: str, title: str):
    ax.set_xlabel("", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=22)
    ax.legend(title="Lattice Parameter", title_fontsize=15, fontsize=15)
    plt.xticks(rotation=30, ha="right", fontsize=13)
    plt.yticks(fontsize=15)
    plt.tight_layout()


def collect_metrics_rows(root: Path, display_names: dict) -> pd.DataFrame:
    rows = []
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        mfp = subdir / "metrics.json"
        if not mfp.is_file():
            print(f"⚠️  no metrics.json in {subdir.name} — skipped for bar charts", file=sys.stderr)
            continue
        with mfp.open() as fh:
            rec = json.load(fh)
        rec.setdefault("benchmark_name", subdir.name)
        # Apply display name mapping
        rec["benchmark_name"] = display_names.get(rec["benchmark_name"], rec["benchmark_name"])
        rows.append(pd.json_normalize(rec, sep=".", max_level=3).iloc[0].to_dict())
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def plot_kld_bar_chart(df: pd.DataFrame, outdir: Path) -> None:
    kld_cols = [f"KLD.{k}" for k in AX_LABEL_MAP]
    if any(c not in df.columns for c in kld_cols):
        print("⚠️  Missing KLD columns — skipping KLD bar chart", file=sys.stderr)
        return
    kld_df = (df.set_index("benchmark_name")[kld_cols]
                .rename(columns=lambda c: AX_LABEL_MAP[c.split(".")[-1]]))
    fig, ax = plt.subplots(figsize=(10, 8))
    kld_df.plot(kind="bar", edgecolor="k", ax=ax)
    _style_bar_axes(ax, "KL Divergence (Nats)",
                    "KL Divergence of Predicted vs. Target\nLattice-Parameter Distributions")
    plt.savefig(outdir / "comparison_bar_chart.png", dpi=300)
    plt.close(fig)
    print("✓ comparison_bar_chart.png")


def plot_mae_abc_bar_chart(df: pd.DataFrame, outdir: Path) -> None:
    for cand in ([f"MAE.average_mae.{k}" for k in AX_LABEL_MAP],
                 [f"MAE.{k}" for k in AX_LABEL_MAP]):
        if all(c in df.columns for c in cand):
            mae_cols = cand
            break
    else:
        print("⚠️  Missing MAE columns — skipping MAE bar charts", file=sys.stderr)
        return

    mae_df = (df.set_index("benchmark_name")[mae_cols]
                .rename(columns=lambda c: AX_LABEL_MAP[c.split(".")[-1]]))
    length_cols = [AX_LABEL_MAP[k] for k in ("a", "b", "c")]

    fig, ax = plt.subplots(figsize=(10, 8))
    mae_df[length_cols].plot(kind="bar", edgecolor="k", ax=ax)
    _style_bar_axes(ax, "Mean Absolute Error (Å)",
                    "Mean Absolute Error – Lattice Lengths (Å)")
    plt.savefig(outdir / "mae_bar_chart_abc.png", dpi=300)
    plt.close(fig)
    print("✓ mae_bar_chart_abc.png")

    return mae_df  # used by angles chart to share the same df


def plot_mae_angles_bar_chart(df: pd.DataFrame, outdir: Path) -> None:
    for cand in ([f"MAE.average_mae.{k}" for k in AX_LABEL_MAP],
                 [f"MAE.{k}" for k in AX_LABEL_MAP]):
        if all(c in df.columns for c in cand):
            mae_cols = cand
            break
    else:
        return  # already warned in abc chart

    mae_df = (df.set_index("benchmark_name")[mae_cols]
                .rename(columns=lambda c: AX_LABEL_MAP[c.split(".")[-1]]))
    angle_cols = [AX_LABEL_MAP[k] for k in ("alpha", "beta", "gamma")]

    fig, ax = plt.subplots(figsize=(10, 8))
    mae_df[angle_cols].plot(kind="bar", edgecolor="k",
                            color=["red", "purple", "brown"], ax=ax)
    _style_bar_axes(ax, "Mean Absolute Error (°)",
                    "Mean Absolute Error – Lattice Angles (°)")
    plt.savefig(outdir / "mae_bar_chart_angles.png", dpi=300)
    plt.close(fig)
    print("✓ mae_bar_chart_angles.png")


def plot_rmse_bar_chart(df: pd.DataFrame, outdir: Path, display_names: dict) -> None:
    rmse_candidates = [
        "RMSE.AtomGen.mean_cartesian_rms_angstrom",
        "RMSE.AtomGen.mean_normalized_cartesian_rms",
        "RMSE.AtomGen",
        "RMSE",
    ]
    for cand in rmse_candidates:
        if cand in df.columns:
            rmse_col = cand
            break
    else:
        print("⚠️  No RMSE column found — skipping RMSE bar chart", file=sys.stderr)
        return

    plot_df = (
        df[["benchmark_name", rmse_col]]
          .rename(columns={rmse_col: "RMSE"})
          .copy()
    )

    # Assign colours from tab10 palette cycling over unique benchmark names
    unique_names = list(dict.fromkeys(plot_df["benchmark_name"]))
    palette = plt.cm.tab10.colors
    color_map = {name: palette[i % len(palette)] for i, name in enumerate(unique_names)}
    plot_df["color"] = plot_df["benchmark_name"].map(color_map)

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = np.arange(len(plot_df))
    ax.bar(pos, plot_df["RMSE"].astype(float).tolist(),
           width=0.55, edgecolor="k", linewidth=0.8,
           color=plot_df["color"].tolist())
    ax.set_xticks(pos)
    ax.set_xticklabels(plot_df["benchmark_name"].tolist(), rotation=30, ha="right", fontsize=13)
    handles = [mpatches.Patch(color=color_map[n], label=n) for n in unique_names]
    ax.legend(handles=handles, title_fontsize=15, fontsize=15)
    ax.set_xlabel("", fontsize=16)
    ax.set_ylabel("Average RMSE (Å)", fontsize=16)
    ax.set_title("Average Root Mean Squared Error\nfor Predicted vs. Target Atomic Coordinates",
                 fontsize=22)
    plt.xticks(rotation=30, ha="right", fontsize=13)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(outdir / "rmse_bar_chart.png", dpi=300)
    plt.close(fig)
    print("✓ rmse_bar_chart.png")


# ── N×3 reconstruction grid ───────────────────────────────────────────────────
def _unescape_poscar(s: str) -> str:
    return (s.replace("\r\n", "\n").replace("\r", "\n")
             .replace("\\n", "\n").replace("\\t", " ").strip())


def _extract_series(csv_path: Path) -> Dict[str, Dict[str, List[float]]]:
    out = {"target": {k: [] for k in PARAMS}, "predicted": {k: [] for k in PARAMS}}
    with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv_mod.DictReader(fh)
        for row in reader:
            try:
                t_str = _unescape_poscar(row["target"])
                p_str = _unescape_poscar(row["prediction"])
                ta, tb, tc, tal, tbe, tga = niggli_params_from_poscar_text(t_str)
                pa, pb, pc, pal, pbe, pga = niggli_params_from_poscar_text(p_str)
                out["target"]["a"].append(float(ta))
                out["target"]["c"].append(float(tc))
                out["target"]["gamma"].append(float(tga))
                out["predicted"]["a"].append(float(pa))
                out["predicted"]["c"].append(float(pc))
                out["predicted"]["gamma"].append(float(pga))
            except Exception:
                continue
    return out


def _load_klds(metrics_path: Path) -> Dict[str, float]:
    try:
        data = json.loads(metrics_path.read_text())
        k = data.get("KLD", {})
        return {"a": float(k.get("a")), "c": float(k.get("c")), "gamma": float(k.get("gamma"))}
    except Exception:
        return {}


def _weights_percent(n: int) -> np.ndarray:
    return np.ones(n, dtype=float) * (100.0 / n) if n > 0 else np.array([])


def _style_grid_axes(ax, left_col: bool, bottom_row: bool) -> None:
    ax.tick_params(axis="both", which="major", labelsize=12, width=1.4, length=7)
    ax.tick_params(axis="y", which="both", left=False)
    ax.minorticks_off()
    if not left_col:
        ax.set_yticklabels([])
    if not bottom_row:
        ax.set_xlabel("")


def _annotate_kld(ax, value: Optional[float]) -> None:
    if value is None:
        return
    ax.text(0.97, 0.92, f"KLD = {value:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=0.8, alpha=0.95))


def _annotate_panel_label(ax, label: str) -> None:
    ax.text(0.03, 0.92, label, transform=ax.transAxes,
            ha="left", va="top", fontsize=9)


def _slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


def plot_group_grid(group_cfg: dict, root: Path, outdir: Path) -> None:
    benchmarks: Dict[str, str] = group_cfg["benchmarks"]  # label -> dir_name
    title: str = group_cfg.get("title", group_cfg["name"])
    bins_cfg   = group_cfg.get("bins", {})

    bins_a_c  = np.arange(
        float(bins_cfg.get("a_c_min", 2.0)),
        float(bins_cfg.get("a_c_max", 10.0)) + 1e-9,
        float(bins_cfg.get("a_c_width", 0.20)),
    )
    bins_gamma = np.arange(
        float(bins_cfg.get("gamma_min", 30.0)),
        float(bins_cfg.get("gamma_max", 140.0)) + 1e-9,
        float(bins_cfg.get("gamma_width", 8.0)),
    )

    n_rows = len(benchmarks)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for ax in axes.ravel():
        ax.set_ylabel("")
        ax.tick_params(axis="y", which="both",
                       left=False, right=False,
                       labelleft=False, labelright=False)

    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.12, top=0.86,
                        wspace=0.10, hspace=0.30)

    for r, (row_label, dir_name) in enumerate(benchmarks.items()):
        bench_dir = root / dir_name
        series, klds = None, {}
        if bench_dir.is_dir():
            csv_path = find_benchmark_csv(bench_dir)
            if csv_path:
                series = _extract_series(csv_path)
            mfp = bench_dir / "metrics.json"
            if mfp.exists():
                klds = _load_klds(mfp)
        else:
            print(f"⚠️  {dir_name}: directory not found — grid row will show 'no data'",
                  file=sys.stderr)

        for c, param in enumerate(PARAMS):
            ax = axes[r, c]
            _annotate_panel_label(ax, PANEL_LETTERS[r * 3 + c])
            if c == 0:
                ax.set_ylabel(row_label, fontsize=22)

            if series is None or not series["target"][param]:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        fontsize=12, alpha=0.7)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            xt = np.asarray(series["target"][param], dtype=float)
            xp = np.asarray(series["predicted"][param], dtype=float)
            bins = bins_a_c if param in ("a", "c") else bins_gamma

            ax.hist(xt, bins=bins, weights=_weights_percent(len(xt)),
                    histtype="stepfilled", alpha=0.68, color=WVU_BLUE,
                    edgecolor="none", label="target")
            ax.hist(xp, bins=bins, weights=_weights_percent(len(xp)),
                    histtype="stepfilled", alpha=0.68, color=WVU_GOLD,
                    edgecolor="none", label="predicted")

            if r == n_rows - 1:
                ax.set_xlabel(PARAM_LABEL[param], fontsize=14)

            _style_grid_axes(ax, left_col=(c == 0), bottom_row=(r == n_rows - 1))
            _annotate_kld(ax, klds.get(param))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        leg = axes[0, 0].legend(handles, labels, loc="center right", frameon=True, fontsize=12)
        leg.get_frame().set_alpha(0.95)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_linewidth(0.6)

    fig.suptitle(title, fontsize=28, y=0.93)
    fig.text(0.91, 0.5, "Materials Percentage (%)",
             rotation=270, va="center", ha="center", fontsize=20)

    out_png = outdir / f"{_slugify(group_cfg['name'])}_reconstruction_grid.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out_png.name}")


# ── Crystal-system MAE charts from pre-computed JSON ─────────────────────────
def _style_crysys_axes(ax, ylabel: str, title: str):
    ax.set_xlabel("", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=22)
    ax.legend(
        title="Lattice Parameter", title_fontsize=15, fontsize=15,
        loc="center left", bbox_to_anchor=(0.02, 0.66), borderaxespad=0.0,
    )
    plt.xticks(rotation=30, ha="right", fontsize=13)
    plt.yticks(fontsize=15)


def _add_group_counts(ax, counts, group_tops, labels, fontsize=12):
    if len(counts) == 0:
        return
    current_ylim = ax.get_ylim()
    ymax_needed  = float(np.max(group_tops)) if len(group_tops) else current_ylim[1]
    new_ymax     = max(current_ylim[1], ymax_needed * 1.22 if ymax_needed > 0 else 1.0)
    ax.set_ylim(0.0, new_ymax)
    offset_y = 0.02 * ax.get_ylim()[1]
    for i, (n, top) in enumerate(zip(counts, group_tops)):
        ax.text(i, float(top) + offset_y, f"n={int(n)}",
                ha="center", va="bottom", fontsize=fontsize)


def plot_crystal_system_mae_from_json(
    metrics_dicts: List[dict], outdir: Path, kmin: int
) -> None:
    # Pool per-crystal-system MAE via weighted mean across benchmarks
    pooled_sum:   Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    pooled_count: Dict[str, int] = defaultdict(int)

    for m in metrics_dicts:
        by_system = m.get("crystal_system_mae", {}).get("by_system", [])
        for entry in by_system:
            cs = entry["crystal_system"]
            n  = int(entry["n_reconstructions"])
            for param, val in entry["mae"].items():
                if np.isfinite(val):
                    pooled_sum[cs][param] += float(val) * n
            pooled_count[cs] += n

    # Build pooled means in plot order, respecting kmin
    systems = [cs for cs in CRYSYS_PLOT_ORDER if pooled_count.get(cs, 0) >= kmin]
    if not systems:
        print("⚠️  No crystal systems pass kmin — skipping crystal-system MAE charts",
              file=sys.stderr)
        return

    PARAMS_LAT = ("a", "b", "c", "alpha", "beta", "gamma")
    data = {}
    counts_arr = []
    for cs in systems:
        n = pooled_count[cs]
        data[cs] = {p: pooled_sum[cs][p] / n for p in PARAMS_LAT}
        counts_arr.append(n)

    g = pd.DataFrame(data).T[list(PARAMS_LAT)]  # shape: (n_systems, 6)
    g.index = [s.capitalize() for s in systems]
    plot_df = g.rename(columns=AX_LABEL_MAP)

    length_cols = [AX_LABEL_MAP[k] for k in ("a", "b", "c")]
    angle_cols  = [AX_LABEL_MAP[k] for k in ("alpha", "beta", "gamma")]
    labels      = list(plot_df.index)
    counts_arr  = np.array(counts_arr, dtype=int)

    title_len = (
        "Mean Absolute Error by Crystal System\n"
        "Results Pooled from All Benchmarks\n"
        "Lattice Lengths (Å)"
    )
    title_ang = (
        "Mean Absolute Error by Crystal System\n"
        "Results Pooled from All Benchmarks\n"
        "Lattice Angles (°)"
    )

    # Lengths chart
    length_tops = g[["a", "b", "c"]].max(axis=1).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_df[length_cols].plot(kind="bar", edgecolor="k", ax=ax, color=LEN_GRANITE)
    _style_crysys_axes(ax, "Mean Absolute Error (Å)", title_len)
    _add_group_counts(ax, counts_arr, length_tops, labels)
    fig.tight_layout()
    plt.savefig(outdir / "crystal_system_mae_bar_chart_abc.png",
                dpi=PNG_DPI_CRYSYS, bbox_inches="tight")
    plt.close(fig)
    print("✓ crystal_system_mae_bar_chart_abc.png")

    # Angles chart
    angle_tops = g[["alpha", "beta", "gamma"]].max(axis=1).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_df[angle_cols].plot(kind="bar", edgecolor="k", ax=ax, color=ANG_GRANITE)
    _style_crysys_axes(ax, "Mean Absolute Error (°)", title_ang)
    _add_group_counts(ax, counts_arr, angle_tops, labels)
    fig.tight_layout()
    plt.savefig(outdir / "crystal_system_mae_bar_chart_angles.png",
                dpi=PNG_DPI_CRYSYS, bbox_inches="tight")
    plt.close(fig)
    print("✓ crystal_system_mae_bar_chart_angles.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Generate all reconstruction benchmark plots from metrics.json + CSV files."
    )
    ap.add_argument("--root",         required=True, type=Path,
                    help="Directory containing benchmark subdirs (each with CSV + metrics.json)")
    ap.add_argument("--outdir",       default="figures", type=Path,
                    help="Output directory for PNGs (default: ./figures)")
    ap.add_argument("--config",       default=None,
                    help="Optional JSON config for grid charts and display names")
    ap.add_argument("--kmin",         default=10, type=int,
                    help="Min pooled structures per crystal system for crystal-system MAE charts")
    args = ap.parse_args()

    root   = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()

    if not root.is_dir():
        sys.exit(f"ERROR: --root does not exist: {root}")
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load config ───────────────────────────────────────────────────────────
    display_names: dict = {}
    groups: list = []
    if args.config:
        config_path = Path(args.config).resolve()
        if not config_path.is_file():
            sys.exit(f"ERROR: --config file not found: {config_path}")
        with config_path.open() as fh:
            cfg = json.load(fh)
        display_names = cfg.get("display_names", {})
        groups        = cfg.get("groups", [])

    # ── Discover benchmarks ───────────────────────────────────────────────────
    benchmarks = discover_benchmarks(root)
    if not benchmarks:
        sys.exit(f"ERROR: no subdirectories found in {root}")

    bench_data   = []   # (name, bench_dir, csv_path, metrics_dict)
    metrics_rows = []   # flat dicts for bar charts (with display names applied)
    metrics_dicts = []  # raw metrics dicts for crystal-system MAE pooling

    for name, bench_dir, csv_path, metrics in benchmarks:
        if csv_path is None:
            print(f"⚠️  {name}: no CSV found — distribution + grid plots skipped",
                  file=sys.stderr)
        if metrics is None:
            print(f"⚠️  {name}: no metrics.json — bar charts skipped for this benchmark",
                  file=sys.stderr)
        bench_data.append((name, bench_dir, csv_path, metrics))

        if metrics is not None:
            m_copy = dict(metrics)
            m_copy["benchmark_name"] = display_names.get(
                m_copy.get("benchmark_name", name), m_copy.get("benchmark_name", name)
            )
            metrics_rows.append(
                pd.json_normalize(m_copy, sep=".", max_level=3).iloc[0].to_dict()
            )
            metrics_dicts.append(metrics)

    # ── Plot 1: per-benchmark distribution grids ──────────────────────────────
    print("\n── Distribution plots ──────────────────────────────────────────")
    for name, bench_dir, csv_path, _ in bench_data:
        if csv_path is None:
            continue
        try:
            plot_distribution(name, csv_path, outdir)
        except Exception as e:
            print(f"⚠️  {name}: distribution plot failed — {e}", file=sys.stderr)

    # ── Plots 2–5: bar charts (need metrics.json) ─────────────────────────────
    if not metrics_rows:
        print("\n⚠️  No metrics.json files found — skipping all bar charts", file=sys.stderr)
    else:
        df = pd.DataFrame(metrics_rows)
        print("\n── Bar charts ──────────────────────────────────────────────────")
        plot_kld_bar_chart(df, outdir)
        plot_mae_abc_bar_chart(df, outdir)
        plot_mae_angles_bar_chart(df, outdir)
        plot_rmse_bar_chart(df, outdir, display_names)

    # ── Plots 6+: reconstruction grids (need config groups) ───────────────────
    if not groups:
        print("\nℹ️  No groups in config — skipping reconstruction grid charts", file=sys.stderr)
    else:
        print("\n── Reconstruction grids ────────────────────────────────────────")
        for group_cfg in groups:
            try:
                plot_group_grid(group_cfg, root, outdir)
            except Exception as e:
                print(f"⚠️  Group '{group_cfg.get('name')}': grid failed — {e}",
                      file=sys.stderr)

    # ── Crystal-system MAE charts ─────────────────────────────────────────────
    if not metrics_dicts:
        print("\n⚠️  No metrics dicts — skipping crystal-system MAE charts", file=sys.stderr)
    else:
        has_crysys = any("crystal_system_mae" in m for m in metrics_dicts)
        if not has_crysys:
            print(
                "\nℹ️  No crystal_system_mae key in any metrics.json "
                "(run compute_metrics.py to generate it) — skipping",
                file=sys.stderr,
            )
        else:
            print("\n── Crystal-system MAE charts ───────────────────────────────────")
            plot_crystal_system_mae_from_json(metrics_dicts, outdir, kmin=args.kmin)

    print(f"\nAll done. Output written to {outdir}")


if __name__ == "__main__":
    main()

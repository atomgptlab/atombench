#!/usr/bin/env python3
"""
atombench.plots — canonical plotting module for crystal reconstruction benchmarks.

Entry point:  atombench-plots --root <job_runs_dir> --outdir <figures_dir>
Importable:   from atombench.plots import plot_rmse_bar_chart, ...

Discovery expects --root to be a directory whose subdirectories each contain
a benchmark CSV (id, target, prediction) and optionally a metrics.json.
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

from atombench._common import discover_benchmark_csvs

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ── Visual constants ───────────────────────────────────────────────────────────
WVU_BLUE    = "#002855"
WVU_GOLD    = "#EEAA00"
LEN_GRANITE = ["#4A6272", "#89A9BC", "#D6E3EC"]
ANG_GRANITE = ["#6A5560", "#B08A97", "#E7D6DC"]

PARAMS_GRID  = ["a", "c", "gamma"]
PARAM_LABEL  = {"a": r"$a$ (Å)", "c": r"$c$ (Å)", "gamma": r"$\gamma$ (°)"}
PANEL_LETTERS = [f"({chr(97+i)})" for i in range(26)]

CRYSYS_PLOT_ORDER = [
    "cubic", "hexagonal", "trigonal",
    "tetragonal", "orthorhombic", "monoclinic",
]
AX_LABEL_MAP = {
    "a": r"$a$", "b": r"$b$", "c": r"$c$",
    "alpha": r"$\alpha$", "beta": r"$\beta$", "gamma": r"$\gamma$",
}
PARAMS_ALL = ("a", "b", "c", "alpha", "beta", "gamma")

# ── Model metadata (benchmark display names, ordering, colors) ─────────────────
BENCHMARK_DISPLAY_NAMES: Dict[str, str] = {
    "agpt_benchmark_alex":               "AtomGPT Alexandria",
    "agpt_benchmark_jarvis":             "AtomGPT JARVIS",
    "cdvae_benchmark_alex":              "CDVAE Alexandria",
    "cdvae_benchmark_jarvis":            "CDVAE JARVIS",
    "flowmm_benchmark_alex":             "FlowMM Alexandria",
    "flowmm_benchmark_jarvis":           "FlowMM JARVIS",
    "agpt_stoich_benchmark_alex":        "AtomGPT Alexandria",
    "agpt_stoich_benchmark_jarvis":      "AtomGPT JARVIS",
    "mattergen_stoich_benchmark_alex":   "MatterGen Alexandria",
    "mattergen_stoich_benchmark_jarvis": "MatterGen JARVIS",
    "mattergen_tc_finetune_benchmark_alex":  "MatterGen Tc Alexandria",
    "mattergen_tc_finetune_benchmark_jarvis": "MatterGen Tc JARVIS",
    "mattergen_benchmark_alex":          "MatterGen Finetuned Alexandria",
    "mattergen_benchmark_jarvis":        "MatterGen Finetuned JARVIS",
}

ICE_ORDER = [
    "cdvae_benchmark_alex",              "cdvae_benchmark_jarvis",
    "agpt_stoich_benchmark_alex",        "agpt_stoich_benchmark_jarvis",
    "mattergen_stoich_benchmark_alex",   "mattergen_stoich_benchmark_jarvis",
    "mattergen_tc_finetune_benchmark_alex", "mattergen_tc_finetune_benchmark_jarvis",
    "agpt_benchmark_alex",               "agpt_benchmark_jarvis",
    "mattergen_benchmark_alex",          "mattergen_benchmark_jarvis",
    "flowmm_benchmark_alex",             "flowmm_benchmark_jarvis",
]


def infer_model(name: str) -> str:
    n = name.lower()
    if n.startswith("agpt_stoich_"):            return "AtomGPT"
    if n.startswith("agpt_"):                   return "AtomGPT"
    if n.startswith("cdvae_"):                  return "CDVAE"
    if n.startswith("flowmm_"):                 return "FlowMM"
    if n.startswith("mattergen_tc_finetune_"):  return "MatterGen Tc"
    if n.startswith("mattergen_stoich_"):        return "MatterGen"
    if n.startswith("mattergen_"):               return "MatterGen Finetuned"
    return "Other"


MODEL_COLORS: Dict[str, str] = {
    "AtomGPT":             "#1f77b4",   # tab:blue
    "CDVAE":               "#ff7f0e",   # tab:orange
    "FlowMM":              "#2ca02c",   # tab:green
    "MatterGen Finetuned": "#d62728",   # tab:red
    "MatterGen":           "#8c564b",   # tab:brown
    "MatterGen Tc":        "#e377c2",   # tab:pink
    "Other":               "#7f7f7f",
}


def _darken(hex_color: str, factor: float = 0.65) -> str:
    """Scale each RGB channel by factor to produce a darker shade."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"


CCRMSE_MODEL_COLORS: Dict[str, str] = {k: _darken(v) for k, v in MODEL_COLORS.items()}

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


# ── Discovery helpers ──────────────────────────────────────────────────────────
def discover_benchmarks(
    root: Path,
) -> List[Tuple[str, Path, Optional[Path], Optional[dict]]]:
    """
    Return (bench_name, bench_dir, csv_path, metrics_dict_or_None) for each
    benchmark found under *root*.

    Input validation is delegated to discover_benchmark_csvs (assert-based):
    *root* must be a benchmark CSV file or a directory of benchmark CSV files.
    metrics.json is loaded from the same directory as each CSV.
    """
    pairs = discover_benchmark_csvs(root)
    results = []
    for name, csv_path in pairs:
        bench_dir = csv_path.parent
        metrics: Optional[dict] = None
        mfp = bench_dir / "metrics.json"
        if mfp.is_file():
            try:
                with mfp.open() as fh:
                    metrics = json.load(fh)
            except Exception as e:
                print(f"⚠  {name}: could not load metrics.json — {e}", file=sys.stderr)
        results.append((name, bench_dir, csv_path, metrics))
    return results


# ── Shared bar-chart helpers ───────────────────────────────────────────────────
def _style_bar_axes(ax, ylabel: str, title: str) -> None:
    ax.set_xlabel("", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=22)
    ax.legend(title="Lattice Parameter", title_fontsize=15, fontsize=15)
    plt.xticks(rotation=30, ha="right", fontsize=13)
    plt.yticks(fontsize=15)
    plt.tight_layout()


def _build_metrics_df(
    benchmarks: List[Tuple[str, Path, Optional[Path], Optional[dict]]],
    display_names: Dict[str, str],
) -> pd.DataFrame:
    """Flatten all metrics dicts into a single DataFrame, applying display names."""
    rows = []
    for name, _, _, metrics in benchmarks:
        if metrics is None:
            continue
        m = dict(metrics)
        m["benchmark_name"] = display_names.get(
            m.get("benchmark_name", name), m.get("benchmark_name", name)
        )
        rows.append(pd.json_normalize(m, sep=".", max_level=3).iloc[0].to_dict())
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _model_bar_chart(
    df: pd.DataFrame,
    value_col: str,
    outdir: Path,
    filename: str,
    ylabel: str,
    title: str,
    colors: Dict[str, str],
    display_names: Dict[str, str],
) -> None:
    """Shared implementation for model-colored bar charts (RMSE and ccRMSE)."""
    if value_col not in df.columns:
        print(f"⚠  '{value_col}' not in metrics — skipping {filename}", file=sys.stderr)
        return

    # Apply ICE ordering and display names
    raw_names = df["benchmark_name"].tolist()
    ice_keys  = [k for k in ICE_ORDER if k in raw_names]
    if not ice_keys:
        ice_keys = raw_names   # fall back to whatever order we have

    plot_df = (
        df[["benchmark_name", value_col]]
          .rename(columns={value_col: "value"})
          .assign(
              model   = lambda x: x["benchmark_name"].apply(infer_model),
              display = lambda x: x["benchmark_name"].map(display_names)
                                    .fillna(x["benchmark_name"]),
          )
          .set_index("benchmark_name")
          .reindex(ice_keys)
          .reset_index()
    )

    plot_df["color"] = plot_df["model"].map(colors).fillna(colors.get("Other", "#7f7f7f"))

    pos = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(pos, plot_df["value"].astype(float).tolist(),
           width=0.55, edgecolor="k", linewidth=0.8,
           color=plot_df["color"].tolist())
    ax.set_xticks(pos)
    ax.set_xticklabels(plot_df["display"].tolist(), rotation=30, ha="right", fontsize=13)

    shown_models = plot_df["model"].unique()
    handles = [
        mpatches.Patch(color=colors[m], label=m)
        for m in MODEL_COLORS   # use canonical order
        if m in shown_models and m in colors
    ]
    ax.legend(handles=handles, title_fontsize=15, fontsize=15)
    ax.set_xlabel("", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=22)
    plt.xticks(rotation=30, ha="right", fontsize=13)
    plt.yticks(fontsize=15)
    plt.tight_layout()

    out_path = outdir / filename
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✓ {filename}")


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
    _overlay(plt.subplot(grid[0, 1]), xs["c"],     ys["c"],     np.arange(2, 7, 0.1),
             r"c ($\AA$)", "(b)")
    _overlay(plt.subplot(grid[0, 2]), xs["gamma"], ys["gamma"], np.arange(30, 150, 10),
             r"$\gamma$ ($^\circ$)", "(c)")

    x_spg, y_spg, x_Z, y_Z, x_lat, y_lat = [], [], [], [], [], []
    lat_order  = ["triclinic", "monoclinic", "orthorhombic",
                  "tetragonal", "trigonal", "hexagonal", "cubic"]
    lat_to_idx = {name: i for i, name in enumerate(lat_order)}

    for _, row in df.iterrows():
        try:
            st = _reduced_struct(str(row["target"]))
            sp = _reduced_struct(str(row["prediction"]))
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

    _overlay(plt.subplot(grid[1, 2]), x_Z, y_Z, np.arange(15, 2000, 100),
             "Weight (AMU)", "(f)")

    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.suptitle(bench_name, fontsize=30)
    out_png = outdir / f"{bench_name}_distribution.png"
    plt.savefig(out_png, format="png", dpi=200)
    plt.close(fig)
    print(f"✓ {out_png.name}")


def plot_kld_bar_chart(df: pd.DataFrame, outdir: Path) -> None:
    kld_cols = [f"KLD.{k}" for k in PARAMS_ALL]
    if any(c not in df.columns for c in kld_cols):
        print("⚠  Missing KLD columns — skipping KLD bar chart", file=sys.stderr)
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
    for cand in ([f"MAE.average_mae.{k}" for k in PARAMS_ALL],
                 [f"MAE.{k}" for k in PARAMS_ALL]):
        if all(c in df.columns for c in cand):
            mae_cols = cand
            break
    else:
        print("⚠  Missing MAE columns — skipping MAE bar charts", file=sys.stderr)
        return
    mae_df = (df.set_index("benchmark_name")[mae_cols]
                .rename(columns=lambda c: AX_LABEL_MAP[c.split(".")[-1]]))
    length_cols = [AX_LABEL_MAP[k] for k in ("a", "b", "c")]
    fig, ax = plt.subplots(figsize=(10, 8))
    mae_df[length_cols].plot(kind="bar", edgecolor="k", ax=ax)
    _style_bar_axes(ax, "Mean Absolute Error (Å)", "Mean Absolute Error – Lattice Lengths (Å)")
    plt.savefig(outdir / "mae_bar_chart_abc.png", dpi=300)
    plt.close(fig)
    print("✓ mae_bar_chart_abc.png")


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
    print("✓ mae_bar_chart_angles.png")


def plot_rmse_bar_chart(
    df: pd.DataFrame, outdir: Path, display_names: Optional[Dict[str, str]] = None
) -> None:
    dn = display_names if display_names is not None else BENCHMARK_DISPLAY_NAMES
    for cand in ["RMSE.AtomGen.mean_cartesian_rms_angstrom",
                 "RMSE.AtomGen.mean_normalized_cartesian_rms",
                 "RMSE.AtomGen", "RMSE"]:
        if cand in df.columns:
            _model_bar_chart(
                df, cand, outdir, "rmse_bar_chart.png",
                "Average RMSE (Å)",
                "Average Root Mean Squared Error\nfor Predicted vs. Target Atomic Coordinates",
                MODEL_COLORS, dn,
            )
            return
    print("⚠  No RMSE column found — skipping RMSE bar chart", file=sys.stderr)


def plot_ccrmse_bar_chart(
    df: pd.DataFrame, outdir: Path, display_names: Optional[Dict[str, str]] = None
) -> None:
    dn = display_names if display_names is not None else BENCHMARK_DISPLAY_NAMES
    if "ccRMSE.value" not in df.columns:
        print("⚠  'ccRMSE.value' not in metrics — skipping ccRMSE bar chart", file=sys.stderr)
        return
    _model_bar_chart(
        df, "ccRMSE.value", outdir, "ccrmse_bar_chart.png",
        "AMD-RMSE (Å)",
        "Continuous Corrected RMSE\nfor Predicted vs. Target Structures",
        CCRMSE_MODEL_COLORS, dn,
    )


def plot_match_rate_bar_chart(
    df: pd.DataFrame, outdir: Path, display_names: Optional[Dict[str, str]] = None
) -> None:
    dn = display_names if display_names is not None else BENCHMARK_DISPLAY_NAMES
    mr_col = "RMSE.AtomGen.match_rate"
    if mr_col not in df.columns:
        print("⚠  No match_rate column — skipping match rate chart", file=sys.stderr)
        return

    raw_names = df["benchmark_name"].tolist()
    ice_keys  = [k for k in ICE_ORDER if k in raw_names] or raw_names
    plot_df   = (
        df[["benchmark_name", mr_col]]
          .rename(columns={mr_col: "match_rate"})
          .assign(
              model   = lambda x: x["benchmark_name"].apply(infer_model),
              display = lambda x: x["benchmark_name"].map(dn).fillna(x["benchmark_name"]),
          )
          .set_index("benchmark_name")
          .reindex(ice_keys)
          .reset_index()
    )
    plot_df["color"] = plot_df["model"].map(MODEL_COLORS).fillna(MODEL_COLORS["Other"])

    pos = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(pos, plot_df["match_rate"].astype(float).tolist(),
           width=0.55, edgecolor="k", linewidth=0.8,
           color=plot_df["color"].tolist())
    ax.set_xticks(pos)
    ax.set_xticklabels(plot_df["display"].tolist(), rotation=30, ha="right", fontsize=13)

    shown_models = plot_df["model"].unique()
    handles = [
        mpatches.Patch(color=MODEL_COLORS[m], label=m)
        for m in MODEL_COLORS if m in shown_models
    ]
    ax.legend(handles=handles, title_fontsize=15, fontsize=15)
    ax.set_ylabel("Match Rate", fontsize=16)
    ax.set_title("Structure Matcher Match Rate (STOL=0.5)", fontsize=22)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha="right", fontsize=13)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(outdir / "match_rate_bar_chart.png", dpi=300)
    plt.close(fig)
    print("✓ match_rate_bar_chart.png")


def plot_crystal_system_mae_from_json(
    metrics_dicts: List[dict], outdir: Path, kmin: int
) -> None:
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
        print("⚠  No crystal systems pass kmin — skipping crystal-system MAE charts",
              file=sys.stderr)
        return

    data, counts_arr = {}, []
    for cs in systems:
        n        = pooled_count[cs]
        data[cs] = {p: pooled_sum[cs][p] / n for p in PARAMS_ALL}
        counts_arr.append(n)

    g       = pd.DataFrame(data).T[list(PARAMS_ALL)]
    g.index = [s.capitalize() for s in systems]
    plot_df = g.rename(columns=AX_LABEL_MAP)
    counts_arr = np.array(counts_arr, dtype=int)

    def _add_counts(ax, counts, tops):
        ylim = ax.get_ylim()
        ymax = max(ylim[1], float(np.max(tops)) * 1.22 if len(tops) else ylim[1])
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
    print("✓ crystal_system_mae_bar_chart_abc.png")

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
    print("✓ crystal_system_mae_bar_chart_angles.png")


def plot_group_grid(group_cfg: dict, root: Path, outdir: Path) -> None:
    """N×3 reconstruction grid for a named group of benchmarks (requires --config)."""
    benchmarks: Dict[str, str] = group_cfg["benchmarks"]   # label -> dir_name
    title      = group_cfg.get("title", group_cfg["name"])
    bins_cfg   = group_cfg.get("bins", {})

    bins_a_c   = np.arange(
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
                       left=False, right=False, labelleft=False, labelright=False)

    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.12, top=0.86,
                        wspace=0.10, hspace=0.30)

    def _weights_pct(n: int) -> np.ndarray:
        return np.ones(n, dtype=float) * (100.0 / n) if n > 0 else np.array([])

    def _unescape(s: str) -> str:
        return s.replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n").strip()

    def _extract(csv_path: Path) -> Dict[str, Dict[str, List[float]]]:
        out: Dict[str, Dict[str, List[float]]] = {
            "target": {k: [] for k in PARAMS_GRID},
            "predicted": {k: [] for k in PARAMS_GRID},
        }
        with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv_mod.DictReader(fh)
            for row in reader:
                try:
                    tp = _niggli_params(_unescape(row["target"]))
                    pp = _niggli_params(_unescape(row["prediction"]))
                    out["target"]["a"].append(tp[0])
                    out["target"]["c"].append(tp[2])
                    out["target"]["gamma"].append(tp[5])
                    out["predicted"]["a"].append(pp[0])
                    out["predicted"]["c"].append(pp[2])
                    out["predicted"]["gamma"].append(pp[5])
                except Exception:
                    continue
        return out

    panel_idx = 0
    for r, (row_label, dir_name) in enumerate(benchmarks.items()):
        bench_dir = root / dir_name
        series: Optional[Dict] = None
        klds:   Dict[str, float] = {}

        if bench_dir.is_dir():
            csv_path = find_benchmark_csv(bench_dir)
            if csv_path:
                series = _extract(csv_path)
            mfp = bench_dir / "metrics.json"
            if mfp.exists():
                try:
                    data = json.loads(mfp.read_text())
                    k = data.get("KLD", {})
                    klds = {p: float(k.get(p, float("nan"))) for p in PARAMS_GRID}
                except Exception:
                    pass
        else:
            print(f"⚠  {dir_name}: directory not found — row shows 'no data'", file=sys.stderr)

        for c, param in enumerate(PARAMS_GRID):
            ax = axes[r, c]
            ax.text(0.03, 0.92, PANEL_LETTERS[panel_idx], transform=ax.transAxes,
                    ha="left", va="top", fontsize=9)
            panel_idx += 1
            if c == 0:
                ax.set_ylabel(row_label, fontsize=22)

            if series is None or not series["target"][param]:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        fontsize=12, alpha=0.7)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            xt = np.asarray(series["target"][param],    dtype=float)
            xp = np.asarray(series["predicted"][param], dtype=float)
            bins = bins_a_c if param in ("a", "c") else bins_gamma

            ax.hist(xt, bins=bins, weights=_weights_pct(len(xt)),
                    histtype="stepfilled", alpha=0.68, color=WVU_BLUE,
                    edgecolor="none", label="target")
            ax.hist(xp, bins=bins, weights=_weights_pct(len(xp)),
                    histtype="stepfilled", alpha=0.68, color=WVU_GOLD,
                    edgecolor="none", label="predicted")

            if r == n_rows - 1:
                ax.set_xlabel(PARAM_LABEL[param], fontsize=14)

            ax.tick_params(axis="both", which="major", labelsize=12, width=1.4, length=7)
            ax.tick_params(axis="y", which="both", left=False)
            ax.minorticks_off()
            if c != 0:
                ax.set_yticklabels([])

            kld_val = klds.get(param)
            if kld_val is not None and np.isfinite(kld_val):
                ax.text(0.97, 0.92, f"KLD = {kld_val:.3f}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black",
                                  lw=0.8, alpha=0.95))

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

    slug = group_cfg["name"].lower().replace(" ", "_").replace("/", "_")
    out_png = outdir / f"{slug}_reconstruction_grid.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out_png.name}")


# ── CLI ────────────────────────────────────────────────────────────────────────
ALL_CHART_TYPES = ("kld", "mae", "rmse", "ccrmse", "match-rate",
                   "grid", "crystal-system-mae", "distribution")


def main(argv=None) -> None:
    """Generate all reconstruction benchmark plots from metrics.json + CSV files."""
    ap = argparse.ArgumentParser(
        description="Generate crystal reconstruction benchmark plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root",    required=True, type=Path,
                    help="Directory containing benchmark subdirs (CSV + metrics.json each).")
    ap.add_argument("--outdir",  default="figures", type=Path,
                    help="Output directory for PNGs (default: figures/).")
    ap.add_argument("--config",  default=None,
                    help="Optional JSON config for grid charts and display name overrides.")
    ap.add_argument("--kmin",    default=10,  type=int,
                    help="Min pooled structures per crystal system (default: 10).")
    ap.add_argument("--symprec", default=0.1, type=float,
                    help="Symmetry tolerance passed through to metrics (informational).")
    ap.add_argument("--only",    nargs="+", default=None,
                    choices=ALL_CHART_TYPES, metavar="CHART",
                    help=(
                        "Generate only the specified chart type(s). "
                        f"Choices: {', '.join(ALL_CHART_TYPES)}. "
                        "Default: all charts."
                    ))
    args = ap.parse_args(argv)

    root   = args.root.resolve()
    outdir = args.outdir.resolve()

    assert root.exists(), f"--root does not exist: {root}"
    outdir.mkdir(parents=True, exist_ok=True)

    only = set(args.only) if args.only else set(ALL_CHART_TYPES)

    # ── Load config ───────────────────────────────────────────────────────────
    display_names: Dict[str, str] = dict(BENCHMARK_DISPLAY_NAMES)
    groups: list = []
    if args.config:
        config_path = Path(args.config).resolve()
        if not config_path.is_file():
            sys.exit(f"ERROR: --config not found: {config_path}")
        with config_path.open() as fh:
            cfg = json.load(fh)
        display_names.update(cfg.get("display_names", {}))
        groups = cfg.get("groups", [])

    # ── Discover benchmarks ───────────────────────────────────────────────────
    benchmarks = discover_benchmarks(root)  # asserts fire inside if no CSVs found

    metrics_dicts: List[dict] = []
    bench_data: List[Tuple[str, Path, Optional[Path], Optional[dict]]] = []

    for name, bench_dir, csv_path, metrics in benchmarks:
        if csv_path is None and ("distribution" in only or "grid" in only):
            print(f"⚠  {name}: no CSV found — distribution/grid skipped", file=sys.stderr)
        if metrics is None and (only - {"distribution", "grid"}):
            print(f"⚠  {name}: no metrics.json — bar charts skipped", file=sys.stderr)
        bench_data.append((name, bench_dir, csv_path, metrics))
        if metrics is not None:
            metrics_dicts.append(metrics)

    # ── Build flat DataFrame for bar charts ───────────────────────────────────
    df_metrics = _build_metrics_df(benchmarks, display_names)

    # ── Distribution plots ────────────────────────────────────────────────────
    if "distribution" in only:
        print("\n── Distribution plots")
        for name, _, csv_path, _ in bench_data:
            if csv_path is None:
                continue
            try:
                plot_distribution(name, csv_path, outdir)
            except Exception as e:
                print(f"⚠  {name}: distribution plot failed — {e}", file=sys.stderr)

    # ── KLD / MAE bar charts ──────────────────────────────────────────────────
    if not df_metrics.empty and ("kld" in only or "mae" in only):
        print("\n── KLD / MAE bar charts")
        if "kld" in only:
            plot_kld_bar_chart(df_metrics, outdir)
        if "mae" in only:
            plot_mae_abc_bar_chart(df_metrics, outdir)
            plot_mae_angles_bar_chart(df_metrics, outdir)

    # ── RMSE bar chart ────────────────────────────────────────────────────────
    if not df_metrics.empty and "rmse" in only:
        print("\n── RMSE bar chart")
        plot_rmse_bar_chart(df_metrics, outdir, display_names)

    # ── ccRMSE bar chart ──────────────────────────────────────────────────────
    if not df_metrics.empty and "ccrmse" in only:
        print("\n── ccRMSE bar chart")
        plot_ccrmse_bar_chart(df_metrics, outdir, display_names)

    # ── Match-rate bar chart ──────────────────────────────────────────────────
    if not df_metrics.empty and "match-rate" in only:
        print("\n── Match-rate bar chart")
        plot_match_rate_bar_chart(df_metrics, outdir, display_names)

    # ── Reconstruction grid charts ────────────────────────────────────────────
    if "grid" in only:
        if not groups:
            print("\nℹ  No groups in config — skipping grid charts", file=sys.stderr)
        else:
            print("\n── Reconstruction grid charts")
            for group_cfg in groups:
                try:
                    plot_group_grid(group_cfg, root, outdir)
                except Exception as e:
                    print(f"⚠  Group '{group_cfg.get('name')}': grid failed — {e}",
                          file=sys.stderr)

    # ── Crystal-system MAE charts ─────────────────────────────────────────────
    if "crystal-system-mae" in only:
        if not metrics_dicts:
            print("\n⚠  No metrics dicts — skipping crystal-system MAE charts", file=sys.stderr)
        elif not any("crystal_system_mae" in m for m in metrics_dicts):
            print("\nℹ  No crystal_system_mae key in any metrics.json — skipping",
                  file=sys.stderr)
        else:
            print("\n── Crystal-system MAE charts")
            plot_crystal_system_mae_from_json(metrics_dicts, outdir, kmin=args.kmin)

    print(f"\nDone. Output: {outdir}")


if __name__ == "__main__":
    main()

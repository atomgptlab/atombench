#!/usr/bin/env python3
"""
reconstruction_grid.py  —  3×3 overlay figures (Alexandria & JARVIS)

- Rows  : AtomGPT, CDVAE, FlowMM
- Cols  : a, c, γ
- Colors: WVU Blue (target) / WVU Gold (predicted)
- KLD   : pulled from each benchmark's metrics.json and shown prominently
- Bins  : HARD-CODED and uniform within each PNG (Alex bins ≠ JARVIS bins)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from jarvis.io.vasp.inputs import Poscar
from functools import lru_cache
from pymatgen.core import Structure

# ───────────────────────── visual style ─────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "axes.linewidth": 0.8,
    #"axes.titleweight": "bold",
    "patch.linewidth": 0.0,
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif", "STIX"],
})

WVU_BLUE = "#002855"  # target
WVU_GOLD = "#EEAA00"  # predicted

MODELS = ["agpt", "cdvae", "flowmm", "mattergen_base", "mattergen_tc_finetune", "mattergen"]
MODEL_LABEL = {
    "agpt":                  "AtomGPT",
    "cdvae":                 "CDVAE",
    "flowmm":                "FlowMM",
    "mattergen_base":        "MatterGen\nBase",
    "mattergen_tc_finetune": "MatterGen\nTC Finetune",
    "mattergen":             "MatterGen\nFinetuned",
}
PARAMS = ["a", "c", "gamma"]
PARAM_LABEL = {"a": r"$a$ (Å)", "c": r"$c$ (Å)", "gamma": r"$\gamma$ (°)"}
PANEL_LETTERS = [f"({chr(97+i)})" for i in range(26)]

# ── HARD-CODED BINS (uniform within each PNG) ─────────────────────────────
# Alexandria: finer for a/c, wide for γ
ALEX_BINS_A_C = np.arange(2.0, 10.0 + 1e-9, 0.10)     # width = 0.20 Å
ALEX_BINS_GAM  = np.arange(30.0, 140.0 + 1e-9, 8.0) # width = 10°
# JARVIS: slightly coarser for a/c given size, γ same width
JARV_BINS_A_C  = np.arange(2.0, 10.0 + 1e-9, 0.25)    # width = 0.25 Å
JARV_BINS_GAM  = np.arange(30.0, 140.0 + 1e-9, 10.0) # width = 10°

# ───────────────────── discovery helpers ─────────────────────
def find_benchmark_csv(dir_path: Path) -> Optional[Path]:
    """Newest CSV under dir_path (recursively) that has id/target/prediction columns."""
    latest: Optional[Tuple[float, Path]] = None
    for p, _, files in os.walk(dir_path):
        for f in files:
            if not f.lower().endswith(".csv"):
                continue
            path = Path(p) / f
            try:
                with path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
                    reader = csv.DictReader(fh)
                    fields = [fn.strip().lower() for fn in (reader.fieldnames or [])]
                    if {"id", "target", "prediction"}.issubset(set(fields)):
                        mt = path.stat().st_mtime
                        if latest is None or mt > latest[0]:
                            latest = (mt, path)
            except Exception:
                continue
    return latest[1] if latest else None

@lru_cache(maxsize=4096)
def _niggli_params(poscar_str: str):
    s = Structure.from_str(_unescape_poscar(poscar_str), fmt="poscar")
    # optional but recommended: compare on primitive cells first
    s = s.get_primitive_structure()
    s = s.get_reduced_structure(reduction_algo="niggli")  # Niggli canonical cell
    a, b, c = s.lattice.abc
    alpha, beta, gamma = s.lattice.angles  # degrees
    return a, b, c, alpha, beta, gamma

def discover_model_dirs(root: Path, tag: str) -> Dict[str, Path]:
    """Find dirs for each model for a dataset tag ('alex' or 'jarvis')."""
    out: Dict[str, Path] = {}
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name.lower()
        if tag not in name:
            continue
        for m in MODELS:
            if m in name:
                out[m] = entry
                break
    return out

# ───────────────────── POSCAR & metrics ─────────────────────
def _unescape_poscar(s: str) -> str:
    return (
        s.replace("\r\n", "\n")
         .replace("\r", "\n")
         .replace("\\n", "\n")
         .replace("\\t", " ")
         .strip()
    )

def extract_series(csv_path: Path) -> Dict[str, Dict[str, List[float]]]:
    out = {"target": {k: [] for k in PARAMS}, "predicted": {k: [] for k in PARAMS}}
    with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                ta, tb, tc, tal, tbe, tga = _niggli_params(row["target"])
                pa, pb, pc, pal, pbe, pga = _niggli_params(row["prediction"])
                out["target"]["a"].append(float(ta))
                out["target"]["c"].append(float(tc))
                out["target"]["gamma"].append(float(tga))
                out["predicted"]["a"].append(float(pa))
                out["predicted"]["c"].append(float(pc))
                out["predicted"]["gamma"].append(float(pga))
            except Exception:
                continue
    return out


def load_klds(metrics_path: Path) -> Dict[str, float]:
    """metrics.json -> {'a':…, 'c':…, 'gamma':…} (best-effort)."""
    try:
        data = json.loads(metrics_path.read_text())
        k = data.get("KLD", {})
        return {"a": float(k.get("a")), "c": float(k.get("c")), "gamma": float(k.get("gamma"))}
    except Exception:
        return {}

# ───────────────────── plotting ─────────────────────
def _weights_percent(n: int) -> np.ndarray:
    return np.ones(n, dtype=float) * (100.0 / n) if n > 0 else np.array([])

def style_axes(ax: plt.Axes, left_col: bool, bottom_row: bool) -> None:
    """Salient ticks/labels, minimal clutter."""
    ax.tick_params(axis="both", which="major", labelsize=12, width=1.4, length=7)
    ax.tick_params(axis="y", which="both", left=False)
    #ax.tick_params(axis="both", which="minor", width=1.0, length=4)
    ax.minorticks_off()
    if not left_col:
        ax.set_yticklabels([])
    if not bottom_row:
        ax.set_xlabel("")

def annotate_kld(ax: plt.Axes, value: Optional[float]) -> None:
    if value is None:
        return
    ax.text(
        0.97, 0.92, f"KLD = {value:.3f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=0.8, alpha=0.95),
    )

def annotate_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.03, 0.92, label,          
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9,                     
    )                                                                   

def plot_dataset_grid(tag: str,
                      root: Path,
                      out_png: Path,
                      title: str,
                      bins_a_c: np.ndarray,
                      bins_gamma: np.ndarray) -> None:
    """Build a 3×3 grid for one dataset tag ('alex' or 'jarvis')."""
    model_dirs = discover_model_dirs(root, tag)

    # Smaller, tighter figure. We'll control margins to keep the right-side ylabel
    # clearly separated from the panels (no overlap).
    n_models = len(MODELS)
    fig, axes = plt.subplots(n_models, 3, figsize=(9, 2.8 * n_models))
    # remove all y-axis labels and tick labels
    for ax in axes.ravel():
        ax.set_ylabel("")                                # kill axis label
        ax.tick_params(axis="y", which="both",
                    left=False, right=False,          # no tick marks
                    labelleft=False, labelright=False)  # no numbers
    fig.subplots_adjust(
        left=0.08, right=0.88, bottom=0.12, top=0.86,
        wspace=0.10, hspace=0.30
    )

    # column titles — extra salient
    #for c, p in enumerate(PARAMS):
        #axes[0, c].set_title(PARAM_LABEL[p], fontsize=18, pad=7)

    for r, model in enumerate(MODELS):
        axrow = axes[r]
        label = MODEL_LABEL[model]
        bench_dir = model_dirs.get(model)

        series = None
        klds = {}
        if bench_dir:
            csv_path = find_benchmark_csv(bench_dir)
            if csv_path:
                series = extract_series(csv_path)
            mfp = bench_dir / "metrics.json"
            if mfp.exists():
                klds = load_klds(mfp)

        for c, param in enumerate(PARAMS):
            ax = axrow[c]
            annotate_panel_label(ax, PANEL_LETTERS[r*3 + c])
            if c == 0:
                ax.set_ylabel(label, fontsize=22)

            if series is None or not series["target"][param] or not series["predicted"][param]:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=12, alpha=0.7)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            xt = np.asarray(series["target"][param], dtype=float)
            xp = np.asarray(series["predicted"][param], dtype=float)

            bins = bins_a_c if param in ("a", "c") else bins_gamma
            wt_t = _weights_percent(len(xt))
            wt_p = _weights_percent(len(xp))

            # Less “rectangular”: stepfilled with rounded joins; no edges.
            ax.hist(xt, bins=bins, weights=wt_t,
                    histtype="stepfilled", alpha=0.68, color=WVU_BLUE,
                    edgecolor="none", label="target")
            ax.hist(xp, bins=bins, weights=wt_p,
                    histtype="stepfilled", alpha=0.68, color=WVU_GOLD,
                    edgecolor="none", label="predicted")

            if r == len(MODELS) - 1:
                ax.set_xlabel(PARAM_LABEL[param], fontsize=14)

            style_axes(ax, left_col=(c == 0), bottom_row=(r == len(MODELS) - 1))
            annotate_kld(ax, klds.get(param))

    # single legend (clear & compact)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        leg = axes[0, 0].legend(handles, labels, loc="center right", frameon=True, fontsize=12)
        leg.get_frame().set_alpha(0.95)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_linewidth(0.6)

    # bold suptitle & global right-side y-label (kept outside the panels)
    fig.suptitle(title, fontsize=28, y=0.93)
    fig.text(0.91, 0.5, "Materials Percentage (%)",
             rotation=270, va="center", ha="center",
             fontsize=20)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ wrote {out_png}")

# ────────────────────────────── CLI ──────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="3×3 overlay figures with salient styling and HARD-CODED per-dataset bins."
    )
    ap.add_argument("--root", type=Path, default=Path("job_runs"), help="Root containing benchmark subdirs")
    ap.add_argument("--outdir", type=Path, default=Path("figures"), help="Directory to write PNGs")
    ap.add_argument("--alex-title", default="Alexandria DS-A/B Reconstruction KLD", help="Suptitle for Alexandria figure")
    ap.add_argument("--jarvis-title", default="JARVIS Supercon-3D Reconstruction KLD", help="Suptitle for JARVIS figure")
    args = ap.parse_args()

    plot_dataset_grid(
        "alex", args.root, args.outdir / "alexandria_reconstruction_grid.png",
        args.alex_title, ALEX_BINS_A_C, ALEX_BINS_GAM
    )
    plot_dataset_grid(
        "jarvis", args.root, args.outdir / "jarvis_reconstruction_grid.png",
        args.jarvis_title, JARV_BINS_A_C, JARV_BINS_GAM
    )

if __name__ == "__main__":
    main()


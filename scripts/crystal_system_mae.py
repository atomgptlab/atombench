#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from functools import lru_cache
import sys
import json

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["font.family"] = "serif"
import matplotlib.pyplot as plt

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# All valid crystal systems for classification
CRYSYS_ALL = [
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
]

# Plot/report order: most symmetric -> least symmetric, excluding triclinic
CRYSYS_PLOT_ORDER = [
    "cubic",
    "hexagonal",
    "trigonal",
    "tetragonal",
    "orthorhombic",
    "monoclinic",
]

AX_LABEL_MAP = {
    "a":     r"$a$",
    "b":     r"$b$",
    "c":     r"$c$",
    "alpha": r"$\alpha$",
    "beta":  r"$\beta$",
    "gamma": r"$\gamma$",
}

# Lengths: brighter granite-ish blue/grey (dark → mid → light)
LEN_GRANITE = ["#4A6272", "#89A9BC", "#D6E3EC"]

# Angles: same vibe, slightly warmer / redder and brighter (dark → mid → light)
ANG_GRANITE = ["#6A5560", "#B08A97", "#E7D6DC"]

PNG_DPI = 500


def style_axes(ax, ylabel: str, title: str):
    ax.set_xlabel("", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=22)
    ax.legend(
        title="Lattice Parameter",
        title_fontsize=15,
        fontsize=15,
        loc="center left",
        bbox_to_anchor=(0.02, 0.66),
        borderaxespad=0.0,
    )
    plt.xticks(rotation=30, ha="right", fontsize=13)
    plt.yticks(fontsize=15)


def add_group_counts(
    ax,
    counts: np.ndarray,
    group_tops: np.ndarray,
    labels: list[str],
    x_offsets: dict[str, float] | None = None,
    fontsize: int = 12,
):
    if len(counts) == 0:
        return
    if x_offsets is None:
        x_offsets = {}

    current_ylim = ax.get_ylim()
    ymax_needed = float(np.max(group_tops)) if len(group_tops) else current_ylim[1]
    new_ymax = max(current_ylim[1], ymax_needed * 1.22 if ymax_needed > 0 else 1.0)
    ax.set_ylim(0.0, new_ymax)

    offset_y = 0.02 * ax.get_ylim()[1]
    for i, (n, top) in enumerate(zip(counts, group_tops)):
        lab = labels[i] if i < len(labels) else ""
        dx = float(x_offsets.get(lab, 0.0))
        ax.text(
            i + dx,
            float(top) + offset_y,
            f"n={int(n)}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


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


def crystal_system_from_structure(s: Structure, symprec: float) -> str | None:
    try:
        sga = SpacegroupAnalyzer(s, symprec=symprec)
        conv = sga.get_conventional_standard_structure()
        cs = SpacegroupAnalyzer(conv, symprec=symprec).get_crystal_system()
        cs = cs.lower() if isinstance(cs, str) else None
        return cs if cs in CRYSYS_ALL else None
    except Exception:
        return None


def find_prediction_csv(exp_dir: Path) -> Path | None:
    candidates = []
    p1 = exp_dir / "AI-AtomGen-prop-dft_3d-test-rmse.csv"
    if p1.is_file():
        candidates.append(p1)
    candidates += sorted(exp_dir.glob("*test-rmse*.csv"))
    candidates += sorted(exp_dir.glob("*.csv"))

    for c in candidates:
        try:
            df = pd.read_csv(c, nrows=5)
            cols = {x.strip().lower() for x in df.columns}
            if "target" in cols and "prediction" in cols:
                return c
        except Exception:
            continue
    return None


def load_rows_from_exp(exp_dir: Path, symprec: float) -> list[dict]:
    csv_path = find_prediction_csv(exp_dir)
    if csv_path is None:
        print(f"⚠️  {exp_dir.name}: no suitable CSV found — skipped", file=sys.stderr)
        return []

    df = pd.read_csv(csv_path)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    if "target" not in df.columns or "prediction" not in df.columns:
        print(f"⚠️  {exp_dir.name}: CSV lacks target/prediction — skipped", file=sys.stderr)
        return []

    out = []
    for _, r in df.iterrows():
        try:
            t = str(r["target"])
            p = str(r["prediction"])

            s_t = reduced_structure_from_poscar_text(t)
            cs = crystal_system_from_structure(s_t, symprec=symprec)
            if cs is None:
                continue

            ta, tb, tc, tal, tbe, tga = niggli_params_from_poscar_text(t)
            pa, pb, pc, pal, pbe, pga = niggli_params_from_poscar_text(p)

            out.append({
                "benchmark": exp_dir.name,
                "crysys": cs,
                "a": abs(pa - ta),
                "b": abs(pb - tb),
                "c": abs(pc - tc),
                "alpha": abs(pal - tal),
                "beta": abs(pbe - tbe),
                "gamma": abs(pga - tga),
            })
        except Exception:
            continue

    print(f"✓ {exp_dir.name}: loaded {len(out)} rows from {csv_path.name}", file=sys.stderr)
    return out


def _round6(x: float) -> float:
    try:
        return float(round(float(x), 6))
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser(
        description="MAE-by-crystal-system bar charts pooled across all benchmarks in job_runs/."
    )
    ap.add_argument("--root", default="job_runs", help="Root directory containing benchmark subdirs")
    ap.add_argument("--symprec", type=float, default=0.1, help="Symmetry tolerance for SpacegroupAnalyzer")
    ap.add_argument("--kmin", type=int, default=10, help="Minimum number of structures per crystal system")
    ap.add_argument("--outdir", default=".", help="Where to write PNGs/CSVs/JSON")
    ap.add_argument("--title", default=None, help="(Unused; kept for CLI compatibility)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not root.is_dir():
        raise SystemExit(f"--root does not exist: {root}")

    rows = []
    benchmarks_included = []
    for exp_dir in sorted(root.iterdir()):
        if exp_dir.is_dir():
            part = load_rows_from_exp(exp_dir, symprec=float(args.symprec))
            if part:
                benchmarks_included.append(exp_dir.name)
                rows.extend(part)

    if not rows:
        raise SystemExit("No usable rows found across benchmarks.")

    d = pd.DataFrame(rows)

    # Filter by kmin, then keep only plotted systems in explicit symmetry order
    counts = d["crysys"].value_counts()
    keep = [cs for cs in CRYSYS_PLOT_ORDER if counts.get(cs, 0) >= int(args.kmin)]
    d = d[d["crysys"].isin(keep)].copy()

    if d.empty:
        raise SystemExit(f"After kmin={args.kmin}, no plotted crystal systems remain.")

    # Force crystal-system ordering explicitly for plotting/reporting
    d["crysys"] = pd.Categorical(
        d["crysys"],
        categories=CRYSYS_PLOT_ORDER,
        ordered=True,
    )
    d = d.sort_values("crysys")

    # Mean MAE per system (pooled), preserving explicit order
    g = (
        d.groupby("crysys", sort=False)[["a", "b", "c", "alpha", "beta", "gamma"]]
         .mean()
         .reindex(keep)
    )

    # Counts per system in the same explicit order
    g_counts = (
        d.groupby("crysys", sort=False)
         .size()
         .reindex(keep)
         .astype(int)
    )

    # Save CSV
    summary = g.copy()
    summary.insert(0, "n_structures", g_counts)
    summary.to_csv(outdir / "mae_by_crystal_system.csv", index=True)

    # Pretty labels for plotting
    plot_df = g.rename(columns=AX_LABEL_MAP)
    plot_df.index = [s.capitalize() for s in plot_df.index]
    labels = list(plot_df.index)

    length_cols = [AX_LABEL_MAP["a"], AX_LABEL_MAP["b"], AX_LABEL_MAP["c"]]
    angle_cols = [AX_LABEL_MAP["alpha"], AX_LABEL_MAP["beta"], AX_LABEL_MAP["gamma"]]

    counts_arr = g_counts.to_numpy(dtype=int)
    length_tops = g[["a", "b", "c"]].max(axis=1).to_numpy(dtype=float)
    angle_tops = g[["alpha", "beta", "gamma"]].max(axis=1).to_numpy(dtype=float)

    # Titles (3 lines, symbols)
    n_bench = len(benchmarks_included)
    title_len = (
        "Mean Absolute Error by Crystal System\n"
        f"Results Pooled from All {n_bench} Benchmarks\n"
        "Lattice Lengths (Å)"
    )
    title_ang = (
        "Mean Absolute Error by Crystal System\n"
        f"Results Pooled from All {n_bench} Benchmarks\n"
        "Lattice Angles (°)"
    )

    # ── Lengths chart ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_df[length_cols].plot(kind="bar", edgecolor="k", ax=ax, color=LEN_GRANITE)
    style_axes(ax, "Mean Absolute Error (Å)", title_len)
    add_group_counts(
        ax,
        counts_arr,
        length_tops,
        labels=labels,
        x_offsets={},
        fontsize=12,
    )
    fig.tight_layout()
    plt.savefig(outdir / "crystal_system_mae_bar_chart_abc.png", dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)

    # ── Angles chart ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_df[angle_cols].plot(kind="bar", edgecolor="k", ax=ax, color=ANG_GRANITE)
    style_axes(ax, "Mean Absolute Error (°)", title_ang)
    add_group_counts(ax, counts_arr, angle_tops, labels=labels, fontsize=12)
    fig.tight_layout()
    plt.savefig(outdir / "crystal_system_mae_bar_chart_angles.png", dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)

    # ── JSON: everything shown in the graphs ─────────────────────────
    systems = list(g.index)
    systems_pretty = [s.capitalize() for s in systems]

    crystal_system_metrics = []
    for cs in systems:
        crystal_system_metrics.append({
            "crystal_system": cs,
            "crystal_system_pretty": cs.capitalize(),
            "n_reconstructions": int(g_counts.loc[cs]),
            "mae": {
                "a": _round6(g.loc[cs, "a"]),
                "b": _round6(g.loc[cs, "b"]),
                "c": _round6(g.loc[cs, "c"]),
                "alpha": _round6(g.loc[cs, "alpha"]),
                "beta": _round6(g.loc[cs, "beta"]),
                "gamma": _round6(g.loc[cs, "gamma"]),
            }
        })

    metrics_json = {
        "meta": {
            "root": str(root),
            "outdir": str(outdir),
            "symprec": float(args.symprec),
            "kmin": int(args.kmin),
            "benchmarks_included": benchmarks_included,
            "plot_titles": {
                "lengths": title_len,
                "angles": title_ang,
            },
            "palette_hex": {
                "lengths": LEN_GRANITE,
                "angles": ANG_GRANITE,
            },
            "png_dpi": PNG_DPI,
            "note": (
                "Values are mean absolute errors computed from Niggli-reduced primitive cells; "
                "crystal system labels computed on conventional standardized structures derived "
                "from the same canonicalized targets. Plot order is most symmetric to least "
                "symmetric, excluding triclinic."
            ),
        },
        "plots": {
            "lengths": {
                "x": systems_pretty,
                "n": [int(g_counts.loc[cs]) for cs in systems],
                "series": {
                    "a": [_round6(g.loc[cs, "a"]) for cs in systems],
                    "b": [_round6(g.loc[cs, "b"]) for cs in systems],
                    "c": [_round6(g.loc[cs, "c"]) for cs in systems],
                },
                "y_label": "Mean Absolute Error (Å)",
                "output_png": "crystal_system_mae_bar_chart_abc.png",
            },
            "angles": {
                "x": systems_pretty,
                "n": [int(g_counts.loc[cs]) for cs in systems],
                "series": {
                    "alpha": [_round6(g.loc[cs, "alpha"]) for cs in systems],
                    "beta": [_round6(g.loc[cs, "beta"]) for cs in systems],
                    "gamma": [_round6(g.loc[cs, "gamma"]) for cs in systems],
                },
                "y_label": "Mean Absolute Error (°)",
                "output_png": "crystal_system_mae_bar_chart_angles.png",
            },
        },
        "by_crystal_system": crystal_system_metrics,
    }

    with open(outdir / "crystal_mae_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    print(f"✓ wrote {outdir/'mae_by_crystal_system.csv'}", file=sys.stderr)
    print(f"✓ wrote {outdir/'crystal_system_mae_bar_chart_abc.png'}", file=sys.stderr)
    print(f"✓ wrote {outdir/'crystal_system_mae_bar_chart_angles.png'}", file=sys.stderr)
    print(f"✓ wrote {outdir/'crystal_mae_metrics.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()

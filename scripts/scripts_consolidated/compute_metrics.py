#!/usr/bin/env python3
"""
compute_metrics.py — compute all reconstruction metrics for one benchmark CSV.

Input : --csv <path>  (columns: id, target, prediction — POSCAR strings)
Output: metrics.json  (default: same directory as the CSV)

The output JSON is backward-compatible with bar_chart.py, rmse_bar_chart.py,
and json_to_csv.py (field names and nesting are preserved exactly).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_absolute_error

import matplotlib as mpl
mpl.use("Agg")

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import amd

# ── Crystal-system constants ──────────────────────────────────────────────────
CRYSYS_ALL = [
    "triclinic", "monoclinic", "orthorhombic",
    "tetragonal", "trigonal", "hexagonal", "cubic",
]
CRYSYS_PLOT_ORDER = [
    "cubic", "hexagonal", "trigonal",
    "tetragonal", "orthorhombic", "monoclinic",
]

# ── Pymatgen / Niggli helpers ─────────────────────────────────────────────────
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


# ── Statistical helpers ───────────────────────────────────────────────────────
def kl_divergence(p, q) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= np.sum(p)
    q /= np.sum(q)
    return float(scipy_entropy(p, q))


def emd_distance(p, q, bins=None) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= np.sum(p)
    q /= np.sum(q)
    if bins is None:
        bins = np.arange(len(p))
    return float(wasserstein_distance(bins, bins, u_weights=p, v_weights=q))


# ── AtomGen RMSE (StructureMatcher) ──────────────────────────────────────────
def compute_atomgen_rmse(df: pd.DataFrame) -> dict:
    STOL = 0.5
    matcher = StructureMatcher(stol=STOL, angle_tol=10, ltol=0.3)
    norm_rms_vals, rms_vals_ang = [], []
    n_total = n_matched = 0

    for _, mm in df.iterrows():
        try:
            s_target = reduced_structure_from_poscar_text(str(mm["target"]))
            s_pred   = reduced_structure_from_poscar_text(str(mm["prediction"]))
            n_total += 1
            rms_dist = matcher.get_rms_dist(s_pred, s_target)
            if rms_dist is not None:
                rms_ang = float(rms_dist[0])
                vol = float(s_target.lattice.volume)
                if np.isfinite(vol) and vol > 0:
                    scale = float(np.cbrt(vol))
                    if np.isfinite(scale) and scale > 0:
                        norm_rms_vals.append(rms_ang / scale)
                        rms_vals_ang.append(rms_ang)
                        n_matched += 1
        except Exception:
            continue

    if n_total == 0:
        return {
            "mean_normalized_cartesian_rms": float("nan"),
            "mean_cartesian_rms_angstrom": float("nan"),
            "match_rate": float("nan"),
            "stol": float(STOL),
            "n_matched": 0,
            "n_total": 0,
        }

    match_rate = float(n_matched) / float(n_total)

    if n_matched == 0:
        return {
            "mean_normalized_cartesian_rms": float("nan"),
            "mean_cartesian_rms_angstrom": float("nan"),
            "match_rate": round(match_rate, 6),
            "stol": float(STOL),
            "n_matched": 0,
            "n_total": int(n_total),
        }

    return {
        "mean_normalized_cartesian_rms": round(float(np.mean(norm_rms_vals)), 6),
        "mean_cartesian_rms_angstrom": round(float(np.mean(rms_vals_ang)), 6),
        "match_rate": round(match_rate, 6),
        "stol": float(STOL),
        "n_matched": int(n_matched),
        "n_total": int(n_total),
    }


# ── ccRMSE (AMD) ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=20000)
def amd_vector_from_poscar_text(poscar_text: str, k: int):
    s = reduced_structure_from_poscar_text(poscar_text)
    motif = np.asarray(s.cart_coords, dtype=np.float64)
    cell  = np.asarray(s.lattice.matrix, dtype=np.float64)
    v = amd.AMD((motif, cell), int(k))
    return tuple(np.asarray(v, dtype=np.float64).tolist())


def compute_ccrmse_amd(df: pd.DataFrame, k: int, tau: float):
    if tau <= 0:
        return float("nan"), 0
    s2 = 0.0
    n = 0
    for _, row in df.iterrows():
        try:
            v_t = np.asarray(amd_vector_from_poscar_text(str(row["target"]),     int(k)), dtype=np.float64)
            v_p = np.asarray(amd_vector_from_poscar_text(str(row["prediction"]), int(k)), dtype=np.float64)
            d = float(np.max(np.abs(v_p - v_t)))
            if not np.isfinite(d) or d < 0:
                continue
            dc = d if d <= tau else tau
            s2 += dc * dc
            n += 1
        except Exception:
            continue
    if n == 0:
        return float("nan"), 0
    return float(np.sqrt(s2 / float(n))), int(n)


# ── Crystal-system MAE ────────────────────────────────────────────────────────
def crystal_system_from_structure(s: Structure, symprec: float) -> str | None:
    try:
        sga  = SpacegroupAnalyzer(s, symprec=symprec)
        conv = sga.get_conventional_standard_structure()
        cs   = SpacegroupAnalyzer(conv, symprec=symprec).get_crystal_system()
        cs   = cs.lower() if isinstance(cs, str) else None
        return cs if cs in CRYSYS_ALL else None
    except Exception:
        return None


def _round6(x: float) -> float:
    try:
        return float(round(float(x), 6))
    except Exception:
        return float("nan")


def compute_crystal_system_mae(df: pd.DataFrame, symprec: float, kmin: int) -> dict:
    """Per-crystal-system MAE for a single benchmark DataFrame."""
    errors: dict[str, list[dict]] = defaultdict(list)

    for _, r in df.iterrows():
        try:
            t = str(r["target"])
            p = str(r["prediction"])
            s_t = reduced_structure_from_poscar_text(t)
            cs  = crystal_system_from_structure(s_t, symprec=symprec)
            if cs is None:
                continue
            ta, tb, tc, tal, tbe, tga = niggli_params_from_poscar_text(t)
            pa, pb, pc, pal, pbe, pga = niggli_params_from_poscar_text(p)
            errors[cs].append({
                "a":     abs(pa - ta),
                "b":     abs(pb - tb),
                "c":     abs(pc - tc),
                "alpha": abs(pal - tal),
                "beta":  abs(pbe - tbe),
                "gamma": abs(pga - tga),
            })
        except Exception:
            continue

    by_system = []
    for cs in CRYSYS_PLOT_ORDER:
        rows = errors.get(cs, [])
        if len(rows) < kmin:
            continue
        mae = {
            param: _round6(float(np.mean([r[param] for r in rows])))
            for param in ("a", "b", "c", "alpha", "beta", "gamma")
        }
        by_system.append({
            "crystal_system":   cs,
            "n_reconstructions": int(len(rows)),
            "mae": mae,
        })

    return {
        "by_system": by_system,
        "symprec":   float(symprec),
        "kmin":      int(kmin),
    }


# ── metrics_metadata loader ───────────────────────────────────────────────────
def find_benchmarks_dir(start: Path) -> Path | None:
    for d in [start, *start.parents]:
        if (d / "benchmarks.csv").is_file():
            return d
    return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Compute all reconstruction metrics for one benchmark CSV → metrics.json"
    )
    ap.add_argument("--csv",            required=True, help="Path to benchmark CSV (id, target, prediction)")
    ap.add_argument("--output",         default=None,  help="Output JSON path (default: metrics.json next to CSV)")
    ap.add_argument("--benchmark-name", default=None,  help="Value for benchmark_name field (default: parent dir name)")
    ap.add_argument("--tau",     type=float, default=0.5,   help="ccRMSE clamp threshold")
    ap.add_argument("--amd-k",   type=int,   default=100,   help="AMD vector length k")
    ap.add_argument("--symprec", type=float, default=0.1,   help="Symmetry tolerance for SpacegroupAnalyzer")
    ap.add_argument("--kmin",    type=int,   default=10,    help="Min structures per crystal system")
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.is_file():
        sys.exit(f"ERROR: CSV not found: {csv_path}")

    out_path = Path(args.output).resolve() if args.output else csv_path.parent / "metrics.json"
    bench_name = args.benchmark_name or csv_path.parent.name

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    for col in ("target", "prediction"):
        if col not in df.columns:
            sys.exit(f"ERROR: CSV missing required column '{col}'")

    print(f"Loaded {len(df)} rows from {csv_path.name}", file=sys.stderr)

    # ── Extract Niggli params ─────────────────────────────────────────────────
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

    def _mae(x, y):
        return float(mean_absolute_error(x, y)) if len(x) > 0 else float("nan")

    def _kld(x, y):
        return kl_divergence(x, y) if len(x) > 0 else float("nan")

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("Computing MAE / KLD ...", file=sys.stderr)
    mae_a     = _mae(x_a,     y_a)
    mae_b     = _mae(x_b,     y_b)
    mae_c     = _mae(x_c,     y_c)
    mae_alpha = _mae(x_alpha, y_alpha)
    mae_beta  = _mae(x_beta,  y_beta)
    mae_gamma = _mae(x_gamma, y_gamma)

    kld_a     = _kld(x_a,     y_a)
    kld_b     = _kld(x_b,     y_b)
    kld_c     = _kld(x_c,     y_c)
    kld_alpha = _kld(x_alpha, y_alpha)
    kld_beta  = _kld(x_beta,  y_beta)
    kld_gamma = _kld(x_gamma, y_gamma)

    print("Computing AtomGen RMSE (StructureMatcher) ...", file=sys.stderr)
    rmse_atomgen = compute_atomgen_rmse(df)

    print(f"Computing ccRMSE/AMD (k={args.amd_k}, tau={args.tau}) ...", file=sys.stderr)
    ccrmse_val, n_ccrmse = compute_ccrmse_amd(df, k=int(args.amd_k), tau=float(args.tau))

    print(f"Computing crystal-system MAE (symprec={args.symprec}, kmin={args.kmin}) ...", file=sys.stderr)
    crysys_mae = compute_crystal_system_mae(df, symprec=float(args.symprec), kmin=int(args.kmin))

    # ── Assemble JSON ─────────────────────────────────────────────────────────
    metrics: dict = {
        "benchmark_name": bench_name,
        "KLD": {
            "a": kld_a, "b": kld_b, "c": kld_c,
            "alpha": kld_alpha, "beta": kld_beta, "gamma": kld_gamma,
        },
        "MAE": {
            "average_mae": {
                "a": mae_a, "b": mae_b, "c": mae_c,
                "alpha": mae_alpha, "beta": mae_beta, "gamma": mae_gamma,
            }
        },
        "RMSE": {
            "AtomGen": rmse_atomgen
        },
        "ccRMSE": {
            "value": ccrmse_val,
            "tau":   float(args.tau),
            "amd_k": int(args.amd_k),
            "n_eval": int(n_ccrmse),
        },
        "crystal_system_mae": crysys_mae,
    }

    # Optional metrics_metadata
    benchmarks_dir = find_benchmarks_dir(csv_path.parent)
    if benchmarks_dir is not None:
        meta_path = benchmarks_dir / "metrics_metadata.json"
        if meta_path.is_file():
            try:
                with open(meta_path) as mf:
                    metrics["metrics_metadata"] = json.load(mf)
                print(f"Appended metrics_metadata from {meta_path}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: could not read metrics_metadata.json: {e}", file=sys.stderr)

    # ── Write output ──────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

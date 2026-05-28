#!/usr/bin/env python3
"""
atombench.tables — generate summary metric tables for crystal reconstruction benchmarks.

Entry point:  atombench-tables --root <job_runs_dir> --outdir <out_dir>
Importable:   from atombench.tables import collect_metrics, build_metrics_tex

Reads each benchmark subdirectory's metrics.json and writes:
  <outdir>/metrics_table.json  — structured metrics for all discovered benchmarks
  <outdir>/metrics_table.tex   — stacked booktabs LaTeX table (KLD / MAE / RMSD / ccRMSD)

Table format matches the manuscript's stacked layout, extended to however many
benchmarks are discovered. Bold marks the best value per column per dataset block
(minimum for all metrics; maximum for match rate).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from atombench._common import discover_benchmark_csvs

# ── Lattice parameter ordering ────────────────────────────────────────────────
PARAMS = ("a", "b", "c", "alpha", "beta", "gamma")

PARAM_TEX = {
    "a": "$a$", "b": "$b$", "c": "$c$",
    "alpha": r"$\alpha$", "beta": r"$\beta$", "gamma": r"$\gamma$",
}

# ── Display-name derivation ───────────────────────────────────────────────────
# Experiment names follow  <model_key>_{alex|jarvis}
# MODEL_ORDER controls row ordering within each dataset block.
MODEL_ORDER = [
    "agpt_benchmark",
    "agpt_stoich_benchmark",
    "cdvae_benchmark",
    "flowmm_benchmark",
    "mattergen_stoich_benchmark",
    "mattergen_tc_finetune_benchmark",
]

MODEL_LABELS: dict[str, str] = {
    "agpt_benchmark":                  "AtomGPT",
    "agpt_stoich_benchmark":           "AtomGPT (stoich.)",
    "cdvae_benchmark":                 "CDVAE",
    "flowmm_benchmark":                "FlowMM",
    "mattergen_stoich_benchmark":      "MatterGen (stoich.)",
    "mattergen_tc_finetune_benchmark": "MatterGen (TC-ft.)",
}

DATASET_ORDER  = ["alex", "jarvis"]
DATASET_LABELS = {"alex": "Alexandria", "jarvis": "JARVIS"}


def _model_key(exp: str) -> str:
    """Extract the model key from an experiment name (strip trailing _alex/_jarvis)."""
    for ds in DATASET_ORDER:
        if exp.endswith(f"_{ds}"):
            return exp[: -(len(ds) + 1)]
    return exp


def _dataset_key(exp: str) -> Optional[str]:
    for ds in DATASET_ORDER:
        if exp.endswith(f"_{ds}"):
            return ds
    return None


def _model_label(exp: str) -> str:
    return MODEL_LABELS.get(_model_key(exp), _model_key(exp))


def _dataset_label(exp: str) -> str:
    return DATASET_LABELS.get(_dataset_key(exp) or "", _dataset_key(exp) or exp)


# ── Metrics loading ───────────────────────────────────────────────────────────
def _load_metrics_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    text = path.read_text(errors="replace").replace("NaN", "null").replace("Infinity", "null")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_metrics(raw: dict) -> dict:
    """Flatten a single metrics.json dict into a plain metrics record."""
    kld_raw = raw.get("KLD", {})
    kld = {k: kld_raw.get(k) for k in PARAMS}
    kld_vals = [v for v in kld.values() if v is not None]
    kld["mean"] = sum(kld_vals) / len(kld_vals) if kld_vals else None

    mae_raw = raw.get("MAE", {}).get("average_mae", {})
    mae = {k: mae_raw.get(k) for k in PARAMS}
    abc_vals   = [mae[k] for k in ("a", "b", "c")           if mae[k] is not None]
    angle_vals = [mae[k] for k in ("alpha", "beta", "gamma") if mae[k] is not None]
    mae["mean_abc"]    = sum(abc_vals)   / len(abc_vals)   if abc_vals   else None
    mae["mean_angles"] = sum(angle_vals) / len(angle_vals) if angle_vals else None

    rmse_raw   = raw.get("RMSE", {}).get("AtomGen", {})
    cc_raw     = raw.get("ccRMSE", {})
    ccrmsd     = cc_raw.get("value")
    if ccrmsd is not None and math.isnan(ccrmsd):
        ccrmsd = None

    return {
        "KLD":            kld,
        "MAE":            mae,
        "RMSD":           rmse_raw.get("mean_cartesian_rms_angstrom"),
        "match_rate":     rmse_raw.get("match_rate"),
        "n_matched":      rmse_raw.get("n_matched"),
        "n_total":        rmse_raw.get("n_total"),
        "ccRMSD":         ccrmsd,
        "ccRMSD_n_eval":  cc_raw.get("n_eval", 0),
    }


def collect_metrics(path: Path) -> dict[str, dict]:
    """
    Discover benchmark CSVs under *path* and return a dict mapping experiment
    name → extracted metrics record.

    Input validation (assert-based) is handled by discover_benchmark_csvs:
    *path* must be a benchmark CSV file or a directory of benchmark CSV files.
    metrics.json is loaded from the same directory as each discovered CSV.
    """
    pairs = discover_benchmark_csvs(path)
    results: dict[str, dict] = {}
    for name, csv_path in pairs:
        mj = csv_path.parent / "metrics.json"
        raw = _load_metrics_json(mj)
        assert raw is not None, (
            f"metrics.json not found or unreadable for benchmark '{name}' "
            f"(expected at {mj}). Run `atombench {csv_path}` first to compute metrics."
        )
        results[name] = extract_metrics(raw)
    return results


# ── Ordering helper ───────────────────────────────────────────────────────────
def _sort_key(exp: str) -> tuple:
    mk = _model_key(exp)
    dk = _dataset_key(exp) or ""
    mi = MODEL_ORDER.index(mk) if mk in MODEL_ORDER else len(MODEL_ORDER)
    di = DATASET_ORDER.index(dk) if dk in DATASET_ORDER else len(DATASET_ORDER)
    return (di, mi)


def _grouped(results: dict[str, dict]) -> dict[str, list[tuple[str, dict]]]:
    """Return {dataset_key: [(exp_name, metrics), ...]} in display order."""
    grouped: dict[str, list] = {ds: [] for ds in DATASET_ORDER}
    other: list = []
    for exp in sorted(results, key=_sort_key):
        ds = _dataset_key(exp)
        if ds in grouped:
            grouped[ds].append((exp, results[exp]))
        else:
            other.append((exp, results[exp]))
    return grouped, other


# ── Bold helpers ──────────────────────────────────────────────────────────────
def _min_per_key(rows: list[dict], keys: list[str]) -> dict[str, Optional[float]]:
    best = {}
    for k in keys:
        vals = [r.get(k) for r in rows if r.get(k) is not None]
        best[k] = min(vals) if vals else None
    return best


def _max_per_key(rows: list[dict], keys: list[str]) -> dict[str, Optional[float]]:
    best = {}
    for k in keys:
        vals = [r.get(k) for r in rows if r.get(k) is not None]
        best[k] = max(vals) if vals else None
    return best


def _fmt(v: Optional[float], dp: int, best: Optional[float] = None,
         higher_is_better: bool = False) -> str:
    if v is None:
        return r"---"
    s = f"{v:.{dp}f}"
    if best is not None and abs(v - best) < 10 ** (-dp - 2):
        s = rf"\textbf{{{s}}}"
    return s


# ── LaTeX builder ─────────────────────────────────────────────────────────────
def build_metrics_tex(results: dict[str, dict]) -> str:
    grouped, other = _grouped(results)

    lines = [
        r"% Requires: \usepackage{booktabs}, \usepackage{multirow}",
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"",
    ]

    def _ds_rows(ds_key: str) -> list[tuple[str, dict]]:
        return grouped.get(ds_key, [])

    # ── Stack 1: KLD ─────────────────────────────────────────────────────────
    kld_keys = list(PARAMS) + ["mean"]
    kld_hdrs = [PARAM_TEX[k] for k in PARAMS] + [r"\textbf{mean}"]
    ncols_kld = 2 + len(kld_keys)

    lines += [
        r"% ── Lattice KLD ──────────────────────────────────────────────────",
        rf"\begin{{tabular}}{{ll{'c' * len(kld_keys)}}}",
        rf"\multicolumn{{{ncols_kld}}}{{c}}{{\textbf{{Lattice KLD}}}} \\",
        r"\hline",
        "Dataset & Model & " + " & ".join(kld_hdrs) + r" \\",
        r"\hline",
    ]
    for ds_key in DATASET_ORDER:
        rows = _ds_rows(ds_key)
        if not rows:
            continue
        ds_label = DATASET_LABELS.get(ds_key, ds_key)
        data  = [e.get("KLD", {}) for _, e in rows]
        best  = _min_per_key(data, kld_keys)
        for i, (exp, e) in enumerate(rows):
            kld   = e.get("KLD", {})
            cells = [_fmt(kld.get(k), 4, best.get(k)) for k in kld_keys]
            pfx   = ds_label if i == 0 else ""
            lines.append(f" {pfx} & {_model_label(exp)} & " + " & ".join(cells) + r" \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"", r"\vspace{2em}", r""]

    # ── Stack 2: MAE ─────────────────────────────────────────────────────────
    mae_keys = list(PARAMS) + ["mean_abc", "mean_angles"]
    mae_hdrs = (
        [PARAM_TEX[k] for k in PARAMS]
        + [r"\textbf{mean}$_{abc}$", r"\textbf{mean}$_{\alpha\beta\gamma}$"]
    )
    ncols_mae = 2 + len(mae_keys)

    lines += [
        r"% ── Lattice MAE ──────────────────────────────────────────────────",
        rf"\begin{{tabular}}{{ll{'c' * len(mae_keys)}}}",
        rf"\multicolumn{{{ncols_mae}}}{{c}}"
        r"{\textbf{Lattice MAE (\AA\ for $abc$, $^\circ$ for $\alpha\beta\gamma$)}} \\",
        r"\hline",
        "Dataset & Model & " + " & ".join(mae_hdrs) + r" \\",
        r"\hline",
    ]
    for ds_key in DATASET_ORDER:
        rows = _ds_rows(ds_key)
        if not rows:
            continue
        ds_label = DATASET_LABELS.get(ds_key, ds_key)
        data  = [e.get("MAE", {}) for _, e in rows]
        best  = _min_per_key(data, mae_keys)
        for i, (exp, e) in enumerate(rows):
            mae   = e.get("MAE", {})
            cells = [_fmt(mae.get(k), 3, best.get(k)) for k in mae_keys]
            pfx   = ds_label if i == 0 else ""
            lines.append(f" {pfx} & {_model_label(exp)} & " + " & ".join(cells) + r" \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"", r"\vspace{2em}", r""]

    # ── Stack 3: RMSD + match rate ────────────────────────────────────────────
    lines += [
        r"% ── Coordinate RMSD + Match Rate ────────────────────────────────",
        r"\begin{tabular}{llcc}",
        r"\multicolumn{4}{c}{\textbf{Coordinate Reconstruction}} \\",
        r"\hline",
        r"Dataset & Model & RMSD (\AA) & Match rate \\",
        r"\hline",
    ]
    for ds_key in DATASET_ORDER:
        rows = _ds_rows(ds_key)
        if not rows:
            continue
        ds_label   = DATASET_LABELS.get(ds_key, ds_key)
        best_rmsd  = _min_per_key([e for _, e in rows], ["RMSD"])["RMSD"]
        best_mr    = _max_per_key([e for _, e in rows], ["match_rate"])["match_rate"]
        for i, (exp, e) in enumerate(rows):
            rmsd = _fmt(e.get("RMSD"),       4, best_rmsd)
            mr   = _fmt(e.get("match_rate"), 4, best_mr)
            pfx  = ds_label if i == 0 else ""
            lines.append(rf" {pfx} & {_model_label(exp)} & {rmsd} & {mr} \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"", r"\vspace{2em}", r""]

    # ── Stack 4: ccRMSD ───────────────────────────────────────────────────────
    lines += [
        r"% ── ccRMSD (AMD-RMSE, k=100) ────────────────────────────────────",
        r"\begin{tabular}{llc}",
        r"\multicolumn{3}{c}{\textbf{ccRMSD (AMD-RMSE, $k=100$)}} \\",
        r"\hline",
        r"Dataset & Model & ccRMSD \\",
        r"\hline",
    ]
    for ds_key in DATASET_ORDER:
        rows = _ds_rows(ds_key)
        if not rows:
            continue
        ds_label = DATASET_LABELS.get(ds_key, ds_key)
        best_cc  = _min_per_key([e for _, e in rows], ["ccRMSD"])["ccRMSD"]
        for i, (exp, e) in enumerate(rows):
            cc  = _fmt(e.get("ccRMSD"), 4, best_cc)
            pfx = ds_label if i == 0 else ""
            lines.append(rf" {pfx} & {_model_label(exp)} & {cc} \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r""]

    lines += [
        r"\caption{Reconstruction metrics for all benchmark instances."
        r" Stacked blocks: lattice KLD, lattice MAE (lengths in \AA, angles in degrees),"
        r" coordinate RMSD and match rate, and ccRMSD (AMD-based RMSE, $k{=}100$)."
        r" Bold marks the best value per column per dataset block."
        r" Match rate is bolded at its maximum; all other metrics at their minimum.}",
        r"\label{tab:reconstruction_all}",
        r"\end{table*}",
    ]

    return "\n".join(lines) + "\n"


# ── CLI ───────────────────────────────────────────────────────────────────────
@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--outdir", "-o", default=None, type=click.Path(path_type=Path),
              help="Output directory for metrics_table.{json,tex}. Defaults to PATH's directory.")
def main(path: Path, outdir: Optional[Path]) -> None:
    """Generate JSON + LaTeX metrics summary tables from pre-computed metrics.json files.

    \b
    PATH can be:
      • a single benchmark CSV file
      • a directory of benchmark CSV files (flat or one per subdirectory)

    For each benchmark CSV, metrics.json must already exist in the same directory
    (produced by `atombench PATH`).  Writes:
      <outdir>/metrics_table.json
      <outdir>/metrics_table.tex
    """
    path = path.resolve()
    if outdir is None:
        outdir = path if path.is_dir() else path.parent
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = collect_metrics(path)

    click.echo(f"Found {len(results)} benchmark(s) with metrics.")

    out_json = outdir / "metrics_table.json"
    out_tex  = outdir / "metrics_table.tex"
    out_csv  = outdir / "epic_metrics.csv"

    out_json.write_text(json.dumps(results, indent=2) + "\n")
    out_tex.write_text(build_metrics_tex(results))

    # Flat CSV: one row per benchmark, all metrics as dot-separated columns
    raw_records = []
    for name, e in results.items():
        rec = {"benchmark_name": name}
        rec.update(pd.json_normalize(e, sep=".").iloc[0].to_dict())
        raw_records.append(rec)
    df_flat = pd.DataFrame(raw_records)
    cols = ["benchmark_name"] + [c for c in df_flat.columns if c != "benchmark_name"]
    df_flat[cols].to_csv(out_csv, index=False)

    click.echo(f"Wrote {out_json}")
    click.echo(f"Wrote {out_tex}")
    click.echo(f"Wrote {out_csv}")

    click.echo("\nSummary (match_rate | RMSD | ccRMSD | KLD_mean):")
    for exp in sorted(results, key=_sort_key):
        e    = results[exp]
        mr   = e.get("match_rate");  mr_s   = f"{mr:.4f}"   if mr   is not None else "---"
        rmsd = e.get("RMSD");        rmsd_s = f"{rmsd:.4f}" if rmsd is not None else "---"
        cc   = e.get("ccRMSD");      cc_s   = f"{cc:.4f}"   if cc   is not None else "---"
        km   = e.get("KLD", {}).get("mean")
        km_s = f"{km:.4f}" if km is not None else "---"
        click.echo(f"  {exp}: MR={mr_s}  RMSD={rmsd_s}  cc={cc_s}  KLD={km_s}")


if __name__ == "__main__":
    main()

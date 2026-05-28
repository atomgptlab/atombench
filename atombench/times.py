#!/usr/bin/env python3
"""
atombench.times — harvest GPU wall-clock times and epoch counts for benchmarks.

Entry point:  atombench-times <path> [--outdir DIR]

Reads SLURM output logs and model config files to produce:
  <outdir>/computational_costs.json  — per-experiment wall-clock times + normalised metrics
  <outdir>/computational_costs.tex   — booktabs LaTeX table ready for Overleaf

Two timing parsers:
  elapsed_footer  — cdvae / flowmm / mattergen: reads  Elapsed: <n>s  footer line
  atomgpt         — atomgpt family: reads  train_runtime  and  Eval time taken  lines

Epoch counts are read from each model's config / YAML files (paths relative to
the repo root, which is inferred as the parent of the job_runs directory).

Test-structure counts come from the benchmark CSV files themselves.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import click

from atombench._common import discover_benchmark_csvs


# ── Experiment → SLURM log and epoch-source config ───────────────────────────
# All file paths are expressed relative to ROOT (the repo root, i.e. the
# parent of the job_runs directory passed as input).
#
# epoch_source:
#   method "json_key"        — json.load(file)[key]
#   method "yaml_flat"       — grep top-level  key: <int>
#   method "yaml_section:<S>"— find max_epochs inside the named YAML block
EXP_CONFIGS: dict = {
    "agpt_benchmark_alex": {
        "parser": "atomgpt",
        "log": "slurm_alex_atomgpt_train.out",
        "epoch_source": {"file": "job_runs/agpt_benchmark_alex/config.json",
                         "method": "json_key", "key": "num_epochs"},
    },
    "agpt_benchmark_jarvis": {
        "parser": "atomgpt",
        "log": "slurm_jarvis_atomgpt_train.out",
        "epoch_source": {"file": "job_runs/agpt_benchmark_jarvis/config.json",
                         "method": "json_key", "key": "num_epochs"},
    },
    "agpt_stoich_benchmark_alex": {
        "parser": "atomgpt",
        "log": "slurm_alex_agpt_stoich_train.out",
        "epoch_source": {"file": "job_runs/agpt_stoich_benchmark_alex/config.json",
                         "method": "json_key", "key": "num_epochs"},
    },
    "agpt_stoich_benchmark_jarvis": {
        "parser": "atomgpt",
        "log": "slurm_jarvis_agpt_stoich_train.out",
        "epoch_source": {"file": "job_runs/agpt_stoich_benchmark_jarvis/config.json",
                         "method": "json_key", "key": "num_epochs"},
    },
    "cdvae_benchmark_alex": {
        "parser": "elapsed_footer",
        "train": ["slurm_alex_cdvae_train.out"],
        "infer":  ["slurm_alex_cdvae_infer.out"],
        "epoch_source": {"file": "models/cdvae/conf/data/alexandria.yaml",
                         "method": "yaml_flat", "key": "train_max_epochs"},
    },
    "cdvae_benchmark_jarvis": {
        "parser": "elapsed_footer",
        "train": ["slurm_jarvis_cdvae_train.out"],
        "infer":  ["slurm_jarvis_cdvae_infer.out"],
        "epoch_source": {"file": "models/cdvae/conf/data/supercon.yaml",
                         "method": "yaml_flat", "key": "train_max_epochs"},
    },
    "flowmm_benchmark_alex": {
        "parser": "elapsed_footer",
        "train": ["slurm_alex_flowmm_train.out"],
        "infer":  ["slurm_alex_flowmm_infer.out"],
        "epoch_source": {"file": "job_runs/flowmm_benchmark_alex/outputs/.hydra/config.yaml",
                         "method": "yaml_section:pl_trainer"},
    },
    "flowmm_benchmark_jarvis": {
        "parser": "elapsed_footer",
        "train": ["slurm_jarvis_flowmm_train.out"],
        "infer":  ["slurm_jarvis_flowmm_infer.out"],
        "epoch_source": {"file": "job_runs/flowmm_benchmark_jarvis/outputs/.hydra/config.yaml",
                         "method": "yaml_section:pl_trainer"},
    },
    "mattergen_stoich_benchmark_alex": {
        "parser": "elapsed_footer",
        "train": ["slurm_alex_mattergen_stoich_train.out"],
        "infer":  ["slurm_alex_mattergen_stoich_infer.out"],
        "epoch_source": {"file": "job_runs/mattergen_stoich_benchmark_alex/outputs/.hydra/config.yaml",
                         "method": "yaml_section:trainer"},
    },
    "mattergen_stoich_benchmark_jarvis": {
        "parser": "elapsed_footer",
        "train": ["slurm_jarvis_mattergen_stoich_train.out"],
        "infer":  ["slurm_jarvis_mattergen_stoich_infer.out"],
        "epoch_source": {"file": "job_runs/mattergen_stoich_benchmark_jarvis/outputs/.hydra/config.yaml",
                         "method": "yaml_section:trainer"},
    },
    "mattergen_tc_finetune_benchmark_alex": {
        "parser": "elapsed_footer",
        "train": ["slurm_alex_mattergen_tc_finetune_train.out"],
        "infer":  ["slurm_alex_mattergen_tc_finetune_infer.out"],
        "epoch_source": {"file": "job_runs/mattergen_tc_finetune_benchmark_alex/outputs/.hydra/config.yaml",
                         "method": "yaml_section:trainer"},
    },
    "mattergen_tc_finetune_benchmark_jarvis": {
        "parser": "elapsed_footer",
        "train": ["slurm_jarvis_mattergen_tc_finetune_train.out"],
        "infer":  ["slurm_jarvis_mattergen_tc_finetune_infer.out"],
        "epoch_source": {"file": "job_runs/mattergen_tc_finetune_benchmark_jarvis/outputs/.hydra/config.yaml",
                         "method": "yaml_section:trainer"},
    },
}

# ── LaTeX display labels ──────────────────────────────────────────────────────
MODEL_ORDER = [
    "agpt_benchmark",
    "agpt_stoich_benchmark",
    "cdvae_benchmark",
    "flowmm_benchmark",
    "mattergen_stoich_benchmark",
    "mattergen_tc_finetune_benchmark",
]
MODEL_LABELS = {
    "agpt_benchmark":                  "AtomGPT",
    "agpt_stoich_benchmark":           "AtomGPT (stoich.)",
    "cdvae_benchmark":                 "CDVAE",
    "flowmm_benchmark":                "FlowMM",
    "mattergen_stoich_benchmark":      "MatterGen (stoich.)",
    "mattergen_tc_finetune_benchmark": "MatterGen (TC-ft.)",
}
DATASET_LABELS = {"alex": "Alexandria", "jarvis": "JARVIS"}
DATASET_ORDER  = ["alex", "jarvis"]


# ── Epoch-count parsing ───────────────────────────────────────────────────────
def _parse_epochs(source: dict, root: Path) -> Optional[int]:
    p = root / source["file"]
    if not p.exists():
        return None
    method = source["method"]
    if method == "json_key":
        try:
            return int(json.loads(p.read_text())[source["key"]])
        except (KeyError, ValueError, json.JSONDecodeError):
            return None
    text = p.read_text(errors="replace")
    if method == "yaml_flat":
        m = re.search(rf"^{re.escape(source['key'])}:\s*(\d+)", text, re.MULTILINE)
        return int(m.group(1)) if m else None
    if method.startswith("yaml_section:"):
        section = method.split(":", 1)[1]
        m_sec = re.search(rf"^\s*{re.escape(section)}:\s*$", text, re.MULTILINE)
        if not m_sec:
            return None
        block = "\n".join(text[m_sec.end():].splitlines()[:30])
        m_ep  = re.search(r"^\s+max_epochs:\s+(\d+)", block, re.MULTILINE)
        return int(m_ep.group(1)) if m_ep else None
    return None


# ── Timing parsers ────────────────────────────────────────────────────────────
def _read(p: Path) -> Optional[str]:
    return p.read_text(errors="replace") if p.exists() else None


def _elapsed_seconds(log_path: Path) -> Optional[int]:
    text = _read(log_path)
    if not text:
        return None
    m = re.search(r"^Elapsed:\s+(\d+)s", text, re.MULTILINE)
    return int(m.group(1)) if m else None


def _parse_atomgpt_log(log_path: Path) -> dict:
    out = {"train_s": None, "infer_s": None, "n_test_from_log": None}
    text = _read(log_path)
    if not text:
        return out
    m = re.search(r"'train_runtime':\s*([\d.]+)", text)
    if m:
        out["train_s"] = float(m.group(1))
    m = re.search(r"^Eval time taken:\s*([\d.]+)", text, re.MULTILINE)
    if m:
        out["infer_s"] = float(m.group(1))
    m = re.search(r"^Testing\s*\n\s*(\d+)", text, re.MULTILINE)
    if m:
        out["n_test_from_log"] = int(m.group(1))
    return out


def _count_csv_rows(csv_path: Path) -> Optional[int]:
    if not csv_path.exists():
        return None
    lines = [ln for ln in csv_path.read_text(errors="replace").splitlines() if ln.strip()]
    return max(0, len(lines) - 1)


# ── Core aggregation ──────────────────────────────────────────────────────────
def _rnd(x: Optional[float], dp: int = 4) -> Optional[float]:
    return round(x, dp) if x is not None else None


def collect_times(root: Path, csv_map: dict[str, Path]) -> dict:
    """
    Collect timing for each experiment in EXP_CONFIGS.

    root     — repo root (parent of job_runs); SLURM logs and config files
               are resolved relative to this.
    csv_map  — {exp_name: csv_path} from discover_benchmark_csvs; used to
               count test structures from the actual benchmark CSV.
    """
    results = {}
    for exp, cfg in EXP_CONFIGS.items():
        e: dict = {}
        num_epochs = _parse_epochs(cfg.get("epoch_source", {}), root) if "epoch_source" in cfg else None
        e["num_epochs"] = num_epochs

        n_test = _count_csv_rows(csv_map[exp]) if exp in csv_map else None

        if cfg["parser"] == "atomgpt":
            p = _parse_atomgpt_log(root / cfg["log"])
            e["train_s"] = _rnd(p["train_s"], 1)
            e["train_h"] = _rnd(p["train_s"] / 3600 if p["train_s"] is not None else None)
            e["infer_s"] = _rnd(p["infer_s"], 1)
            e["infer_h"] = _rnd(p["infer_s"] / 3600 if p["infer_s"] is not None else None)
            if p["train_s"] is not None and p["infer_s"] is not None:
                total = p["train_s"] + p["infer_s"]
                e["total_s"] = _rnd(total, 1)
                e["total_h"] = _rnd(total / 3600)
            else:
                e["total_s"] = e["total_h"] = None
            e["train_s_per_epoch"] = (
                _rnd(p["train_s"] / num_epochs, 1)
                if p["train_s"] is not None and num_epochs else None
            )
            n_test = n_test or p["n_test_from_log"]
            e["num_test_structures"] = n_test
            e["infer_s_per_structure"] = (
                _rnd(p["infer_s"] / n_test)
                if p["infer_s"] is not None and n_test else None
            )

        else:  # elapsed_footer
            train_raw = [_elapsed_seconds(root / f) for f in cfg.get("train", [])]
            infer_raw = [_elapsed_seconds(root / f) for f in cfg.get("infer",  [])]
            train_s = sum(train_raw) if train_raw and all(v is not None for v in train_raw) else None
            infer_s = sum(infer_raw) if infer_raw and all(v is not None for v in infer_raw) else None
            e["train_s"] = train_s
            e["train_h"] = _rnd(train_s / 3600 if train_s is not None else None)
            e["infer_s"] = infer_s
            e["infer_h"] = _rnd(infer_s / 3600 if infer_s is not None else None)
            if train_s is not None and infer_s is not None:
                e["total_s"] = train_s + infer_s
                e["total_h"] = _rnd((train_s + infer_s) / 3600)
            else:
                e["total_s"] = e["total_h"] = None
            e["train_s_per_epoch"] = (
                _rnd(train_s / num_epochs, 1)
                if train_s is not None and num_epochs else None
            )
            e["num_test_structures"] = n_test
            e["infer_s_per_structure"] = (
                _rnd(infer_s / n_test)
                if infer_s is not None and n_test else None
            )

        results[exp] = e
    return results


# ── LaTeX table ───────────────────────────────────────────────────────────────
def _fmth(h: Optional[float]) -> str:
    return f"{h:.2f}" if h is not None else r"---"

def _fmts(s: Optional[float]) -> str:
    return f"{s:.1f}" if s is not None else r"---"


def build_latex(results: dict) -> str:
    lines = [
        r"% Requires: \usepackage{booktabs}, \usepackage{multirow}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{GPU wall-clock time per benchmark experiment."
        r" Train/epoch and Infer/struct are normalised by configured max epochs"
        r" and number of test structures, respectively.}",
        r"\label{tab:compute_costs}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Model & Dataset & Train (h) & Infer (h) & Total (h)"
        r" & Epochs & Train/epoch (s) & Infer/struct (s) \\",
        r"\midrule",
    ]
    for i, model_key in enumerate(MODEL_ORDER):
        label = MODEL_LABELS[model_key]
        for j, ds_key in enumerate(DATASET_ORDER):
            exp = f"{model_key}_{ds_key}"
            e   = results.get(exp, {})
            model_cell = rf"\multirow{{2}}{{*}}{{{label}}}" if j == 0 else ""
            lines.append(
                f"{model_cell} & {DATASET_LABELS[ds_key]}"
                f" & {_fmth(e.get('train_h'))}"
                f" & {_fmth(e.get('infer_h')) if 'infer_h' in e else 'N/A'}"
                f" & {_fmth(e.get('total_h'))}"
                f" & {e['num_epochs'] if e.get('num_epochs') else '---'}"
                f" & {_fmts(e.get('train_s_per_epoch'))}"
                f" & {_fmts(e.get('infer_s_per_structure'))} \\\\"
            )
        if i < len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ── CLI ───────────────────────────────────────────────────────────────────────
@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--outdir", "-o", default=None, type=click.Path(path_type=Path),
              help="Output directory for computational_costs.{json,tex}. "
                   "Defaults to PATH's directory.")
def main(path: Path, outdir: Optional[Path]) -> None:
    """Harvest GPU wall-clock times and epoch counts for benchmark experiments.

    \b
    PATH can be:
      • a single benchmark CSV file
      • a directory of benchmark CSV files (flat or one per subdirectory)

    SLURM output logs and model config files are resolved relative to the
    parent of PATH (the repo root when PATH is the job_runs directory).

    Writes:
      <outdir>/computational_costs.json
      <outdir>/computational_costs.tex
    """
    path = path.resolve()
    root = path.parent if path.is_dir() else path.parent.parent
    if outdir is None:
        outdir = path if path.is_dir() else path.parent
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pairs   = discover_benchmark_csvs(path)
    csv_map = {name: csv_path for name, csv_path in pairs}

    results = collect_times(root, csv_map)

    out_json = outdir / "computational_costs.json"
    out_tex  = outdir / "computational_costs.tex"
    out_json.write_text(json.dumps(results, indent=2) + "\n")
    out_tex.write_text(build_latex(results))

    click.echo(f"Wrote {out_json}")
    click.echo(f"Wrote {out_tex}")

    click.echo("\nSummary:")
    for exp, e in results.items():
        total_h = e.get("total_h")
        tag = f"{total_h:.2f}h" if total_h is not None else "N/A"
        n_ep    = e.get("num_epochs")
        per_ep  = e.get("train_s_per_epoch")
        per_str = e.get("infer_s_per_structure")
        extra = f", {n_ep} epochs" if n_ep else ""
        if per_ep  is not None: extra += f", {per_ep:.1f}s/epoch"
        if per_str is not None: extra += f", {per_str:.2f}s/struct"
        click.echo(f"  {exp}: {tag}{extra}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Harvest wall-clock GPU times from SLURM output logs.

Two timing parsers:
  elapsed_footer  — cdvae/flowmm/mattergen: reads  Elapsed: <n>s  from job-script footer
  atomgpt         — atomgpt family: reads  train_runtime  and  Eval time taken  lines

Epoch counts are read from each model's config/YAML files.

Writes:
  job_runs/computational_costs.json  — per-experiment times + normalized metrics
  job_runs/computational_costs.tex   — booktabs LaTeX table ready for Overleaf
"""

import json
import re
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

# ── Experiment configuration ──────────────────────────────────────────────────
# epoch_source keys:
#   file      — path relative to ROOT
#   method    — "json_key"   : json.load(file)[key]
#               "yaml_flat"  : grep top-level  key: <int>
#               "yaml_section: <SECTION>": find max_epochs under named YAML block
# timing keys for elapsed_footer experiments:
#   train/infer — lists of SLURM log paths relative to ROOT
# timing key for atomgpt experiments:
#   log — single SLURM log containing train_runtime + Eval time taken
EXP_CONFIGS = {
    "agpt_benchmark_alex": {
        "parser": "atomgpt",
        "log": "slurm_alex_atomgpt_train.out",
        "epoch_source": {
            "file":   "job_runs/agpt_benchmark_alex/config.json",
            "method": "json_key",
            "key":    "num_epochs",
        },
    },
    "agpt_benchmark_jarvis": {
        "parser": "atomgpt",
        "log": "slurm_jarvis_atomgpt_train.out",
        "epoch_source": {
            "file":   "job_runs/agpt_benchmark_jarvis/config.json",
            "method": "json_key",
            "key":    "num_epochs",
        },
    },
    "agpt_stoich_benchmark_alex": {
        "parser": "atomgpt",
        "log": "slurm_alex_agpt_stoich_train.out",
        "epoch_source": {
            "file":   "job_runs/agpt_stoich_benchmark_alex/config.json",
            "method": "json_key",
            "key":    "num_epochs",
        },
    },
    "agpt_stoich_benchmark_jarvis": {
        "parser": "atomgpt",
        "log": "slurm_jarvis_agpt_stoich_train.out",
        "epoch_source": {
            "file":   "job_runs/agpt_stoich_benchmark_jarvis/config.json",
            "method": "json_key",
            "key":    "num_epochs",
        },
    },
    "cdvae_benchmark_alex": {
        "parser": "elapsed_footer",
        "train": ["slurm_alex_cdvae_train.out"],
        "infer":  ["slurm_alex_cdvae_infer.out"],
        "epoch_source": {
            "file":   "models/cdvae/conf/data/alexandria.yaml",
            "method": "yaml_flat",
            "key":    "train_max_epochs",
        },
    },
    "cdvae_benchmark_jarvis": {
        "parser": "elapsed_footer",
        "train": ["slurm_jarvis_cdvae_train.out"],
        "infer":  ["slurm_jarvis_cdvae_infer.out"],
        "epoch_source": {
            "file":   "models/cdvae/conf/data/supercon.yaml",
            "method": "yaml_flat",
            "key":    "train_max_epochs",
        },
    },
    "flowmm_benchmark_alex": {
        "parser": "elapsed_footer",
        "train": ["slurm_alex_flowmm_train.out"],
        "infer":  ["slurm_alex_flowmm_infer.out"],
        "epoch_source": {
            "file":   "job_runs/flowmm_benchmark_alex/outputs/.hydra/config.yaml",
            "method": "yaml_section:pl_trainer",
        },
    },
    "flowmm_benchmark_jarvis": {
        "parser": "elapsed_footer",
        "train": ["slurm_jarvis_flowmm_train.out"],
        "infer":  ["slurm_jarvis_flowmm_infer.out"],
        "epoch_source": {
            "file":   "job_runs/flowmm_benchmark_jarvis/outputs/.hydra/config.yaml",
            "method": "yaml_section:pl_trainer",
        },
    },
    "mattergen_stoich_benchmark_alex": {
        "parser": "elapsed_footer",
        "train": ["slurm_alex_mattergen_stoich_train.out"],
        "infer":  ["slurm_alex_mattergen_stoich_infer.out"],
        "epoch_source": {
            "file":   "job_runs/mattergen_stoich_benchmark_alex/outputs/.hydra/config.yaml",
            "method": "yaml_section:trainer",
        },
    },
    "mattergen_stoich_benchmark_jarvis": {
        "parser": "elapsed_footer",
        "train": ["slurm_jarvis_mattergen_stoich_train.out"],
        "infer":  ["slurm_jarvis_mattergen_stoich_infer.out"],
        "epoch_source": {
            "file":   "job_runs/mattergen_stoich_benchmark_jarvis/outputs/.hydra/config.yaml",
            "method": "yaml_section:trainer",
        },
    },
    "mattergen_tc_finetune_benchmark_alex": {
        "parser": "elapsed_footer",
        "train": ["slurm_alex_mattergen_tc_finetune_train.out"],
        "infer":  ["slurm_alex_mattergen_tc_finetune_infer.out"],
        "epoch_source": {
            "file":   "job_runs/mattergen_tc_finetune_benchmark_alex/outputs/.hydra/config.yaml",
            "method": "yaml_section:trainer",
        },
    },
    "mattergen_tc_finetune_benchmark_jarvis": {
        "parser": "elapsed_footer",
        "train": ["slurm_jarvis_mattergen_tc_finetune_train.out"],
        "infer":  ["slurm_jarvis_mattergen_tc_finetune_infer.out"],
        "epoch_source": {
            "file":   "job_runs/mattergen_tc_finetune_benchmark_jarvis/outputs/.hydra/config.yaml",
            "method": "yaml_section:trainer",
        },
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
def parse_epochs(source: dict) -> Optional[int]:
    """Read epoch count from a model config file per the epoch_source spec."""
    p = ROOT / source["file"]
    if not p.exists():
        return None
    method = source["method"]

    if method == "json_key":
        try:
            return int(json.loads(p.read_text())[source["key"]])
        except (KeyError, ValueError):
            return None

    text = p.read_text(errors="replace")

    if method == "yaml_flat":
        # top-level key: e.g.  train_max_epochs: 100
        m = re.search(rf"^{re.escape(source['key'])}:\s*(\d+)", text, re.MULTILINE)
        return int(m.group(1)) if m else None

    if method.startswith("yaml_section:"):
        # find max_epochs under a named YAML block, e.g.
        #   trainer:          <- section header (zero-indent)
        #     max_epochs: 200
        # or
        #   pl_trainer:       <- section header (any indent)
        #     max_epochs: 10
        section = method.split(":", 1)[1]
        m_sec = re.search(
            rf"^\s*{re.escape(section)}:\s*$", text, re.MULTILINE
        )
        if not m_sec:
            return None
        # search the 30 lines that follow the section header
        block = "\n".join(text[m_sec.end():].splitlines()[:30])
        m_ep = re.search(r"^\s+max_epochs:\s+(\d+)", block, re.MULTILINE)
        return int(m_ep.group(1)) if m_ep else None

    return None


# ── Timing parsers ────────────────────────────────────────────────────────────
def _read(p: Path) -> Optional[str]:
    return p.read_text(errors="replace") if p.exists() else None


def parse_elapsed_seconds(log_path: Path) -> Optional[int]:
    """Return seconds from 'Elapsed: <n>s ...' footer, or None if absent."""
    text = _read(log_path)
    if not text:
        return None
    m = re.search(r"^Elapsed:\s+(\d+)s", text, re.MULTILINE)
    return int(m.group(1)) if m else None


def parse_atomgpt_log(log_path: Path) -> dict:
    """
    Parse an AtomGPT SLURM log.  Returns a dict with (possibly None) keys:
      train_s, infer_s, num_test_structures
    """
    out = {"train_s": None, "infer_s": None, "num_test_structures": None}
    text = _read(log_path)
    if not text:
        return out

    # HuggingFace Trainer summary: {'train_runtime': XXXX.X, ...}
    m = re.search(r"'train_runtime':\s*([\d.]+)", text)
    if m:
        out["train_s"] = float(m.group(1))

    # AtomGPT inference loop total: "Eval time taken: XXXX.X"
    m = re.search(r"^Eval time taken:\s*([\d.]+)", text, re.MULTILINE)
    if m:
        out["infer_s"] = float(m.group(1))

    # Test-set size: "Testing\n <n>"
    m = re.search(r"^Testing\s*\n\s*(\d+)", text, re.MULTILINE)
    if m:
        out["num_test_structures"] = int(m.group(1))

    return out


# ── Benchmark CSV row counter (test-structure count for elapsed_footer exps) ──
def count_benchmark_rows(exp: str) -> Optional[int]:
    for subdir in ("", "saved/"):
        p = ROOT / "job_runs" / exp / f"{subdir}AI-AtomGen-prop-dft_3d-test-rmse.csv"
        if p.exists():
            lines = [l for l in p.read_text(errors="replace").splitlines() if l.strip()]
            return max(0, len(lines) - 1)   # subtract header
    return None


# ── Core aggregation ──────────────────────────────────────────────────────────
def _rnd(x: Optional[float], dp: int = 4) -> Optional[float]:
    return round(x, dp) if x is not None else None


def collect_times() -> dict:
    results = {}
    for exp, cfg in EXP_CONFIGS.items():
        e: dict = {}

        # ── epoch count (from config file) ─────────────────────────────────
        num_epochs = parse_epochs(cfg["epoch_source"]) if "epoch_source" in cfg else None
        e["num_epochs"] = num_epochs

        if cfg["parser"] == "atomgpt":
            p = parse_atomgpt_log(ROOT / cfg["log"])
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
                if p["train_s"] is not None and num_epochs
                else None
            )

            n_test = p["num_test_structures"] or count_benchmark_rows(exp)
            e["num_test_structures"] = n_test
            e["infer_s_per_structure"] = (
                _rnd(p["infer_s"] / n_test)
                if p["infer_s"] is not None and n_test
                else None
            )

        else:  # elapsed_footer
            train_raw = [parse_elapsed_seconds(ROOT / f) for f in cfg.get("train", [])]
            infer_raw = [parse_elapsed_seconds(ROOT / f) for f in cfg.get("infer",  [])]

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
                if train_s is not None and num_epochs
                else None
            )

            n_test = count_benchmark_rows(exp)
            e["num_test_structures"] = n_test
            e["infer_s_per_structure"] = (
                _rnd(infer_s / n_test)
                if infer_s is not None and n_test
                else None
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
        r" Train/epoch and Infer/struct are normalized by configured max epochs"
        r" and number of test structures, respectively.}",
        r"\label{tab:compute_costs}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        (r"Model & Dataset & Train (h) & Infer (h) & Total (h)"
         r" & Epochs & Train/epoch (s) & Infer/struct (s) \\"),
        r"\midrule",
    ]

    for i, model_key in enumerate(MODEL_ORDER):
        label = MODEL_LABELS[model_key]
        for j, ds_key in enumerate(DATASET_ORDER):
            exp = f"{model_key}_{ds_key}"
            e = results.get(exp, {})

            train_h  = _fmth(e.get("train_h"))
            infer_h  = _fmth(e.get("infer_h")) if "infer_h" in e else r"N/A"
            total_h  = _fmth(e.get("total_h"))
            n_ep     = str(e["num_epochs"]) if e.get("num_epochs") else r"---"
            trn_ep   = _fmts(e.get("train_s_per_epoch"))
            inf_str  = _fmts(e.get("infer_s_per_structure"))
            ds_label = DATASET_LABELS[ds_key]

            model_cell = (
                rf"\multirow{{2}}{{*}}{{{label}}}" if j == 0 else ""
            )
            lines.append(
                f"{model_cell} & {ds_label}"
                f" & {train_h} & {infer_h} & {total_h}"
                f" & {n_ep} & {trn_ep} & {inf_str} \\\\"
            )

        if i < len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    out_json = ROOT / "job_runs" / "computational_costs.json"
    out_tex  = ROOT / "job_runs" / "computational_costs.tex"

    results = collect_times()

    out_json.write_text(json.dumps(results, indent=2) + "\n")
    out_tex.write_text(build_latex(results))

    print(f"Wrote {out_json.relative_to(ROOT)}")
    print(f"Wrote {out_tex.relative_to(ROOT)}")

    print("\nSummary:")
    for exp, e in results.items():
        total_h = e.get("total_h")
        tag = f"{total_h:.2f}h total" if total_h is not None else "N/A (missing timing)"
        per_ep  = e.get("train_s_per_epoch")
        per_str = e.get("infer_s_per_structure")
        n_ep    = e.get("num_epochs")
        extra = f", {n_ep} epochs" if n_ep else ""
        if per_ep  is not None: extra += f", {per_ep:.1f}s/epoch"
        if per_str is not None: extra += f", {per_str:.2f}s/struct"
        print(f"  {exp}: {tag}{extra}")


if __name__ == "__main__":
    main()

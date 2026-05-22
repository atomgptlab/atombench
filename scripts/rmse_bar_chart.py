#!/usr/bin/env python3
import json
import numpy as np
import sys
from pathlib import Path

# Non-interactive backend + font to match other charts
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['font.family'] = 'serif'

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

ROOT = Path.cwd()
print(f"DEBUG: Running RMSE script in {ROOT}", file=sys.stderr)

# ───────────────────── ingest metrics.json ─────────────────────
rows = []
for subdir in sorted(ROOT.iterdir()):
    if not subdir.is_dir():
        continue
    mfp = subdir / "metrics.json"
    if not mfp.is_file():
        print(f"⚠️  no metrics.json in {subdir.name} – skipped", file=sys.stderr)
        continue
    with mfp.open() as fh:
        rec = json.load(fh)
    rec.setdefault("benchmark_name", subdir.name)
    # flatten so we can access RMSE.AtomGen directly
    rows.append(pd.json_normalize(rec, sep=".", max_level=3).iloc[0].to_dict())
    print(f"DEBUG: Loaded metrics for {rec['benchmark_name']}", file=sys.stderr)

if not rows:
    print("ERROR: No metrics.json files found – exiting", file=sys.stderr)
    sys.exit(1)

df = pd.DataFrame(rows)

# ───────────────────── pretty names / model parsing ─────────────────────
bnchmk_name_dict = {
    "agpt_benchmark_alex":              "AtomGPT Alexandria",
    "agpt_benchmark_jarvis":            "AtomGPT JARVIS",
    "cdvae_benchmark_alex":             "CDVAE Alexandria",
    "cdvae_benchmark_jarvis":           "CDVAE JARVIS",
    "flowmm_benchmark_alex":            "FlowMM Alexandria",
    "flowmm_benchmark_jarvis":          "FlowMM JARVIS",
    "mattergen_benchmark_alex":         "MatterGen Finetuned Alexandria",
    "mattergen_benchmark_jarvis":       "MatterGen Finetuned JARVIS",
    "mattergen_base_benchmark_alex":         "MatterGen Base Alexandria",
    "mattergen_base_benchmark_jarvis":       "MatterGen Base JARVIS",
    "mattergen_stoich_benchmark_alex":       "MatterGen Stoich Alexandria",
    "mattergen_stoich_benchmark_jarvis":     "MatterGen Stoich JARVIS",
    "mattergen_tc_finetune_benchmark_alex":  "MatterGen TC+Stoich Alexandria",
    "mattergen_tc_finetune_benchmark_jarvis":"MatterGen TC+Stoich JARVIS",
}

def infer_model(name: str) -> str:
    name = name.lower()
    if name.startswith("agpt_"):                  return "AtomGPT"
    if name.startswith("cdvae_"):                 return "CDVAE"
    if name.startswith("flowmm_"):                return "FlowMM"
    if name.startswith("mattergen_tc_finetune_"): return "MatterGen TC+Stoich"
    if name.startswith("mattergen_stoich_"):       return "MatterGen Stoich"
    if name.startswith("mattergen_base_"):         return "MatterGen Base"
    if name.startswith("mattergen_"):              return "MatterGen Finetuned"
    return "Other"

# Try preferred RMSE key; fallback to a scalar RMSE if present
rmse_candidates = [
    "RMSE.AtomGen.mean_cartesian_rms_angstrom",
    "RMSE.AtomGen.mean_normalized_cartesian_rms",
    "RMSE.AtomGen",
    "RMSE",
]  # last handles legacy flat values
for cand in rmse_candidates:
    if cand in df.columns:
        rmse_col = cand
        break
else:
    print("ERROR: Could not find RMSE column in metrics.json files", file=sys.stderr)
    sys.exit(1)

# Build tidy frame with display names, models, colors
plot_df = (
    df[["benchmark_name", rmse_col]]
      .rename(columns={rmse_col: "RMSE"})
      .assign(model=lambda x: x["benchmark_name"].apply(infer_model),
              display=lambda x: x["benchmark_name"].map(bnchmk_name_dict).fillna(x["benchmark_name"]))
)

# ───────────────────── colors by model (3 colors) ─────────────────────
model_colors = {
    "AtomGPT":              "#1f77b4",  # tab:blue
    "CDVAE":                "#ff7f0e",  # tab:orange
    "FlowMM":               "#2ca02c",  # tab:green
    "MatterGen Finetuned":  "#d62728",  # tab:red
    "MatterGen Base":       "#9467bd",  # tab:purple
    "MatterGen Stoich":     "#8c564b",  # tab:brown
    "MatterGen TC+Stoich":  "#e377c2",  # tab:pink
    "Other":                "#7f7f7f",
}
plot_df["color"] = plot_df["model"].map(model_colors)

# Preserve the same ordering as other charts (alphabetical by subdir)
x_labels = plot_df["display"].tolist()
heights  = plot_df["RMSE"].astype(float).tolist()
colors   = plot_df["color"].tolist()

# ───────────────────── styling to match MAE (length) chart ─────────────────────
def style_axes(ax, ylabel, title):
    ax.set_xlabel('', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=22)
    plt.xticks(rotation=30, ha='right', fontsize=13)
    plt.yticks(fontsize=15)
    plt.tight_layout()

# ───────────────────── plot ─────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

pos = np.arange(len(x_labels))
bar_width = 0.55  # ↓ smaller = thinner bars (try 0.5, 0.45, etc.)

ax.bar(
    pos,
    heights,
    width=bar_width,
    edgecolor='k',
    linewidth=0.8,   # outline thickness (optional)
    color=colors
)

# keep your pretty labels
ax.set_xticks(pos)
ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=13)

# Legend: one color per model (unchanged)
handles = [
    mpatches.Patch(color=model_colors["AtomGPT"],             label="AtomGPT"),
    mpatches.Patch(color=model_colors["CDVAE"],               label="CDVAE"),
    mpatches.Patch(color=model_colors["FlowMM"],              label="FlowMM"),
    mpatches.Patch(color=model_colors["MatterGen Finetuned"], label="MatterGen Finetuned"),
    mpatches.Patch(color=model_colors["MatterGen Base"],      label="MatterGen Base"),
    mpatches.Patch(color=model_colors["MatterGen Stoich"],    label="MatterGen Stoich"),
    mpatches.Patch(color=model_colors["MatterGen TC+Stoich"], label="MatterGen TC+Stoich"),
]
ax.legend(handles=handles, title_fontsize=15, fontsize=15)

style_axes(
    ax,
    ylabel='Average RMSE (Å)',
    title='Average Root Mean Squared Error \nfor Predicted vs. Target Atomic Coordinates'
)

out_path = ROOT / 'rmse_bar_chart.png'
plt.savefig(out_path, dpi=300)
plt.close(fig)


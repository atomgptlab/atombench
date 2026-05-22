#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['font.family'] = 'serif'

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

ROOT = Path.cwd()
print(f"DEBUG: Running match-rate script in {ROOT}", file=sys.stderr)

# ───────────────────────── ingest metrics ─────────────────────────
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
    rows.append(pd.json_normalize(rec, sep=".", max_level=3).iloc[0].to_dict())
    print(f"DEBUG: Loaded metrics for {rec['benchmark_name']}", file=sys.stderr)

if not rows:
    print("ERROR: No metrics.json files found – exiting", file=sys.stderr)
    sys.exit(1)

df = pd.DataFrame(rows)

# ───────────────────── pretty names / labels ─────────────────────
bnchmk_name_dict = {
    "agpt_benchmark_alex":               "AtomGPT Alexandria",
    "agpt_benchmark_jarvis":             "AtomGPT JARVIS",
    "cdvae_benchmark_alex":              "CDVAE Alexandria",
    "cdvae_benchmark_jarvis":            "CDVAE JARVIS",
    "flowmm_benchmark_alex":             "FlowMM Alexandria",
    "flowmm_benchmark_jarvis":           "FlowMM JARVIS",
    "mattergen_benchmark_alex":          "MatterGen Finetuned Alexandria",
    "mattergen_benchmark_jarvis":        "MatterGen Finetuned JARVIS",
    "mattergen_base_benchmark_alex":     "MatterGen Base Alexandria",
    "mattergen_base_benchmark_jarvis":   "MatterGen Base JARVIS",
    "mattergen_stoich_benchmark_alex":   "MatterGen Stoich Alexandria",
    "mattergen_stoich_benchmark_jarvis": "MatterGen Stoich JARVIS",
    "mattergen_tc_finetune_benchmark_alex":  "MatterGen TC+Stoich Alexandria",
    "mattergen_tc_finetune_benchmark_jarvis":"MatterGen TC+Stoich JARVIS",
}

model_colors = {
    "AtomGPT":             "#1f77b4",  # tab:blue
    "CDVAE":               "#ff7f0e",  # tab:orange
    "FlowMM":              "#2ca02c",  # tab:green
    "MatterGen Finetuned": "#d62728",  # tab:red
    "MatterGen Base":      "#9467bd",  # tab:purple
    "MatterGen Stoich":    "#8c564b",  # tab:brown
    "MatterGen TC+Stoich": "#e377c2",  # tab:pink
    "Other":               "#7f7f7f",
}

def infer_model(name: str) -> str:
    name = name.lower()
    if name.startswith("agpt_"):                   return "AtomGPT"
    if name.startswith("cdvae_"):                  return "CDVAE"
    if name.startswith("flowmm_"):                 return "FlowMM"
    if name.startswith("mattergen_tc_finetune_"):  return "MatterGen TC+Stoich"
    if name.startswith("mattergen_stoich_"):       return "MatterGen Stoich"
    if name.startswith("mattergen_base_"):         return "MatterGen Base"
    if name.startswith("mattergen_"):              return "MatterGen Finetuned"
    return "Other"

def style_axes(ax, ylabel, title):
    ax.set_xlabel('', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=22)
    plt.xticks(rotation=30, ha='right', fontsize=13)
    plt.yticks(fontsize=15)
    plt.tight_layout()

# ───────────────────────── match-rate plot ────────────────────────
match_col = "RMSE.AtomGen.match_rate"
if match_col not in df.columns:
    print("ERROR: Missing match_rate column", file=sys.stderr)
    sys.exit(1)

match_df = (df.set_index('benchmark_name')[[match_col]]
              .rename(index=bnchmk_name_dict)
              .rename(columns={match_col: 'Match Rate'}))

x_labels  = match_df.index.tolist()
heights   = match_df["Match Rate"].astype(float).tolist()
bar_colors = [
    model_colors.get(infer_model(name), model_colors["Other"])
    for name in df.sort_values("benchmark_name")["benchmark_name"]
]

import numpy as np
fig, ax = plt.subplots(figsize=(10, 8))
pos = np.arange(len(x_labels))
ax.bar(pos, heights, width=0.55, edgecolor='k', linewidth=0.8, color=bar_colors)
ax.set_xticks(pos)
ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=13)

handles = [
    mpatches.Patch(color=model_colors[m], label=m)
    for m in list(model_colors.keys())[:-1]  # exclude "Other"
    if any(infer_model(n) == m for n in df["benchmark_name"])
]
ax.legend(handles=handles, title='Model', title_fontsize=15, fontsize=15)

style_axes(ax,
           ylabel='Match Rate',
           title='Structure Match Rate of\nPredicted vs. Target Crystals')

plt.savefig(ROOT / 'match_rate_bar_chart.png', dpi=300)
plt.close(fig)

print("DEBUG: All done.", file=sys.stderr)

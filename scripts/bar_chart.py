import json
import os
from pathlib import Path
import sys

# Force a non-interactive backend in case you’re headless
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['font.family'] = 'serif'

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd()
print(f"DEBUG: Running script in {ROOT}", file=sys.stderr)

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
    "agpt_benchmark_alex":              "AtomGPT Alexandria",
    "agpt_benchmark_jarvis":            "AtomGPT JARVIS",
    "cdvae_benchmark_alex":             "CDVAE Alexandria",
    "cdvae_benchmark_jarvis":           "CDVAE JARVIS",
    "flowmm_benchmark_alex":            "FlowMM Alexandria",
    "flowmm_benchmark_jarvis":          "FlowMM JARVIS",
    "mattergen_benchmark_alex":         "MatterGen Finetuned Alexandria",
    "mattergen_benchmark_jarvis":       "MatterGen Finetuned JARVIS",
    "agpt_stoich_benchmark_alex":            "AtomGPT Alexandria",
    "agpt_stoich_benchmark_jarvis":          "AtomGPT JARVIS",
    "mattergen_stoich_benchmark_alex":       "MatterGen Alexandria",
    "mattergen_stoich_benchmark_jarvis":     "MatterGen JARVIS",
    "mattergen_tc_finetune_benchmark_alex":  "MatterGen Tc Alexandria",
    "mattergen_tc_finetune_benchmark_jarvis":"MatterGen Tc JARVIS",
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

ax_label_map = {
    'a':     r'$a$',
    'b':     r'$b$',
    'c':     r'$c$',
    'alpha': r'$\alpha$',
    'beta':  r'$\beta$',
    'gamma': r'$\gamma$'
}

def style_axes(ax, ylabel, title):
    ax.set_xlabel('', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=22)
    ax.legend(title='Lattice Parameter', title_fontsize=15, fontsize=15)
    plt.xticks(rotation=30, ha='right', fontsize=13)
    plt.yticks(fontsize=15)
    plt.tight_layout()

# ───────────────────────────── KLD plot ────────────────────────────
kld_cols = [f'KLD.{k}' for k in ax_label_map]
if any(c not in df.columns for c in kld_cols):
    print("ERROR: Missing KLD columns", file=sys.stderr); sys.exit(1)

kld_df = (df.set_index('benchmark_name')[kld_cols]
            .reindex([k for k in ICE_ORDER if k in df['benchmark_name'].values])
            .rename(index=bnchmk_name_dict)
            .rename(columns=lambda c: ax_label_map[c.split('.')[-1]]))

fig, ax = plt.subplots(figsize=(10, 8))
kld_df.plot(kind='bar', edgecolor='k', ax=ax)
style_axes(ax, 'KL Divergence (Nats)',
           'KL Divergence of Predicted vs. Target\nLattice-Parameter Distributions')
plt.savefig(ROOT / 'comparison_bar_chart.png', dpi=300)
plt.close(fig)

# ───────────────────────────── MAE plots ───────────────────────────
mae_candidates = [
    [f'MAE.average_mae.{k}' for k in ax_label_map],
    [f'MAE.{k}' for k in ax_label_map]
]
for cand in mae_candidates:
    if all(col in df.columns for col in cand):
        mae_cols = cand; break
else:
    print("ERROR: Could not find MAE columns", file=sys.stderr); sys.exit(1)

mae_df = (df.set_index('benchmark_name')[mae_cols]
            .reindex([k for k in ICE_ORDER if k in df['benchmark_name'].values])
            .rename(index=bnchmk_name_dict)
            .rename(columns=lambda c: ax_label_map[c.split('.')[-1]]))

length_cols = [ax_label_map[k] for k in ('a', 'b', 'c')]
angle_cols  = [ax_label_map[k] for k in ('alpha', 'beta', 'gamma')]

# ----- a, b, c -----
fig_len, ax_len = plt.subplots(figsize=(10, 8))
mae_df[length_cols].plot(kind='bar', edgecolor='k', ax=ax_len)
style_axes(ax_len, 'Mean Absolute Error (Å)',
           'Mean Absolute Error – Lattice Lengths (Å)')
plt.savefig(ROOT / 'mae_bar_chart_abc.png', dpi=300)
plt.close(fig_len)

# ----- α, β, γ -----
fig_ang, ax_ang = plt.subplots(figsize=(10, 8))
mae_df[angle_cols].plot(kind='bar',
                        edgecolor='k',
                        color=['red', 'purple', 'brown'],   # ← the new colors
                        ax=ax_ang)
style_axes(ax_ang, 'Mean Absolute Error (°)',
           'Mean Absolute Error – Lattice Angles (°)')
plt.savefig(ROOT / 'mae_bar_chart_angles.png', dpi=300)
plt.close(fig_ang)

print("DEBUG: All done.", file=sys.stderr)

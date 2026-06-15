# AtomBench tutorial

Score a model's crystal reconstructions, then submit them to the JARVIS-Leaderboard.

## Install

```bash
pip install atombench            # metrics, figures, tables
pip install 'atombench[submit]'  # also enables leaderboard submission
```

Submitting also needs `git` on your PATH and a GitHub token (see
[step 3](#3-submit-to-the-jarvis-leaderboard)).

## 1. Prepare a benchmark CSV

A benchmark CSV has three columns:

| column | meaning |
|---|---|
| `id` | unique identifier for each crystal (e.g. `JVASP-1002`) |
| `target` | ground-truth structure, as a string |
| `prediction` | the model's reconstructed structure, as a string |

Each cell is one structure. **POSCAR** and **CIF** are both accepted (auto-detected);
the convention is a single-line, `\n`-escaped POSCAR. Build it from your structures
with pymatgen:

```python
import pandas as pd
from pymatgen.io.vasp import Poscar

def to_poscar_line(structure):
    """pymatgen Structure -> single-line, \\n-escaped POSCAR string."""
    return str(Poscar(structure)).replace("\n", r"\n")

rows = []
for crystal_id, target, prediction in my_results:   # your (id, Structure, Structure) triples
    rows.append({
        "id": crystal_id,
        "target": to_poscar_line(target),
        "prediction": to_poscar_line(prediction),
    })

pd.DataFrame(rows).to_csv("mymodel.csv", index=False)
```

When submitting against an existing benchmark, the `id`s must match that
benchmark's test set.

**Comparing several models?** Put one CSV per model in a directory and AtomBench
overlays them:

```
benchmarks/
├── atomgpt.csv
├── cdvae.csv
├── flowmm.csv
└── mattergen.csv
```

## 2. Compute reconstruction metrics

```bash
atombench mymodel.csv out/        # one model
atombench benchmarks/ out/        # several models, overlaid
```

Metrics: lattice-parameter KLD and MAE, atomic-coordinate RMSD and ccRMSD,
structure-match rate, and per-crystal-system MAE. Output:

```
out/
├── figures/                  # PNG plots (all models overlaid)
└── numerical_calculations/
    ├── metrics_table.json
    ├── metrics_table.tex     # paste into a LaTeX manuscript
    └── epic_metrics.csv
```

A `metrics.json` is cached next to each input CSV. Options: `--amd-k` (default
`100`), `--symprec` (`0.1`), `--kmin` (`10`); see `atombench -h`.

Or from Python:

```python
import pandas as pd
from atombench.cli import compute_metrics

df = pd.read_csv("mymodel.csv")
metrics = compute_metrics(df, "mymodel", amd_k=100, symprec=0.1, kmin=10)
print(metrics["RMSE"], metrics["ccRMSD"])
```

## 3. Submit to the JARVIS-Leaderboard

The same CSV can be published as an `AI / AtomGen` contribution. `atombench-submit`
validates it, builds the contribution (normalizing predictions to POSCAR), and
opens a pull request.

Create a [GitHub token](https://github.com/settings/tokens) (classic with the
`repo` scope, or fine-grained with Contents + Pull requests set to *Read and
write*) and export it:

```bash
export GITHUB_TOKEN=ghp_xxx
```

**Preview without pushing** — build and inspect the contribution first:

```bash
atombench-submit mymodel.csv \
  --dataset dft_3d --prop Tc_supercon \
  --model-name MyModel --author-email me@example.com \
  --project-url https://example.com/paper \
  --git-url https://github.com/me/mymodel \
  --no-push --out ./submission
```

Validation fails with a clear error if an `id` doesn't match the benchmark test
set — before anything is pushed.

**Submit against an existing benchmark** — drop `--no-push`:

```bash
atombench-submit mymodel.csv \
  --dataset dft_3d --prop Tc_supercon \
  --model-name MyModel --author-email me@example.com \
  --project-url https://example.com/paper \
  --git-url https://github.com/me/mymodel
```

`--dataset` and `--prop` name the benchmark `AI-AtomGen-<prop>-<dataset>-test-rmse`.
Existing ones: `dft_3d`/`Tc_supercon`, `carbon24`/`energy_per_atom`,
`perov5`/`heat_ref`. PRs go to `atomgptlab/jarvis_leaderboard` by default; use
`--repo usnistgov/jarvis_leaderboard --base develop` for the NIST leaderboard.

**Create a new benchmark** from your CSV's `target` column (for a dataset the
leaderboard doesn't have yet):

```bash
atombench-submit mymodel.csv --dataset alex --prop Tc \
  --new-benchmark --description "Reconstruction on the Alexandria DS-A/B superconductors." \
  --model-name MyModel --author-email me@example.com \
  --project-url https://example.com/paper \
  --git-url https://github.com/me/mymodel
```

**From Python:**

```python
from atombench import submit

submit(
    "mymodel.csv",
    dataset="dft_3d", prop="Tc_supercon",
    model_name="MyModel", author_email="me@example.com",
    project_url="https://example.com/paper",
    git_url="https://github.com/me/mymodel",
    push=False, out_dir="./submission",   # drop push=False to open a PR
)
```

See `atombench-submit -h` for all options.

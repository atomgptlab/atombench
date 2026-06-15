<div align="center">

<img src="logo.png" alt="AtomBench" width=550 />

**A Python package for benchmarking generative crystal reconstruction models**

[![arXiv](https://img.shields.io/badge/arXiv-2510.16165-FF5050.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.16165)
[![License: MIT](https://img.shields.io/badge/License-MIT-FF9999.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.9-FF5050.svg)](pyproject.toml)
[![Open in Colab](https://img.shields.io/badge/Colab-tutorial-FF9999.svg)](https://github.com/crhysc/jarvis-tools-notebooks/blob/master/atombench_example.ipynb)
[![Documentation](https://img.shields.io/badge/docs-latest-FF5050.svg)](https://atomgptlab.github.io/atombench/)

</div>

---

**AtomBench automates the statistical analysis of how well generative inverse materials design models reconstruct known crystals.** Point it at a model's predicted structures and it scores how faithfully they reconstruct the targets. Its main use is running several models in one go, overlaying them in shared figures and gathering their metrics into a single table.

We also used AtomBench to run our own study, benchmarking four models (AtomGPT, CDVAE, FlowMM, and MatterGen) on the JARVIS Supercon-3D and Alexandria DS-A/B superconductivity datasets. Those benchmarks are fully reproducible through the Snakemake pipeline in this repository.

> [!NOTE]
> **Full documentation:** <https://atomgptlab.github.io/atombench/>

The repository has two parts you can use independently:

| Part | What it does | Runs on |
| --- | --- | --- |
| **`atombench`** (Python package) | Turns a model's benchmark CSVs into reconstruction metrics, figures, and tables | Any OS with Python 3.9+ |
| **Snakemake pipeline** | Trains and evaluates the models that generate those CSVs — reproduces our study | Linux HPC · SLURM · CUDA 11.8 |

## Contents

- [Quick Start: the `atombench` package](#quick-start-the-atombench-package)
- [Submit to the JARVIS-Leaderboard](#submit-to-the-jarvis-leaderboard)
- [Reproducing Our Benchmarks](#reproducing-our-benchmarks)
- [Tutorials](#tutorials)
- [Citation](#citation)
- [License](#license)

## Quick Start: the `atombench` package

The `atombench` package reads benchmark CSVs from any generative model and produces reconstruction metrics, figures, and summary tables for fast, accurate comparison.

```bash
pip install atombench
```

Point it at your data and an output directory:

```bash
atombench PATH OUTDIR
```

`PATH` is usually a directory of benchmark CSVs — one per model — which AtomBench runs together and overlays in the figures and metrics table. A single CSV, for a single-model benchmark, is also valid:

```
benchmarks/
├── atomgpt.csv
├── cdvae.csv
├── flowmm.csv
└── mattergen.csv
```

```bash
atombench benchmarks/ out/
```

`out/` then holds two folders:

```
out/
├── figures/                  # plots (PNG), all models overlaid
└── numerical_calculations/   # metrics_table.{json,tex}, epic_metrics.csv
```

Every input CSV needs three columns:

| Column | Description |
| --- | --- |
| `id` | A unique identifier for the structure |
| `target` | The ground-truth structure, as a POSCAR-formatted string |
| `prediction` | The model's predicted structure, as a POSCAR-formatted string |

> [!TIP]
> A `metrics.json` is written next to each input CSV and reused as a cache on later runs.

## Submit to the JARVIS-Leaderboard

`atombench-submit` turns the same CSV you score into a valid `AI / AtomGen` contribution and opens a pull request. It validates the CSV, normalizes predictions to POSCAR, and wraps the leaderboard's workflow without modifying it.

```bash
pip install 'atombench[submit]'
atombench-submit predictions.csv --dataset dft_3d --prop Tc_supercon \
  --model-name MyModel --author-email me@example.com \
  --project-url https://example.com/paper --git-url https://github.com/me/mymodel
```

See the [documentation](https://atomgptlab.github.io/atombench/) for the GitHub token, previewing without pushing, creating new benchmarks, and the Python API.

## Reproducing Our Benchmarks

Our study's benchmarks are fully reproducible through the Snakemake pipeline in this repository. It trains and evaluates AtomGPT, CDVAE, FlowMM, and MatterGen on the JARVIS Supercon-3D and Alexandria DS-A/B superconductivity datasets, then runs `atombench` on the resulting CSVs.

> [!IMPORTANT]
> The pipeline targets Linux HPC clusters (SLURM + CUDA 11.8) and will not run on macOS or Windows. Run `bash depcheck.sh` to check your environment before you start.

The full guide — requirements, installation, running and debugging the Snakemake DAG, manual recovery, GPU selection, and troubleshooting — lives in the documentation:

**[Reproducing our benchmarks →](https://atomgptlab.github.io/atombench/reproducing.html)**

There's also a [guided walkthrough in Google Colab](https://github.com/crhysc/jarvis-tools-notebooks/blob/master/atombench_example.ipynb).

## Tutorials

Per-model setup and usage notebooks:

| Model | Notebook |
| --- | --- |
| AtomGPT | [Open in Colab](https://github.com/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_example.ipynb) |
| CDVAE | [Open in Colab](https://github.com/crhysc/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/cdvae_example.ipynb) |
| FlowMM | [Open in Colab](https://github.com/crhysc/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/flowmm_example.ipynb) |

## Citation

If you use AtomBench in your research, please cite:

```bibtex
@article{campbell2026atombench,
  title   = {AtomBench: A Benchmarking Framework for Generative Crystal Reconstruction Models in Conventional Superconductors},
  author  = {Campbell, Charles Rhys and Romero, Aldo H. and Choudhary, Kamal},
  year    = {2026},
}
```

## License

Released under the [MIT License](LICENSE).

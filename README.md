<div align="center">

<img src="logo.png" alt="AtomBench" width=550 />

**A Python package for benchmarking generative crystal reconstruction models**

[![arXiv](https://img.shields.io/badge/arXiv-2510.16165-FF5050.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.16165)
[![License: MIT](https://img.shields.io/badge/License-MIT-FF9999.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.9-FF5050.svg)](pyproject.toml)
[![Open in Colab](https://img.shields.io/badge/Colab-tutorial-FF9999.svg)](https://github.com/crhysc/jarvis-tools-notebooks/blob/master/atombench_example.ipynb)
[![Documentation](https://img.shields.io/badge/docs-latest-FF5050.svg)](https://atomgptlab.github.io/atombench/)

</div>

<h1></h1>

**AtomBench is a Python package that automates the statistical analysis of the reconstruction performance of generative inverse materials design models.** Point it at a model's predicted structures and it scores how faithfully they reconstruct the targets. Its main use is running several models in one go, where it overlays them in shared figures and gathers their metrics into a single table.

We also used AtomBench to run our own study, benchmarking four models (AtomGPT, CDVAE, FlowMM, and MatterGen) on the JARVIS Supercon-3D and Alexandria DS-A/B superconductivity datasets. Those benchmarks are fully reproducible through the Snakemake pipeline in this repository.

**Documentation:** <https://atomgptlab.github.io/atombench/>

## Contents

- [Quick Start: the `atombench` package](#quick-start-the-atombench-package)
- [Reproducing Our Benchmarks](#reproducing-our-benchmarks)
- [Tutorials](#tutorials)
- [Citation](#citation)
- [License](#license)

The repository has two parts you can use independently:

- **`atombench`** is the Python package. It turns a model's benchmark CSVs into reconstruction metrics, figures, and tables, and runs anywhere with Python 3.9+.
- **The Snakemake pipeline** is how we produced the benchmarks in our study. It trains and evaluates the models that generate those CSVs, and needs a Linux HPC cluster with SLURM and CUDA 11.8.

<h1></h1>

## Quick Start: the `atombench` package

The `atombench` package reads benchmark CSVs from any generative model and produces reconstruction metrics, figures, and summary tables for fast and accurate comparison.

Install it:

```bash
pip install atombench
```

Run it:

```bash
atombench PATH OUTDIR
```

`PATH` is usually a directory of benchmark CSVs, one per model, which AtomBench runs together and overlays in the figures and metrics table. One CSV for a single-model benchmark is also a valid input. For example:

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

A `metrics.json` is also written next to each input CSV and reused as a cache on later runs.

Every input CSV needs three columns:

- `id`: a unique identifier for the structure
- `target`: the ground-truth structure, as a POSCAR-formatted string
- `prediction`: the model's structure, as a POSCAR-formatted string

<h1></h1>

## Submit to the JARVIS-Leaderboard

`atombench-submit` turns the same CSV you score into a valid `AI / AtomGen`
contribution and opens a pull request. It validates the CSV, normalizes
predictions to POSCAR, and wraps the leaderboard's workflow without modifying it.

```bash
pip install 'atombench[submit]'
atombench-submit predictions.csv --dataset dft_3d --prop Tc_supercon \
  --model-name MyModel --author-email me@example.com \
  --project-url https://example.com/paper --git-url https://github.com/me/mymodel
```

See the [documentation](https://atomgptlab.github.io/atombench/) for the GitHub
token, previewing without pushing, creating new benchmarks, and the Python API.

<h1></h1>

## Reproducing Our Benchmarks

Our study's benchmarks are fully reproducible through the Snakemake pipeline in this repository. It trains and evaluates AtomGPT, CDVAE, FlowMM, and MatterGen on the JARVIS Supercon-3D and Alexandria DS-A/B superconductivity datasets, then runs `atombench` on the resulting CSVs.

The pipeline targets Linux HPC clusters (SLURM + CUDA 11.8) and will not run on macOS or Windows. Run `bash depcheck.sh` to check your environment before you start.

The full guide — requirements, installation, running and debugging the Snakemake DAG, manual recovery, GPU selection, and troubleshooting — lives in the documentation:

**[Reproducing our benchmarks →](https://atomgptlab.github.io/atombench/reproducing.html)**

There's also a [guided walkthrough in Google Colab](https://github.com/crhysc/jarvis-tools-notebooks/blob/master/atombench_example.ipynb).

<h1></h1>

## Tutorials

Per-model setup and usage notebooks:

- [AtomGPT](https://github.com/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_example.ipynb)
- [CDVAE](https://github.com/crhysc/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/cdvae_example.ipynb)
- [FlowMM](https://github.com/crhysc/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/flowmm_example.ipynb)

<h1></h1>

## Citation

If you use AtomBench in your research, please cite:

```bibtex
@article{campbell2026atombench,
  title   = {AtomBench: A Benchmarking Framework for Generative Crystal Reconstruction Models in Conventional Superconductors},
  author  = {Campbell, Charles Rhys and Romero, Aldo H. and Choudhary, Kamal},
  year    = {2026},
}
```

<h1></h1>

## License

Released under the [MIT License](LICENSE).

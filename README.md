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

The repository has two independent parts: the `atombench` package, which runs anywhere with Python 3.9+, and the Snakemake pipeline that reproduces our study, which needs a Linux HPC cluster.

<h1></h1>

## Quick Start: the `atombench` package

Install the package:

```bash
pip install atombench
```

Run it on a directory of benchmark CSVs, one per model, which it overlays in shared figures and a metrics table. A single CSV also works.

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

Each input CSV needs three columns: `id` (a unique identifier), `target` (the ground-truth structure as a POSCAR string), and `prediction` (the model's structure as a POSCAR string). A `metrics.json` is cached next to each CSV for later runs.

<h1></h1>

## Submit to the JARVIS-Leaderboard

`atombench-submit` turns the same CSV into a valid `AI / AtomGen` contribution and opens a pull request.

```bash
pip install 'atombench[submit]'
atombench-submit predictions.csv --dataset dft_3d --prop Tc_supercon \
  --model-name MyModel --author-email me@example.com \
  --project-url https://example.com/paper --git-url https://github.com/me/mymodel
```

See the [documentation](https://atomgptlab.github.io/atombench/) for the GitHub token, new benchmarks, and the Python API.

<h1></h1>

## Reproducing Our Benchmarks

The Snakemake pipeline reproduces our study, training and evaluating all four models on the JARVIS Supercon-3D and Alexandria DS-A/B datasets. It needs a Linux HPC cluster (SLURM + CUDA 11.8); run `bash depcheck.sh` first.

The full guide lives in the [documentation](https://atomgptlab.github.io/atombench/reproducing.html), with a [walkthrough in Google Colab](https://github.com/crhysc/jarvis-tools-notebooks/blob/master/atombench_example.ipynb).

<h1></h1>

## Tutorials

Per-model notebooks: [AtomGPT](https://github.com/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_example.ipynb), [CDVAE](https://github.com/crhysc/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/cdvae_example.ipynb), and [FlowMM](https://github.com/crhysc/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/flowmm_example.ipynb).

<h1></h1>

## Citation

If you use AtomBench, please cite:

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

<h1>
<p align="center">
    <img src="logo.png" alt="AtomBench logo" width="450"/>
</p>
</h1>
<h4 align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2312.03687-blue.svg?logo=arxiv&logoColor=white.svg)](https://arxiv.org/abs/2510.16165)
[![Requires Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)]()
</h4>

The rapid development of generative AI models for materials discovery has created a need for standardized benchmarks to evaluate their performance. In this work, we present $\textbf{AtomBench}$, a systematic benchmarking framework that comparatively evaluates three representative generative architectures-AtomGPT (transformer-based), CDVAE (diffusion variational autoencoder), and FlowMM (Riemannian flow matching)-for inverse crystal structure design. We train and evaluate these models on two high-quality DFT superconductivity datasets: JARVIS Supercon-3D and Alexandria DS-A/B, comprising over 9,000 structures with computed electron-phonon coupling properties.

</h1>

## Installation & Usage

The `atombench` package takes benchmark CSV files produced by any generative crystal structure model and computes reconstruction metrics, generates publication-ready figures, and writes summary tables — all in a single command. It is independent of the Snakemake pipeline and can be used on any benchmark CSVs you already have.

Each CSV must have columns `id`, `target`, and `prediction`, where `target` and `prediction` are POSCAR-formatted crystal structures stored as strings.

```bash
pip install atombench
```

```bash
atombench  PATH  OUTDIR
```

`PATH` can be a single benchmark CSV or a directory of CSVs. `OUTDIR` is the output directory; outputs are written to two subdirectories:

```
OUTDIR/
  figures/                     ← all plots (PNG)
  numerical_calculations/      ← metrics_table.{json,tex}, epic_metrics.csv
```

Per-benchmark `metrics.json` files are also written beside each input CSV for caching.

</h1>

## Cite Us

If you use AtomBench in your research, please cite:

```bibtex
@article{campbell2026atombench,
  title   = {AtomBench: A Benchmarking Framework for Generative Crystal Reconstruction Models in Conventional Superconductors},
  author  = {Campbell, Charles Rhys and Romero, Aldo H. and Choudhary, Kamal},
  year    = {2026},
}
```

</h1>

## Recompute AtomBench Benchmarks

The sections below describe how to re-run the full Snakemake pipeline on an HPC cluster to reproduce the benchmark results from scratch.

### [Tutorial in Google Colab](https://github.com/crhysc/jarvis-tools-notebooks/blob/master/atombench_example.ipynb)

### Installation Instructions

#### Step 1: Confirm operating system, workload manager, and Python package manager
This repo is built for Linux-based High-Performance Computing (HPC) clusters that use the SLURM workload manager and provide a CUDA 11.8 module. It will not run on MacOS devices, Windows devices, and non-SLURM HPC clusters. A valid `conda` installation is also required, and it must have the capacity to be initialized using the `conda` shell integration hook via the following command:
```bash
eval "$(conda shell.bash hook)"
```
Documentation about activating `conda` environments using this command can be found [here](https://docs.conda.io/projects/conda/en/latest/dev-guide/deep-dives/activation.html). To verify that using the shell hook works, running `conda deactivate` should remove the `(base)` prefix, and running `eval "$(conda shell.bash hook)"` should re-add the `(base)` prefix:
```bash
(base) [user@hpc-cluster ~]$ conda deactivate
[user@hpc-cluster ~]$ eval "$(conda shell.bash hook)"
(base) [user@hpc-cluster ~]$
```
Moreover, we use `mamba` to perform the FlowMM environment setup. To determine if your system has the correct dependencies, run `depcheck.sh`.

#### Step 2: Download this repository
Download this repository using the following command:
```bash
git clone git@github.com:atomgptlab/atombench.git
```
NOTE: This repository must be cloned using `ssh` rather than `https` due to constraints from the generative models used in this repository.

#### Step 3: Initialize the generative models
To recompute the benchmarks, the generative models need to be downloaded into the `models/` directory, and we automate this using `git submodules`. In the root directory of this repository, run the following command to download and initialize the generative models used in this study:
```bash
git submodule update --init --recursive
```

#### Step 4: Create and activate a `conda` environment
```bash
conda create --name atombench python=3.11 pip -y
conda activate atombench
```

#### Step 5: Add `mamba` to the Atombench `conda` environment
FlowMM's environment setup is very memory-hungry and can potentially cause OOM errors. To avoid this, install `mamba` into the atombench environment:
```bash
conda install -n atombench -c conda-forge mamba -y
```
NOTE: It may take 20 minutes to an hour to set up the FlowMM `conda` environment.

#### Step 6: Download Python dependencies
```bash
pip install atombench snakemake dvc ase jarvis-tools==2026.1.10
```

### Running the Pipeline

#### Step 1: Provide the location of this atombench repository
Locate the file `scripts/absolute_path.sh` and set the `ABS_PATH` environment variable equal to the repository's absolute path:
```bash
(atombench) [user@hpc-cluster atombench]$ pwd
/path/to/this/repository
```
then,
```bash
vi scripts/absolute_path.sh
```
finally,
```bash
#!/bin/bash
export ABS_PATH="/path/to/this/repository"
```

#### Step 2: Activate the `Snakemake` pipeline to recompute the benchmarks
After the previous setup steps have been completed, run the pipeline using the following command:
```bash
snakemake -p --verbose all --cores 1
```

If a job fails and you want the underlying error output (rather than only "job failed"), rerun with:
```bash
snakemake -p --verbose --show-failed-logs all --cores 1
```

If the failure occurs during environment creation, you can also run the environment job directly to see the full error output:
```bash
bash job_runs/agpt_benchmark_alex/conda_env.job
bash job_runs/cdvae_benchmark_alex/conda_env.job
bash job_runs/flowmm_benchmark_alex/conda_env.job
```

## Generative Model Installation & Usage Tutorials
### [AtomGPT](https://github.com/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_example.ipynb)
### [CDVAE](https://github.com/crhysc/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/cdvae_example.ipynb)
### [FlowMM](https://github.com/crhysc/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/flowmm_example.ipynb)

## Common Installation Pitfalls

- **`conda` doesn't activate properly**  
  If `conda activate` does nothing or errors, your shell is not initialized. Run  
  `eval "$(conda shell.bash hook)"` and confirm that `(base)` appears in your prompt.

- **Mixing multiple Python managers**  
  Having `conda`, `micromamba`, or system Python all active at once will cause hard-to-diagnose failures. Use *one* `conda` installation and ensure `which python` points inside the active environment.

- **Running on the wrong system**  
  This code assumes Linux + SLURM + CUDA 11.8. MacOS, Windows, or non-SLURM clusters will fail early, often with cryptic CUDA or scheduler errors.

- **Forgetting to initialize submodules**  
  If `models/` is empty or incomplete, you likely skipped  
  `git submodule update --init --recursive`.

- **Snakemake hides the real error**  
  When something fails, rerun with `--show-failed-logs` or execute the failing job script directly. The actual issue is almost always in the job log.

If you hit an issue not listed here, run `depcheck.sh` first — it catches most environment-level problems before you waste time debugging downstream failures.

## Manual Recovery if the Snakemake Pipeline Fails

In some HPC environments, long-running jobs or interactive installations may cause the `Snakemake` pipeline to terminate prematurely, even though the underlying computation would eventually complete. In these cases, the benchmarks can still be recomputed manually while preserving compatibility with the automated pipeline.

To see the next jobs that need to be executed to satisfy the `all` rule, run a dry run:
```bash
snakemake -n all
```

If you want a more concise view that focuses on the next jobs without extra log output, use:
```bash
snakemake -n --quiet all
```

The rule dependency DAG is defined directly in the root `Snakefile` (via the `rule all` target list and the downstream rules/modules it pulls in), and it can be inspected by reading the `Snakefile` directly.

Each benchmark job can be executed manually by navigating to the corresponding directory under `job_runs/` and running the relevant scripts directly (e.g., via `bash` or `python`, depending on the job). Once a job has completed successfully, you must explicitly mark it as finished by creating its expected output file in the root directory of the repository using `touch`. For example:
```bash
touch agpt_benchmark_jarvis.final
```

This signals to `Snakemake` that the job's outputs exist and that the rule should be considered complete.

Once all benchmark CSVs exist (all `*.final` touch files are present), the analysis step can be run directly:
```bash
atombench job_runs/ atombench_output/
```

To see the rest of the remaining pipeline after manually completing one or more jobs, rerun the dry run:
```bash
snakemake -n all
```

`Snakemake` will exclude the completed jobs and report only the remaining missing targets. This process can be repeated iteratively until all benchmarks have been completed and the `all` rule is satisfied.

This manual fallback preserves the dependency structure and bookkeeping guarantees of `Snakemake` while allowing recovery from transient scheduler, environment, or responsiveness issues common on shared HPC systems.

## Notes on GPU Selection and Model-Specific Snakefiles

In addition to the root Snakefile, each subdirectory under `job_runs/` contains its own model-specific Snakefile responsible for executing individual benchmark computations. These Snakefiles define `sbatch` directives directly, including explicit GPU requests (e.g., A100, H100, L40S).

On some HPC systems, a pipeline failure may occur if the requested GPU type is unavailable or restricted on the target cluster. In such cases, the fix is straightforward: edit the corresponding Snakefile (or associated job script) under `job_runs/` and replace the GPU specification with a valid GPU available on your system.

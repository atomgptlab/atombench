# AtomBench

The rapid development of generative AI models for materials discovery has created a need for standardized benchmarks to evaluate their performance. In this work, we present $\textbf{AtomBench}$, a systematic benchmarking framework that comparatively evaluates three representative generative architectures-AtomGPT (transformer-based), CDVAE (diffusion variational autoencoder), and FlowMM (Riemannian flow matching)-for inverse crystal structure design. We train and evaluate these models on two high-quality DFT superconductivity datasets: JARVIS Supercon-3D and Alexandria DS-A/B, comprising over 9,000 structures with computed electron-phonon coupling properties.

---

## `atombench` Python Package

The `atombench` package provides command-line tools for computing metrics and generating publication-quality figures from benchmark CSV files (columns: `id`, `target`, `prediction`). It is independent of the Snakemake pipeline and can be used on any benchmark CSVs you already have.

### Installation

```bash
pip install atombench
```

### CLI Tools

| Command | Description |
|---|---|
| `atombench <path>` | Compute all metrics and generate plots for one CSV or a directory of CSVs |
| `atombench-plots --root <job_runs/> --outdir <figures/>` | Generate all comparison charts from pre-computed metrics |
| `atombench-tables --root <job_runs/> --outdir <out/>` | Build summary metric tables (JSON + LaTeX) |
| `atombench-times <path>` | Harvest GPU wall-clock times and epoch counts from SLURM logs |
| `atombench-verify <path>` | Cross-benchmark consistency checks (shared test-set IDs) |

### Quick Example

```bash
# Compute metrics and plots for a single benchmark CSV
atombench path/to/benchmark.csv

# Process a directory of benchmark CSVs, writing outputs to my_figures/
atombench path/to/job_runs/ --outdir my_figures/

# Skip re-computing metrics if metrics.json already exists
atombench path/to/job_runs/ --skip-metrics

# Metrics only, no plots
atombench path/to/benchmark.csv --metrics-only
```

See `atombench --help` (and `atombench-plots --help`, etc.) for the full option reference.

---

## Cite Us!

If you use AtomBench in your research, please cite:

```bibtex
@article{campbell2025atombench,
  title   = {AtomBench: A Benchmarking Framework for Generative Crystal Reconstruction Models in Conventional Superconductors},
  author  = {Campbell, Charles Rhys and Romero, Aldo H. and Choudhary, Kamal},
  journal = {Preprint},
  year    = {2025},
  url     = {https://github.com/atomgptlab/atombench_inverse},
}
```

---

## Recompute AtomBench Benchmarks
### [Tutorial in Google Colab](https://github.com/crhysc/jarvis-tools-notebooks/blob/master/atombench_example.ipynb)

The sections below describe how to re-run the full Snakemake pipeline on an HPC cluster to reproduce the benchmark results from scratch.

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
git clone git@github.com:atomgptlab/atombench_inverse.git
```
NOTE: This repository must be cloned using `ssh` rather than `https` due to constraints from the generative models used in this repository.

#### Step 3: Initialize the generative models
To recompute the benchmarks, the generative models need to be downloaded into the `models/` directory, and we automate this using `git submodules`. In the root directory of this repository, run the following command to download and initialize the generative models used in this study:
```bash
git submodule update --init --recursive
```

#### Step 4: Create and activate a `conda` environment
Normally, it is best-practice to avoid installing Python packages to one's base `conda` environment. Make an environment to store required Python deps (note that `pip` is intentionally included here since it is used in downstream steps):
```bash
conda create --name atombench python=3.11 pip -y
conda activate atombench
```

#### Step 5: Add `mamba` to the Atombench `conda` environment
FlowMM's environment setup is very memory-hungry and can potentially cause OOM errors. To avoid this, download `mamba` to the base `conda` environment:
```bash
conda install -n atombench -c conda-forge mamba -y
```
NOTE: It may take 20 minutes to and hour to setup the FlowMM `conda` environment.

#### Step 6: Download Python dependencies
This repository recomputes the AtomBench benchmarks using a semi-automated `Snakemake` pipeline. For more information about `Snakemake`, visit their [documentation](https://snakemake.readthedocs.io/en/stable/) site. Moreover, we use `uv` to speed up downstream package installation, and we use `DVC` to automate dataset preprocessing.
```bash
pip install uv; uv pip install snakemake dvc average-minimum-distance jarvis-tools==2026.1.10 pymatgen ase tqdm numpy pandas atombench
```

### Common Installation Pitfalls

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

If you hit an issue not listed here, run `depcheck.sh` first—it catches most environment-level problems before you waste time debugging downstream failures.

### Running the Pipeline

#### Step 1: Provide the location of this atombench repository
For this repository to execute its core functionalities, it must know its own location in the computer's filesystem. To accomplish this, locate a file in the `scripts/` directory called `absolute_path.sh`, and set the `ABS_PATH` environment variable equal to the repository's absolute path, e.g.
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
As mentioned previously, we use an automated pipeline to compute these benchmarks. After the previous setup steps have been completed, run the pipeline using the following command:
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

To see the rest of the remaining pipeline after manually completing one or more jobs, rerun the dry run:
```bash
snakemake -n all
```

`Snakemake` will exclude the completed jobs and report only the remaining missing targets. This process can be repeated iteratively until all benchmarks have been completed and the `all` rule is satisfied.

This manual fallback preserves the dependency structure and bookkeeping guarantees of `Snakemake` while allowing recovery from transient scheduler, environment, or responsiveness issues common on shared HPC systems.

## Notes on GPU Selection and Model-Specific Snakefiles

In addition to the root Snakefile, each subdirectory under `job_runs/` contains its own model-specific Snakefile responsible for executing individual benchmark computations. These Snakefiles define `sbatch` directives directly, including explicit GPU requests (e.g., A100, H100, L40S).

On some HPC systems, a pipeline failure may occur if the requested GPU type is unavailable or restricted on the target cluster. In such cases, the fix is straightforward: edit the corresponding Snakefile (or associated job script) under job_runs/ and replace the GPU specification with a valid GPU available on your system.

# AtomBench

```{rst-class} lead
Benchmark generative crystal **reconstruction** models: score how faithfully a
model rebuilds known crystals, render figures and tables, and publish results to
the JARVIS-Leaderboard.
```

```{button-ref} tutorial
:ref-type: doc
:color: primary
:class: sd-px-4 sd-fs-6
Get started  →
```

```bash
pip install atombench
atombench mymodel.csv out/        # metrics, figures, tables
```

::::{grid} 1 2 2 2
:gutter: 3
:margin: 4 0 0 0

:::{grid-item-card} {octicon}`download;1em;sd-mr-1` Installation
:link: installation
:link-type: doc

Install AtomBench and the optional `submit` extras.
:::

:::{grid-item-card} {octicon}`book;1em;sd-mr-1` Tutorial
:link: tutorial
:link-type: doc

Build a benchmark CSV, compute metrics, and submit — end to end.
:::

:::{grid-item-card} {octicon}`code-square;1em;sd-mr-1` API reference
:link: api
:link-type: doc

The `submit()` and `compute_metrics()` Python functions.
:::

:::{grid-item-card} {octicon}`workflow;1em;sd-mr-1` Reproducing our benchmarks
:link: reproducing
:link-type: doc

Re-run the Snakemake pipeline on an HPC cluster.
:::

:::{grid-item-card} {octicon}`trophy;1em;sd-mr-1` Leaderboard
:link: https://atomgptlab.github.io/jarvis_leaderboard/Special/AtomGenBench/

Browse the live AtomGenBench leaderboard.
:::
::::

```{toctree}
:maxdepth: 2
:hidden:

Home <self>
installation
tutorial
api
reproducing
```

EXPS = [
    "agpt_benchmark_alex",
    "agpt_benchmark_jarvis",
    "cdvae_benchmark_alex",
    "cdvae_benchmark_jarvis",
    "flowmm_benchmark_alex",
    "flowmm_benchmark_jarvis",
    "agpt_stoich_benchmark_alex",
    "agpt_stoich_benchmark_jarvis",
    "mattergen_stoich_benchmark_alex",
    "mattergen_stoich_benchmark_jarvis",
    "mattergen_tc_finetune_benchmark_alex",
    "mattergen_tc_finetune_benchmark_jarvis",
]

for exp in EXPS:
    module:
        name: exp
        snakefile: f"job_runs/{exp}/Snakefile"
    use rule * from exp

rule all:
    input:
        expand("{exp}.final", exp=EXPS),
        "analysis.done",
        "overlay_charts.created",
        "benchmarks.verified",
        "all_figures.collected",
        "job_runs/computational_costs.json",
        "job_runs/computational_costs.tex"

rule make_atomgpt_env:
    output:
        touch("atomgpt_env.created")
    shell:
        """
        bash job_runs/agpt_benchmark_alex/conda_env.job
        """

rule make_cdvae_env:
    output:
        touch("cdvae_env.created")
    shell:
        """
        bash job_runs/cdvae_benchmark_alex/conda_env.job
        """

rule make_flowmm_env:
    output:
        touch("flowmm_env.created")
    shell:
        """
        bash job_runs/flowmm_benchmark_alex/conda_env.job
        """

rule make_mattergen_env:
    output:
        touch("mattergen_env.created")
    shell:
        """
        bash job_runs/mattergen_benchmark_alex/conda_env.job
        """

rule envs_ready:
    input:
        "atomgpt_env.created",
        "cdvae_env.created",
        "flowmm_env.created",
        "mattergen_env.created"
    output:
        touch("all_envs_ready.txt")
    shell:
        """
        echo 'all conda envs ready' > {output}
        """

rule make_jarvis_data:
    input:
        "all_envs_ready.txt"
    output:
        touch("jarvis_data.created")
    shell:
        """
        dvc --cd tc_supercon repro
        """

rule make_alex_data:
    input:
        "all_envs_ready.txt"
    output:
        touch("alex_data.created")
    shell:
        """
        dvc --cd alexandria repro
        """

rule prepare_mattergen_tc_data:
    input:
        "alex_data.created",
        "jarvis_data.created",
        "mattergen_env.created"
    output:
        touch("mattergen_tc_data.created")
    shell:
        """
        eval "$(conda shell.bash hook)"
        conda activate mattergen
        python scripts/patch_mattergen_tc_caches.py
        """

rule make_stats_yamls:
    input:
        "flowmm_env.created",
        "jarvis_data.created",
        "alex_data.created"
    output:
        touch("flowmm_yamls.created")
    shell:
        """
        bash job_runs/flowmm_benchmark_alex/yamls.sh
        """

rule analyze_benchmarks:
    input:
        expand("{exp}.final", exp=EXPS)
    output:
        touch("analysis.done")
    shell:
        "atombench job_runs/ atombench_output/"

rule verify_benchmarks:
    input:
        "analysis.done"
    output:
        touch("benchmarks.verified")
    shell:
        "atombench-verify job_runs/"

rule make_overlay_charts:
    input:
        "alex_data.created",
        "jarvis_data.created"
    output:
        touch("overlay_charts.created")
    shell:
        """
        bash scripts/make_overlay_charts.sh
        """

rule harvest_compute_times:
    input:
        expand("{exp}.final", exp=EXPS)
    output:
        "job_runs/computational_costs.json",
        "job_runs/computational_costs.tex"
    shell:
        "python scripts/harvest_compute_times.py"

rule collect_all_figures:
    input:
        "analysis.done",
        "overlay_charts.created"
    output:
        touch("all_figures.collected")
    shell:
        """
        mkdir -p all_figures/comparison \
                 all_figures/distributions \
                 all_figures/crystal_system \
                 all_figures/reconstruction \
                 all_figures/overlays \
                 all_figures/dataset
        cp atombench_output/figures/comparison_bar_chart.png    all_figures/comparison/
        cp atombench_output/figures/mae_bar_chart_abc.png       all_figures/comparison/
        cp atombench_output/figures/mae_bar_chart_angles.png    all_figures/comparison/
        cp atombench_output/figures/rmse_bar_chart.png          all_figures/comparison/
        cp atombench_output/figures/ccrmse_bar_chart.png        all_figures/comparison/
        cp atombench_output/figures/match_rate_bar_chart.png    all_figures/comparison/
        find atombench_output/figures -maxdepth 1 -name '*_distribution.png' \
            -exec cp {{}} all_figures/distributions/ \\;
        cp atombench_output/figures/crystal_system_mae_bar_chart_abc.png    all_figures/crystal_system/
        cp atombench_output/figures/crystal_system_mae_bar_chart_angles.png all_figures/crystal_system/
        cp atombench_output/figures/alexandria_reconstruction_grid.png all_figures/reconstruction/ 2>/dev/null || true
        cp atombench_output/figures/jarvis_reconstruction_grid.png     all_figures/reconstruction/ 2>/dev/null || true
        cp overlay_outputs/*.png all_figures/overlays/
        cp alexandria/alex_composition_pie_chart.png  all_figures/dataset/
        cp alexandria/alex_tc_histogram.png           all_figures/dataset/
        cp tc_supercon/jarvis_composition_pie_chart.png all_figures/dataset/
        cp tc_supercon/jarvis_tc_histogram.png          all_figures/dataset/
        """

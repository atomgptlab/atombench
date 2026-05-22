EXPS = [
    "agpt_benchmark_alex",
    "agpt_benchmark_jarvis",
    "cdvae_benchmark_alex",
    "cdvae_benchmark_jarvis",
    "flowmm_benchmark_alex",
    "flowmm_benchmark_jarvis",
    "mattergen_base_benchmark_alex",
    "mattergen_base_benchmark_jarvis",
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
        "charts.made",
        "overlay_charts.created",
        "benchmarks.verified",
        "grid_charts.created",
        "rmse_chart.made",
        "crystal_system_mae_charts.created",
        "match_rate_chart.made"

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

rule compile_results:
    input:
        expand("{exp}.final", exp=EXPS),
    output:
        touch("metrics.computed")
    shell:
        """
        cd job_runs/ && bash ../scripts/loop.sh
        """

rule verify_benchmarks:
    input:
        "metrics.computed"
    output:
        touch("benchmarks.verified")
    shell:
        """
        python scripts/verify_benchmarks.py --root job_runs/
        """

rule make_bar_charts:
    input:
        "metrics.computed"
    output:
        touch("charts.made")
    shell:
        "cd job_runs/ && python ../scripts/bar_chart.py"

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

rule make_grid_charts:
    input:
        "metrics.computed"
    output:
        touch("grid_charts.created")
    shell:
        """
        python scripts/grid_charts.py
        """

rule make_crystal_system_mae_charts:
    input:
        "metrics.computed"
    output:
        touch("crystal_system_mae_charts.created")
    shell:
        "python scripts/crystal_system_mae.py --root job_runs --symprec 0.1 --kmin 10 --outdir ./figures"

rule make_rmse_chart:
    input:
        "metrics.computed"
    output:
        touch("rmse_chart.made")
    shell:
        "cd job_runs/ && python ../scripts/rmse_bar_chart.py"

rule make_match_rate_chart:
    input:
        "metrics.computed"
    output:
        touch("match_rate_chart.made")
    shell:
        "cd job_runs/ && python ../scripts/match_rate_bar_chart.py"


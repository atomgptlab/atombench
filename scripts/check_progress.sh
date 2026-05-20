#!/usr/bin/env bash
# check_progress.sh — real-time meta-status for the atombench pipeline.
#
# STATIC MODE  (no atombench jobs in SLURM queue): print once and exit.
# LIVE MODE    (jobs detected): clear-screen refresh loop every REFRESH_SECS.
#
# Run from any directory; no conda activation required.
# Flags: --no-color, --once (force static), --interval N (live refresh secs)

set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
JOB_RUNS="$REPO/job_runs"
REFRESH_SECS=15
FORCE_ONCE=false

# ── Argument parsing ──────────────────────────────────────────────────────────
USE_COLOR=true
[[ -n "${NO_COLOR:-}" ]] && USE_COLOR=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-color)  USE_COLOR=false ;;
        --once)      FORCE_ONCE=true ;;
        --interval)  shift; REFRESH_SECS="$1" ;;
        --help|-h)
            cat <<'EOF'
Usage: check_progress.sh [--no-color] [--once] [--interval N] [--help]

Prints a pipeline status dashboard for all 10 atombench experiments.
Auto-detects whether atombench SLURM jobs are active:
  - No jobs → prints once and exits (static mode)
  - Jobs found → refreshes every REFRESH_SECS seconds (live mode, Ctrl+C exits)

Options:
  --no-color      Disable ANSI color output
  --once          Force single static print even if jobs are running
  --interval N    Live refresh interval in seconds (default: 15)

Sections:
  1. Prerequisites      conda envs + dataset preparation touch files
  2. Data directories   model-format splits on disk
  3. Pipeline table     all 12 experiments: touch files, CSV rows, key artifacts
  4. Active SLURM jobs  (live mode only) queue view with elapsed/limit per job
  5. SLURM log tails    recent output for in-flight experiments
  6. Metrics summary    match_rate, ccRMSE, MAE_avg, KLD_avg from metrics.json
  7. Post-processing    metrics.computed, charts, verify, epic_metrics.csv
EOF
            exit 0 ;;
    esac
    shift
done

# ── Colour helpers ────────────────────────────────────────────────────────────
if $USE_COLOR; then
    RED=$'\033[0;31m'; GRN=$'\033[0;32m'; YLW=$'\033[0;33m'
    CYN=$'\033[0;36m'; BLD=$'\033[1m';    DIM=$'\033[2m';    RST=$'\033[0m'
    REDB=$'\033[1;31m'
else
    RED=''; GRN=''; YLW=''; CYN=''; BLD=''; DIM=''; RST=''; REDB=''
fi

# ── SLURM job-name → experiment mapping ──────────────────────────────────────
# (sourced from #SBATCH --job-name= lines in every *.job file)
declare -A JOB_TO_EXP=(
    [runner]="(snakemake orchestrator)"
    [aagpttr]="agpt_benchmark_alex"
    [agpt_tc]="agpt_benchmark_jarvis"
    [cdvae_alex]="cdvae_benchmark_alex"
    [cdvae_jarvis]="cdvae_benchmark_jarvis"
    [cdvae_tc]="cdvae_benchmark_jarvis"
    [flow_alex]="flowmm_benchmark_alex"
    [flow_tc]="flowmm_benchmark_jarvis"
    [mgen_alex]="mattergen_benchmark_alex"
    [mgen_tc]="mattergen_benchmark_jarvis"
    [mgen_tc_alex]="mattergen_tc_finetune_benchmark_alex"
    [mgen_tc_jrv]="mattergen_tc_finetune_benchmark_jarvis"
)

# Experiment → primary SLURM job name(s) (space-separated, for squeue lookup)
declare -A EXP_JOBS=(
    [agpt_benchmark_alex]="aagpttr"
    [agpt_benchmark_jarvis]="agpt_tc"
    [cdvae_benchmark_alex]="cdvae_alex"
    [cdvae_benchmark_jarvis]="cdvae_jarvis cdvae_tc"
    [flowmm_benchmark_alex]="flow_alex"
    [flowmm_benchmark_jarvis]="flow_tc"
    [mattergen_benchmark_alex]="mgen_alex"
    [mattergen_benchmark_jarvis]="mgen_tc"
    [mattergen_tc_finetune_benchmark_alex]="mgen_tc_alex"
    [mattergen_tc_finetune_benchmark_jarvis]="mgen_tc_jrv"
)

# Experiment → slurm output log files (space-separated paths)
declare -A SLURM_LOGS=(
    [agpt_benchmark_alex]="$REPO/slurm_alex_atomgpt_train.out"
    [agpt_benchmark_jarvis]="$REPO/slurm_jarvis_atomgpt_train.out"
    [cdvae_benchmark_alex]="$REPO/slurm_alex_cdvae_train.out $REPO/slurm_alex_cdvae_infer.out $REPO/slurm_alex_cdvae_benchmark.out"
    [cdvae_benchmark_jarvis]="$REPO/slurm_jarvis_cdvae_train.out $REPO/slurm_jarvis_cdvae_infer.out $REPO/slurm_jarvis_cdvae_benchmark.out"
    [flowmm_benchmark_alex]="$REPO/slurm_alex_flowmm_train.out $REPO/slurm_alex_flowmm_infer.out"
    [flowmm_benchmark_jarvis]="$REPO/slurm_jarvis_flowmm_train.out $REPO/slurm_jarvis_flowmm_infer.out"
    [mattergen_benchmark_alex]="$REPO/slurm_alex_mattergen_train.out $REPO/slurm_alex_mattergen_infer.out"
    [mattergen_benchmark_jarvis]="$REPO/slurm_jarvis_mattergen_train.out $REPO/slurm_jarvis_mattergen_infer.out"
    [mattergen_tc_finetune_benchmark_alex]="$REPO/slurm_alex_mattergen_tc_finetune_train.out $REPO/slurm_alex_mattergen_tc_finetune_infer.out"
    [mattergen_tc_finetune_benchmark_jarvis]="$REPO/slurm_jarvis_mattergen_tc_finetune_train.out $REPO/slurm_jarvis_mattergen_tc_finetune_infer.out"
)

EXPS=(
    agpt_benchmark_alex
    agpt_benchmark_jarvis
    cdvae_benchmark_alex
    cdvae_benchmark_jarvis
    flowmm_benchmark_alex
    flowmm_benchmark_jarvis
    mattergen_benchmark_alex
    mattergen_benchmark_jarvis
    mattergen_tc_finetune_benchmark_alex
    mattergen_tc_finetune_benchmark_jarvis
)

# ── Utility: human-readable mtime age ────────────────────────────────────────
age_str() {
    local f="$1"
    [[ ! -e "$f" ]] && printf -- "-" && return
    local age
    age=$(( $(date +%s) - $(stat -c %Y "$f") ))
    if   (( age < 60 ));    then printf "%ds ago"   "$age"
    elif (( age < 3600 ));  then printf "%.0fm ago" "$(awk "BEGIN{printf \"%.0f\",$age/60}")"
    elif (( age < 86400 )); then printf "%.1fh ago" "$(awk "BEGIN{printf \"%.1f\",$age/3600}")"
    else                         printf "%.1fd ago" "$(awk "BEGIN{printf \"%.1f\",$age/86400}")"
    fi
}

# touch file: ✓ age / ✗
touch_status() {
    local f="$1"
    if [[ -f "$f" ]]; then
        printf "${GRN}✓${RST} %s" "$(age_str "$f")"
    else
        printf "${RED}✗${RST}"
    fi
}

chk()   { [[ -e "$1" ]] && printf "${GRN}✓${RST}" || printf "${RED}✗${RST}"; }
tick()  { printf "${GRN}✓${RST}"; }
cross() { printf "${RED}✗${RST}"; }

# bench CSV row count (handles agpt saved/ subdir)
bench_csv_rows() {
    local exp="$1"
    local csv1="$JOB_RUNS/$exp/AI-AtomGen-prop-dft_3d-test-rmse.csv"
    local csv2="$JOB_RUNS/$exp/saved/AI-AtomGen-prop-dft_3d-test-rmse.csv"
    local csv=""
    [[ -f "$csv1" ]] && csv="$csv1"
    [[ -z "$csv" && -f "$csv2" ]] && csv="$csv2"
    if [[ -n "$csv" ]]; then
        printf "%d" "$(( $(wc -l < "$csv") - 1 ))"
    else
        printf "${DIM}-%s${RST}" ""
    fi
}

# most recently modified file from a list of paths (safe against missing files)
latest_of() {
    { ls -t "$@" 2>/dev/null || true; } | head -1
}

slurm_info() {
    local log="$1"
    if [[ -z "$log" || ! -f "$log" ]]; then
        printf "${DIM}(no log)${RST}"
    else
        printf "%s [%s]" "$(basename "$log")" "$(age_str "$log")"
    fi
}

# model-specific key artifact probe
key_artifact() {
    local exp="$1"
    local dir="$JOB_RUNS/$exp"
    case "$exp" in
    agpt_benchmark_*)
        if [[ -d "$dir/saved" ]] && ls "$dir/saved/" &>/dev/null; then
            printf "${GRN}saved/ dir exists${RST}"
        else
            printf "${DIM}no saved/${RST}"
        fi
        ;;
    cdvae_benchmark_*)
        local hydra="$dir/hydra_outputs/singlerun"
        local run_dir
        run_dir=$({ ls -dt "$hydra"/????-??-??/ 2>/dev/null || true; } | head -1)
        if [[ -z "$run_dir" ]]; then
            printf "${DIM}no hydra run${RST}"
        else
            local model_dir
            model_dir=$({ ls -d "$run_dir"*/ 2>/dev/null || true; } | head -1)
            if [[ -n "$model_dir" ]]; then
                local pt
                pt=$({ find "$model_dir" -name "eval_recon.pt" 2>/dev/null || true; } | head -1)
                if [[ -n "$pt" ]]; then
                    printf "${GRN}eval_recon.pt found${RST}"
                else
                    printf "${YLW}hydra run, no eval_recon.pt${RST}"
                fi
            else
                printf "${YLW}singlerun dir, no model subdir${RST}"
            fi
        fi
        ;;
    flowmm_benchmark_*)
        local ckpt
        ckpt=$({ find "$dir/outputs" -name "*.ckpt" 2>/dev/null || true; } | head -1)
        if [[ -z "$ckpt" ]]; then
            printf "${DIM}no .ckpt${RST}"
        else
            local recon
            recon=$({ find "$dir/outputs" -name "consolidated_reconstruct.pt" 2>/dev/null || true; } | head -1)
            if [[ -n "$recon" ]]; then
                printf "${GRN}consolidated_reconstruct.pt${RST}"
            else
                printf "${YLW}.ckpt found, no consolidated${RST}"
            fi
        fi
        ;;
    mattergen_benchmark_* | mattergen_tc_finetune_benchmark_*)
        local extxyz="$dir/results/generated_crystals.extxyz"
        if [[ -f "$extxyz" ]]; then
            printf "${GRN}generated_crystals.extxyz (%d lines)${RST}" "$(wc -l < "$extxyz")"
        elif [[ -d "$dir/outputs" ]] && ls "$dir/outputs/" &>/dev/null; then
            printf "${YLW}outputs/ exists, no extxyz yet${RST}"
        else
            printf "${DIM}no outputs/${RST}"
        fi
        ;;
    *) printf "${DIM}?${RST}" ;;
    esac
}

read_json_num() {
    grep -oP "\"${2}\"\s*:\s*\K[0-9.eE+\-]+" "$1" 2>/dev/null | head -1
}

# Pad a string (which may contain ANSI codes) to N visible columns by appending spaces.
vpad() {
    local str="$1" width="$2"
    local visible
    visible=$(printf '%s' "$str" | sed 's/\x1b\[[0-9;]*m//g')
    local pad=$(( width - ${#visible} ))
    (( pad < 0 )) && pad=0
    printf '%s%*s' "$str" "$pad" ''
}

sep() { printf "\n${BLD}── %s %s${RST}\n" "$1" "$(printf '%.0s─' {1..60} | cut -c1-$(( 65 - ${#1} )))"; }

# ── SLURM detection ───────────────────────────────────────────────────────────
# Returns 0 (true) if any known atombench job names are in the queue for this user.
# Also sets ACTIVE_QUEUE_RAW to the squeue output for later use.
ACTIVE_QUEUE_RAW=""
SQUEUE_AVAILABLE=false
command -v squeue &>/dev/null && SQUEUE_AVAILABLE=true

query_active_jobs() {
    $SQUEUE_AVAILABLE || return 1
    # Fetch all user jobs: JOBID STATE ELAPSED TIME_LIMIT JOBNAME NODELIST
    ACTIVE_QUEUE_RAW=$(squeue --me --noheader \
        --format="%-10i %-10T %-12M %-12l %-16j %N" 2>/dev/null) || true

    local known_names
    known_names=$(printf '%s\n' "${!JOB_TO_EXP[@]}" | tr '\n' '|' | sed 's/|$//')
    echo "$ACTIVE_QUEUE_RAW" | grep -qE "($known_names)"
}

# Lookup state of a specific job name in the cached queue snapshot.
# Prints "RUNNING", "PENDING", or "" if not found.
job_state_in_queue() {
    local jname="$1"
    echo "$ACTIVE_QUEUE_RAW" | awk -v n="$jname" '$5 == n { print $2; exit }'
}

# True if any of the experiment's known job names are in the queue.
exp_in_queue() {
    local exp="$1"
    local jnames="${EXP_JOBS[$exp]:-}"
    for jn in $jnames; do
        local st
        st=$(job_state_in_queue "$jn")
        [[ -n "$st" ]] && return 0
    done
    return 1
}

# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY SECTIONS (each is a function so we can call from the refresh loop)
# ═════════════════════════════════════════════════════════════════════════════

section_header() {
    local mode_label="$1"
    printf "\n"
    printf "${BLD}${CYN}╔══════════════════════════════════════════════════════════════╗${RST}\n"
    printf "${BLD}${CYN}║   ATOMBENCH PIPELINE STATUS   %s  ║${RST}\n" "$(date '+%Y-%m-%d %H:%M:%S')"
    printf "${BLD}${CYN}╚══════════════════════════════════════════════════════════════╝${RST}\n"
    printf "  repo: %s  %b\n" "$REPO" "$mode_label"
}

section_prerequisites() {
    sep "PREREQUISITES"
    printf "\n  ${BLD}Conda envs:${RST}\n"
    for env_name in atomgpt cdvae flowmm mattergen; do
        printf "    %-22s " "${env_name}_env.created"
        touch_status "$REPO/${env_name}_env.created"
        printf "\n"
    done
    printf "\n  ${BLD}Dataset preparation:${RST}\n"
    for touch_name in alex_data.created jarvis_data.created flowmm_yamls.created all_envs_ready.txt; do
        printf "    %-26s " "$touch_name"
        touch_status "$REPO/$touch_name"
        printf "\n"
    done
}

section_data_dirs() {
    sep "DATA DIRECTORIES (model-format splits on disk)"
    printf "\n"
    _check_dir() {
        local label="$1" path="$2"
        printf "  "
        if [[ -d "$path" ]]; then
            local n
            n=$(find "$path" -maxdepth 1 -name "*.csv" 2>/dev/null | wc -l)
            tick; printf "  %-54s  ${DIM}%d CSVs${RST}\n" "$label" "$n"
        else
            cross; printf "  %-54s  ${DIM}missing${RST}\n" "$label"
        fi
    }
    _check_dir "models/cdvae/data/alexandria/"                           "$REPO/models/cdvae/data/alexandria"
    _check_dir "models/cdvae/data/supercon/"                             "$REPO/models/cdvae/data/supercon"
    _check_dir "models/flowmm/data/alexandria/"                          "$REPO/models/flowmm/data/alexandria"
    _check_dir "models/flowmm/data/supercon/"                            "$REPO/models/flowmm/data/supercon"
    _check_dir "models/mattergen/datasets/cache/alex_atombench/"         "$REPO/models/mattergen/datasets/cache/alex_atombench"
    _check_dir "models/mattergen/datasets/cache/alex_atombench_tc/"      "$REPO/models/mattergen/datasets/cache/alex_atombench_tc"
    _check_dir "models/mattergen/datasets/cache/supercon_atombench/"     "$REPO/models/mattergen/datasets/cache/supercon_atombench"
    _check_dir "models/mattergen/datasets/cache/supercon_atombench_tc/"  "$REPO/models/mattergen/datasets/cache/supercon_atombench_tc"
}

section_pipeline_table() {
    sep "EXPERIMENT PIPELINES"
    printf "\n"
    # Column widths (visible chars): exp=46 trained=16 inferred=16 final=16 csv=8 artifact=34 log=28
    printf "  %-46s  %-16s %-16s %-16s  %8s  %-34s  %s\n" \
        "Experiment" "TRAINED" "INFERRED" "FINAL" "CSV rows" "Key artifact" "Latest SLURM log"
    printf "  %-46s  %-16s %-16s %-16s  %8s  %-34s  %s\n" \
        "$(printf '%.0s─' {1..46})" "$(printf '%.0s─' {1..16})" \
        "$(printf '%.0s─' {1..16})" "$(printf '%.0s─' {1..16})" \
        "$(printf '%.0s─' {1..8})"  "$(printf '%.0s─' {1..34})" \
        "$(printf '%.0s─' {1..28})"

    INFLIGHT_EXPS=()

    for EXP in "${EXPS[@]}"; do
        local TRAINED_F="$REPO/${EXP}.trained"
        local INFERRED_F="$REPO/${EXP}.inferred"
        local FINAL_F="$REPO/${EXP}.final"

        local TRAINED_COL INFERRED_COL FINAL_COL
        case "$EXP" in
        agpt_benchmark_*)
            TRAINED_COL="${DIM}N/A${RST}"; INFERRED_COL="${DIM}N/A${RST}" ;;
        *)
            TRAINED_COL="$(touch_status "$TRAINED_F")"
            INFERRED_COL="$(touch_status "$INFERRED_F")" ;;
        esac
        FINAL_COL="$(touch_status "$FINAL_F")"

        # Queue state annotation (live mode only)
        local QUEUE_ANN=""
        if $LIVE_MODE; then
            local jnames="${EXP_JOBS[$EXP]:-}"
            for jn in $jnames; do
                local st
                st=$(job_state_in_queue "$jn")
                if [[ "$st" == "RUNNING" ]]; then
                    QUEUE_ANN=" ${GRN}[RUNNING]${RST}"; break
                elif [[ "$st" == "PENDING" ]]; then
                    QUEUE_ANN=" ${YLW}[PENDING]${RST}"
                fi
            done
        fi

        local CSV_ROWS ARTIFACT LOG LOG_INFO
        CSV_ROWS="$(bench_csv_rows "$EXP")"
        ARTIFACT="$(key_artifact "$EXP")"
        LOG="$(latest_of ${SLURM_LOGS[$EXP]})"
        LOG_INFO="$(slurm_info "$LOG")"

        # Use vpad so ANSI codes don't break column widths
        printf "  %s  %s  %s  %s  %8s  %s  %s\n" \
            "$(vpad "${EXP}${QUEUE_ANN}" 46)" \
            "$(vpad "$TRAINED_COL"  16)" \
            "$(vpad "$INFERRED_COL" 16)" \
            "$(vpad "$FINAL_COL"    16)" \
            "$CSV_ROWS" \
            "$(vpad "$ARTIFACT" 34)" \
            "$LOG_INFO"

        [[ ! -f "$FINAL_F" ]] && INFLIGHT_EXPS+=("$EXP")
    done
}

section_active_slurm() {
    # Only called in live mode.
    sep "ACTIVE SLURM JOBS"
    printf "\n"

    if [[ -z "$ACTIVE_QUEUE_RAW" ]]; then
        printf "  ${DIM}(squeue returned no output)${RST}\n"
        return
    fi

    # Print a header, then all atombench-relevant rows with experiment annotation.
    printf "  ${BLD}%-10s %-12s %-12s %-13s %-16s %-30s %s${RST}\n" \
        "JOBID" "STATE" "ELAPSED" "TIME LIMIT" "JOB NAME" "EXPERIMENT" "NODE"
    printf "  %-10s %-12s %-12s %-13s %-16s %-30s %s\n" \
        "$(printf '%.0s─' {1..10})" "$(printf '%.0s─' {1..12})" \
        "$(printf '%.0s─' {1..12})" "$(printf '%.0s─' {1..13})" \
        "$(printf '%.0s─' {1..16})" "$(printf '%.0s─' {1..30})" \
        "$(printf '%.0s─' {1..12})"

    local found_any=false
    local known_names
    known_names=$(printf '%s\n' "${!JOB_TO_EXP[@]}" | tr '\n' '|' | sed 's/|$//')

    while IFS= read -r row; do
        [[ -z "$row" ]] && continue
        local jobid state elapsed timelim jobname node
        read -r jobid state elapsed timelim jobname node <<< "$row"
        # Only show rows whose job name is in our known list
        echo "$jobname" | grep -qE "^($known_names)$" || continue
        found_any=true

        local exp_label="${JOB_TO_EXP[$jobname]:-$jobname}"
        local state_colored
        case "$state" in
            RUNNING)  state_colored="${GRN}RUNNING${RST}"  ;;
            PENDING)  state_colored="${YLW}PENDING${RST}"  ;;
            FAILED*)  state_colored="${REDB}FAILED${RST}"  ;;
            CANCELLED*)state_colored="${RED}CANCELLED${RST}" ;;
            *)        state_colored="${DIM}$state${RST}"   ;;
        esac

        printf "  %-10s %-22b %-12s %-13s %-16s %-30s %s\n" \
            "$jobid" "$state_colored" "$elapsed" "$timelim" "$jobname" "$exp_label" "${node:--}"
    done <<< "$ACTIVE_QUEUE_RAW"

    if ! $found_any; then
        printf "  ${DIM}No known atombench jobs found in queue.${RST}\n"
    fi

    # Also show runner if present
    local runner_row
    runner_row=$(echo "$ACTIVE_QUEUE_RAW" | awk '$5 == "runner"' | head -1)
    if [[ -n "$runner_row" ]]; then
        : # already handled above via JOB_TO_EXP
    fi
}

section_slurm_tails() {
    local live="$1"  # true or false
    sep "SLURM LOG TAILS (in-flight experiments)"

    if [[ ${#INFLIGHT_EXPS[@]} -eq 0 ]]; then
        printf "\n  ${GRN}All experiments have reached .final — nothing in flight.${RST}\n"
        return
    fi

    # In live mode: show more lines and prioritize RUNNING jobs first.
    local tail_lines=6
    $live && tail_lines=10

    local showed_any=false
    local running_exps=() pending_exps=() other_exps=()

    if $live; then
        for EXP in "${INFLIGHT_EXPS[@]}"; do
            local jnames="${EXP_JOBS[$EXP]:-}"
            local best_state=""
            for jn in $jnames; do
                local st
                st=$(job_state_in_queue "$jn")
                if [[ "$st" == "RUNNING" ]]; then best_state="RUNNING"; break; fi
                [[ "$st" == "PENDING" ]] && best_state="PENDING"
            done
            case "$best_state" in
                RUNNING) running_exps+=("$EXP") ;;
                PENDING) pending_exps+=("$EXP") ;;
                *)       other_exps+=("$EXP")   ;;
            esac
        done
        # Show RUNNING first, then PENDING, then other in-flight
        INFLIGHT_EXPS=("${running_exps[@]}" "${pending_exps[@]}" "${other_exps[@]}")
    fi

    for EXP in "${INFLIGHT_EXPS[@]}"; do
        LOG="$(latest_of ${SLURM_LOGS[$EXP]})"
        [[ -z "$LOG" || ! -f "$LOG" ]] && continue
        showed_any=true

        # State annotation for live mode
        local state_ann=""
        if $live; then
            local jnames="${EXP_JOBS[$EXP]:-}"
            for jn in $jnames; do
                local st
                st=$(job_state_in_queue "$jn")
                if [[ "$st" == "RUNNING" ]]; then
                    state_ann="  ${GRN}● RUNNING${RST}"; break
                elif [[ "$st" == "PENDING" ]]; then
                    state_ann="  ${YLW}◌ PENDING${RST}"
                fi
            done
        fi

        printf "\n  ${BLD}%s${RST}%b  ${DIM}← %s  [%s]${RST}\n" \
            "$EXP" "$state_ann" "$(basename "$LOG")" "$(age_str "$LOG")"

        tail -n 60 "$LOG" | grep -v '^[[:space:]]*$' | tail -n "$tail_lines" \
        | while IFS= read -r line; do
            if echo "$line" | grep -qiE 'error|traceback|failed|oom|killed|exception|abort|cuda out'; then
                printf "    ${REDB}%s${RST}\n" "$line"
            elif echo "$line" | grep -qiE '^done$|elapsed:|warning:|warn:|epoch [0-9]|step [0-9]|loss|it/s|it\]'; then
                printf "    ${YLW}%s${RST}\n" "$line"
            else
                printf "    ${DIM}%s${RST}\n" "$line"
            fi
        done
    done

    if ! $showed_any; then
        printf "\n  ${DIM}No SLURM logs found yet — jobs may not have been submitted, or logs are not at repo root.${RST}\n"
    fi
}

section_metrics() {
    sep "METRICS SUMMARY (from metrics.json)"
    printf "\n"
    local found=false
    for EXP in "${EXPS[@]}"; do
        local MJSON="$JOB_RUNS/$EXP/metrics.json"
        [[ ! -f "$MJSON" ]] && continue
        found=true

        local MATCH_RATE N_MATCHED N_TOTAL CCRMSE
        MATCH_RATE=$(read_json_num "$MJSON" "match_rate")
        N_MATCHED=$(read_json_num  "$MJSON" "n_matched")
        N_TOTAL=$(read_json_num    "$MJSON" "n_total")
        CCRMSE=$(read_json_num     "$MJSON" "value")

        local MAE_AVG KLD_AVG
        MAE_AVG=$(python3 -c "
import json
try:
    d = json.load(open('$MJSON'))
    v = list(d['MAE']['average_mae'].values())
    print(f'{sum(v)/len(v):.4f}')
except: print('-')
" 2>/dev/null)
        KLD_AVG=$(python3 -c "
import json
try:
    d = json.load(open('$MJSON'))
    v = list(d['KLD'].values())
    print(f'{sum(v)/len(v):.4f}')
except: print('-')
" 2>/dev/null)

        printf "  ${BLD}%-48s${RST}" "$EXP"
        printf "  match_rate=${GRN}%-6s${RST}" "${MATCH_RATE:--}"
        printf " (%s/%s)" "${N_MATCHED:--}" "${N_TOTAL:--}"
        printf "  ccRMSE=${CYN}%-8s${RST}" "${CCRMSE:--}"
        printf "  MAE_avg=${YLW}%-8s${RST}" "${MAE_AVG:--}"
        printf "  KLD_avg=${YLW}%s${RST}" "${KLD_AVG:--}"
        printf "\n"
    done
    if ! $found; then
        printf "  ${DIM}No metrics.json found yet. Run: snakemake metrics.computed${RST}\n"
    fi
}

section_postprocessing() {
    sep "POST-PROCESSING & CONSOLIDATED OUTPUTS"
    printf "\n  ${BLD}Pipeline completion steps:${RST}\n"
    for touch_name in \
        metrics.computed benchmarks.verified charts.made \
        overlay_charts.created grid_charts.created \
        rmse_chart.made crystal_system_mae_charts.created
    do
        printf "    "
        touch_status "$REPO/$touch_name"
        printf "  %-36s" "$touch_name"
        [[ -f "$REPO/$touch_name" ]] && printf "  ${DIM}(%s)${RST}" "$(age_str "$REPO/$touch_name")"
        printf "\n"
    done

    printf "\n  ${BLD}Consolidated metrics table:${RST}\n"
    local EPIC="$JOB_RUNS/epic_metrics.csv"
    printf "    "
    if [[ -f "$EPIC" ]]; then
        local rows=$(( $(wc -l < "$EPIC") - 1 ))
        tick; printf "  job_runs/epic_metrics.csv  ${DIM}(%d experiment rows, %s)${RST}\n" \
            "$rows" "$(age_str "$EPIC")"
    else
        cross; printf "  job_runs/epic_metrics.csv  ${DIM}(not yet created)${RST}\n"
    fi
    printf "\n"
}

# ═════════════════════════════════════════════════════════════════════════════
# FULL DISPLAY — called once per render cycle
# ═════════════════════════════════════════════════════════════════════════════
LIVE_MODE=false
INFLIGHT_EXPS=()

render() {
    local mode_label="$1"
    # Refresh queue snapshot before every render in live mode
    $LIVE_MODE && query_active_jobs 2>/dev/null || true

    section_header "$mode_label"
    section_prerequisites
    section_data_dirs
    section_pipeline_table        # also populates INFLIGHT_EXPS
    $LIVE_MODE && section_active_slurm
    section_slurm_tails "$LIVE_MODE"
    section_metrics
    section_postprocessing
}

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
main() {
    # Initial queue query to decide mode
    if ! $FORCE_ONCE && $SQUEUE_AVAILABLE && query_active_jobs 2>/dev/null; then
        LIVE_MODE=true
    fi

    if $LIVE_MODE; then
        trap 'printf "\n\n  Interrupted — exiting live mode.\n\n"; exit 0' INT TERM
        local mode_label="${CYN}[LIVE — refreshing every ${REFRESH_SECS}s — Ctrl+C to exit]${RST}"
        while true; do
            clear
            render "$mode_label"
            sleep "$REFRESH_SECS"
            # Re-check if jobs are still active; drop to static if they're gone
            if ! query_active_jobs 2>/dev/null; then
                clear
                LIVE_MODE=false
                render "${DIM}[jobs finished — final snapshot]${RST}"
                printf "\n  ${GRN}No more atombench jobs in queue. Exiting live mode.${RST}\n\n"
                break
            fi
        done
    else
        render "${DIM}[static snapshot]${RST}"
    fi
}

main

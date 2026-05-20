#!/usr/bin/env bash
# depcheck.sh — verbose dependency checker for Linux + Slurm + CUDA module + conda hook (+ mamba compatibility)
# Usage: bash depcheck.sh
#
# This script is intentionally chatty: it explains *exactly* what failed and how to fix it.

set -u  # treat unset vars as errors (but we won't use set -e; we want custom diagnostics)

PASS_COUNT=0
FAIL_COUNT=0

hr() { printf '%s\n' "------------------------------------------------------------"; }
ok() { printf '✅ %s\n' "$1"; PASS_COUNT=$((PASS_COUNT+1)); }
warn() { printf '⚠️  %s\n' "$1"; }
fail() { printf '❌ %s\n' "$1"; FAIL_COUNT=$((FAIL_COUNT+1)); }

section() {
  hr
  printf '%s\n' "$1"
  hr
}

# Run a command, capture output, and report with context.
run_capture() {
  # args: <label> <command...>
  local label="$1"; shift
  local out rc
  out="$("$@" 2>&1)"; rc=$?
  printf '%s\n' "$out"
  return "$rc"
}

# ---------- 1) Linux check ----------
check_linux() {
  section "1) OS check (must be Linux)"

  local uname_s
  uname_s="$(uname -s 2>/dev/null || true)"

  if [[ "$uname_s" == "Linux" ]]; then
    ok "Detected Linux (uname -s = Linux)."
    return 0
  fi

  fail "Not Linux: uname -s returned '${uname_s:-<empty>}'.
This script expects a Linux environment (typical for HPC login nodes)."
  return 1
}

# ---------- 2) Slurm present ----------
check_slurm() {
  section "2) Slurm check (must have Slurm commands available)"

  # Prefer squeue; fallback to sbatch/sinfo
  if command -v squeue >/dev/null 2>&1; then
    ok "Found Slurm command: squeue ($(command -v squeue))."
  elif command -v sbatch >/dev/null 2>&1; then
    ok "Found Slurm command: sbatch ($(command -v sbatch))."
  elif command -v sinfo >/dev/null 2>&1; then
    ok "Found Slurm command: sinfo ($(command -v sinfo))."
  else
    fail "Slurm does not appear to be present in PATH.
Tried: squeue, sbatch, sinfo.
Fix: load your site environment/modules, or use the correct login node, or add Slurm bin to PATH."
    return 1
  fi

  # Optional: sanity-call squeue (doesn't require allocations, but may fail if Slurm is down)
  if command -v squeue >/dev/null 2>&1; then
    local out rc
    out="$(squeue -h 2>&1)"; rc=$?
    if [[ $rc -eq 0 ]]; then
      ok "squeue runs successfully."
    else
      warn "squeue exists but returned a non-zero exit code ($rc).
Output:
$out
This may indicate a transient Slurm/controller issue, or a permissions/config problem."
    fi
  fi

  return 0
}

# ---------- 3) module load cuda/11.5.0 works ----------
check_cuda_module() {
  section "3) Environment Modules check (module load cuda/11.5.0 must work)"

  # Many clusters define `module` as a shell function; ensure it's available.
  if ! type module >/dev/null 2>&1; then
    warn "The 'module' command is not currently available in this shell."
    warn "Attempting to initialize Environment Modules (common paths)..."

    # Common init scripts across distros/HPC sites:
    if [[ -f /etc/profile.d/modules.sh ]]; then
      # shellcheck disable=SC1091
      source /etc/profile.d/modules.sh
    elif [[ -f /usr/share/Modules/init/bash ]]; then
      # shellcheck disable=SC1091
      source /usr/share/Modules/init/bash
    elif [[ -f /usr/share/modules/init/bash ]]; then
      # shellcheck disable=SC1091
      source /usr/share/modules/init/bash
    fi
  fi

  if ! type module >/dev/null 2>&1; then
    fail "Still cannot find the 'module' command after attempted initialization.
Fix options:
- Start a login shell (e.g., 'bash -l') so /etc/profile is sourced
- Ask your cluster docs where modules are initialized
- Ensure Environment Modules / Lmod is installed on the node"
    return 1
  fi

  # Try loading CUDA 11.8
  local out rc
  out="$(module load cuda/11.5.0 2>&1)"; rc=$?
  if [[ $rc -eq 0 ]]; then
    ok "module load cuda/11.5.0 succeeded."
  else
    fail "module load cuda/11.5.0 FAILED (exit code $rc).
Output:
$out
Likely causes:
- That module name/version doesn't exist on this system
- You need to run 'module avail cuda' to see valid CUDA modules
- Your modules cache or MODULEPATH is misconfigured"
    return 1
  fi

  # Optional: verify nvcc or nvidia-smi if expected
  if command -v nvcc >/dev/null 2>&1; then
    ok "nvcc is now in PATH ($(command -v nvcc))."
  else
    warn "nvcc not found in PATH even after loading cuda/11.5.0.
This can be normal on some sites (they may only set runtime libs). If you need nvcc, check the module’s contents."
  fi

  return 0
}

# ---------- 4) conda hook works + "reloads" base after deactivate ----------
check_conda_hook() {
  section "4) Conda hook check (eval \"\$(conda shell.bash hook)\" + base availability)"

  # Run in a clean login shell, but feed the script via heredoc to avoid quoting hazards.
  local out rc
  out="$(
    bash -lc "$(cat <<'CONDA_CHECK_SCRIPT'
set -u

echo "[INFO] Shell: $0"
echo "[INFO] Starting PATH: $PATH"

ok()   { echo "✅ $*"; }
warn() { echo "⚠️  $*"; }
fail() { echo "❌ $*"; exit 1; }

# 4a) Ensure conda executable exists somewhere (hook needs it)
if ! command -v conda >/dev/null 2>&1; then
  fail "conda executable not found in PATH.
Fix: add Miniconda/Anaconda bin to PATH or source conda.sh, e.g.
  source <miniconda>/etc/profile.d/conda.sh
or ensure your ~/.bashrc conda init block runs in login shells."
fi
ok "Found conda executable: $(command -v conda)"

# 4b) Try to deactivate (may be harmless if not active)
echo "[INFO] Attempting conda deactivate (may be a no-op)..."
conda deactivate >/dev/null 2>&1 || true

echo "[INFO] Before hook: $(type -t conda 2>/dev/null || echo "<missing>")"

# 4c) Run the hook
echo "[INFO] Running: eval \"\$(conda shell.bash hook)\""
HOOK_OUT="$(conda shell.bash hook 2>&1)" || {
  echo "$HOOK_OUT"
  fail "conda shell.bash hook failed."
}
eval "$HOOK_OUT" || fail "eval of conda hook output failed."

echo "[INFO] After hook: $(type -t conda 2>/dev/null || echo "<missing>")"

# 4d) Verify that conda is now properly initialized
BASE_DIR="$(conda info --base 2>/dev/null || true)"
if [[ -z "$BASE_DIR" ]]; then
  fail "conda appears present, but conda info --base did not return a base directory.
This suggests partial/broken initialization."
fi
ok "conda base directory: $BASE_DIR"

# 4e) Confirm we can activate base (this is the closest meaningful reload test)
echo "[INFO] Attempting: conda activate base"
ACT_OUT="$(conda activate base 2>&1)" || {
  echo "$ACT_OUT"
  fail "conda activate base failed."
}

if [[ "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
  ok "Successfully activated base environment (CONDA_DEFAULT_ENV=base)."
else
  warn "conda activate base did not set CONDA_DEFAULT_ENV=base (got: ${CONDA_DEFAULT_ENV:-<unset>}).
Activation may still have worked, but environment variables look odd."
fi

if [[ "$(type -t conda)" == "function" ]]; then
  ok "conda is initialized as a shell function (good)."
else
  warn "conda is not a shell function after hook (type -t conda = $(type -t conda)).
This can still work, but usually indicates incomplete shell integration."
fi

echo "[INFO] Conda hook test complete."
CONDA_CHECK_SCRIPT
)"
  )"
  rc=$?

  printf '%s\n' "$out"

  if [[ $rc -eq 0 ]]; then
    ok "Conda hook + base activation check passed."
    return 0
  else
    fail "Conda hook + base activation check FAILED (see output above)."
    return 1
  fi
}

# ---------- 5) mamba / micromamba present OR conda is compatible with installing mamba ----------
check_mamba_compat() {
  section "5) Mamba check (already installed, or conda can install it cleanly)"

  local out rc
  out="$(
    bash -lc "$(cat <<'MAMBA_CHECK_SCRIPT'
set -u

echo "[INFO] Shell: $0"
echo "[INFO] Starting PATH: $PATH"

ok()   { echo "✅ $*"; }
warn() { echo "⚠️  $*"; }
fail() { echo "❌ $*"; exit 1; }
inconclusive() { echo "⚠️  $*"; exit 2; }

# 5a) If micromamba exists, we are done (doesn't depend on conda).
if command -v micromamba >/dev/null 2>&1; then
  ok "Found micromamba: $(command -v micromamba)"
  micromamba --version 2>/dev/null || true
  ok "Mamba-family tooling is already available (micromamba)."
  exit 0
fi

# 5b) Need conda for mamba (classic) checks.
if ! command -v conda >/dev/null 2>&1; then
  inconclusive "conda is not in PATH, so I can't assess mamba-via-conda compatibility.
Fix: ensure conda is available (see section 4)."
fi
ok "Found conda executable: $(command -v conda)"

# Initialize conda shell integration (mirrors section 4, but self-contained).
HOOK_OUT="$(conda shell.bash hook 2>&1)" || {
  echo "$HOOK_OUT"
  inconclusive "conda shell.bash hook failed here, so I can't test mamba installation."
}
eval "$HOOK_OUT" || inconclusive "eval of conda hook output failed."

# Activate base (so `mamba` in base would be on PATH).
conda activate base >/dev/null 2>&1 || inconclusive "Could not activate base; can't reliably check for mamba."

# 5c) If mamba already exists, great.
if command -v mamba >/dev/null 2>&1; then
  ok "Found mamba: $(command -v mamba)"
  mamba --version 2>/dev/null || true
  ok "Mamba is already installed and available."
  exit 0
fi

# 5d) Heuristic compatibility signals (channels, pins, conda version).
CONDA_VER_RAW="$(conda --version 2>/dev/null || true)"
echo "[INFO] $CONDA_VER_RAW"

# Extract something like "24.11.0" from "conda 24.11.0"
CONDA_VER="$(echo "$CONDA_VER_RAW" | awk '{print $2}' | tr -d '\r' || true)"
if [[ -z "$CONDA_VER" ]]; then
  warn "Could not parse conda version string. Continuing anyway."
fi

CHANNELS_OUT="$(conda config --show channels 2>/dev/null || true)"
echo "[INFO] conda channels:"
echo "$CHANNELS_OUT"

if echo "$CHANNELS_OUT" | grep -qE '^\s*-\s*conda-forge\s*$'; then
  ok "conda-forge channel is present (best source for installing mamba)."
else
  warn "conda-forge channel is NOT present. Installing mamba often fails or is unavailable without it."
  echo "Fix (recommended):"
  echo "  conda config --add channels conda-forge"
  echo "  conda config --set channel_priority strict"
fi

PINS_OUT="$(conda config --show pinned_packages 2>/dev/null || true)"
echo "[INFO] pinned_packages:"
echo "$PINS_OUT"
if ! echo "$PINS_OUT" | grep -qE 'pinned_packages:\s*\[\s*\]\s*$'; then
  warn "You appear to have pinned packages. Pins can cause mamba installs to be UNSAT."
  echo "Tip: inspect pins via:"
  echo "  conda config --show pinned_packages"
fi

# 5e) Try a DRY-RUN solve for installing mamba (no changes made).
# This may require internet access to fetch repodata.
# Use timeout if available to avoid hanging forever.
DRYRUN_CMD=(conda create -n _mamba_probe -c conda-forge mamba --dry-run -y)

echo "[INFO] Attempting DRY-RUN solve:"
echo "       ${DRYRUN_CMD[*]}"

if command -v timeout >/dev/null 2>&1; then
  # 45s is usually plenty for a simple solve; adjust by setting DEPCHECK_MAMBA_TIMEOUT.
  T="${DEPCHECK_MAMBA_TIMEOUT:-45}"
  echo "[INFO] Using timeout: ${T}s"
  DRYRUN_OUT="$(timeout "${T}"s "${DRYRUN_CMD[@]}" 2>&1)"; RC=$?
else
  DRYRUN_OUT="$("${DRYRUN_CMD[@]}" 2>&1)"; RC=$?
fi

echo "$DRYRUN_OUT"

# timeout returns 124 when it kills the command
if [[ ${RC:-0} -eq 124 ]]; then
  inconclusive "Dry-run timed out. This could be slow metadata fetching or a blocked network."
fi

if [[ ${RC:-0} -eq 0 ]]; then
  ok "Dry-run solve succeeded: this conda installation is compatible with installing mamba."
  echo "Install command (real):"
  echo "  conda install -n base -c conda-forge mamba -y"
  exit 0
fi

# Heuristic: distinguish network issues vs true UNSAT.
if echo "$DRYRUN_OUT" | grep -qiE 'CondaHTTPError|Connection|Failed to establish|Temporary failure|Name or service not known|SSLError|Read timed out|ProxyError|Network is unreachable'; then
  inconclusive "Dry-run failed due to network / channel access issues.
Fix: ensure the node can reach conda channels (or configure your proxy / mirror), or try from a different node."
fi

if echo "$DRYRUN_OUT" | grep -qiE 'UnsatisfiableError|PackagesNotFoundError'; then
  fail "Dry-run indicates mamba install is NOT currently solvable (UNSAT / package not found).
Common fixes:
- Add conda-forge + strict priority:
    conda config --add channels conda-forge
    conda config --set channel_priority strict
- Avoid installing into base; try a fresh env:
    conda create -n mamba_env -c conda-forge mamba
- If base is heavily pinned/brittle, consider Miniforge/Mambaforge or use micromamba."
fi

# Otherwise: unknown failure mode, but it's still a failure for compatibility.
fail "Dry-run failed for an unknown reason (exit code ${RC:-<unknown>}).
Scroll output above for the exact error."
MAMBA_CHECK_SCRIPT
)"
  )"
  rc=$?

  printf '%s\n' "$out"

  if [[ $rc -eq 0 ]]; then
    ok "Mamba compatibility check passed."
    return 0
  elif [[ $rc -eq 2 ]]; then
    warn "Mamba compatibility check was inconclusive (likely network/timeout)."
    return 0
  else
    fail "Mamba compatibility check FAILED (see output above)."
    return 1
  fi
}

main() {
  section "Dependency checker (Linux + Slurm + CUDA module + conda hook + mamba)"

  check_linux || true
  check_slurm || true
  check_cuda_module || true
  check_conda_hook || true
  check_mamba_compat || true

  hr
  printf "Result: %d passed, %d failed\n" "$PASS_COUNT" "$FAIL_COUNT"
  hr

  if [[ $FAIL_COUNT -gt 0 ]]; then
    printf "Overall: ❌ One or more checks failed.\n"
    exit 1
  else
    printf "Overall: ✅ All checks passed.\n"
    exit 0
  fi
}

main "$@"

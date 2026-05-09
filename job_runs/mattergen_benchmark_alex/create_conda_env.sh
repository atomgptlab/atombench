#!/usr/bin/env bash
set -euo pipefail

( while true; do echo "[mattergen env] still working... $(date)"; sleep 60; done ) &
KEEPALIVE_PID=$!
trap 'kill "$KEEPALIVE_PID" 2>/dev/null || true' EXIT

if conda env list | awk '{print $1}' | grep -qx mattergen; then
  : # env already registered
else
  prefix="$(conda info --base)/envs/mattergen"
  if [[ -d "$prefix" && ! -f "$prefix/conda-meta/history" ]]; then
    echo "Error: non-conda folder exists at $prefix. Remove or rename it, then re-run." >&2
    exit 1
  fi
fi

cd models/mattergen

conda create -y -n mattergen python=3.10
set +u
eval "$(conda shell.bash hook)"
conda activate mattergen
set -u

python -c 'import sys; print(sys.version)'
pip --version || (echo "pip missing after activation"; exit 1)

pip install uv
module load cuda/11.8 2>/dev/null || true
uv pip install -e .

touch "${PROJECT_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}/mattergen_env.created"
echo "Done"

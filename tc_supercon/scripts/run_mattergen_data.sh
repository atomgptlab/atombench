#!/bin/bash
# Convert the shared CDVAE-format supercon CSV splits into a mattergen LMDB
# cache.  We reuse the cdvae splits to guarantee identical test sets.
set -euo pipefail

source scripts/absolute_path.sh
ROOT="${ABS_PATH%/}"

CACHE_ROOT="${ROOT}/models/mattergen/datasets/cache"
RAW_DIR="${ROOT}/models/cdvae/data/supercon"

mkdir -p "${CACHE_ROOT}"

conda run -n mattergen --no-capture-output \
    csv-to-dataset \
        --csv-folder   "${RAW_DIR}" \
        --dataset-name supercon_atombench \
        --cache-folder "${CACHE_ROOT}"

echo "Done: mattergen supercon_atombench cache → ${CACHE_ROOT}/supercon_atombench"

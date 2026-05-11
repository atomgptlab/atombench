#!/bin/bash
# Convert CDVAE-format Alexandria CSVs into a mattergen LMDB cache with the
# Tc column renamed to tc_supercon so it is recognised by PROPERTY_SOURCE_IDS.
# Outputs go to the alex_atombench_tc cache dir, leaving the existing
# alex_atombench cache untouched.
set -euo pipefail

source scripts/absolute_path.sh
ROOT="${ABS_PATH%/}"

CACHE_ROOT="${ROOT}/models/mattergen/datasets/cache"
RAW_DIR="${ROOT}/models/cdvae/data/alexandria"
TMP_DIR="${CACHE_ROOT}/.tmp_alex_tc_csv"

mkdir -p "${TMP_DIR}"

conda run -n mattergen --no-capture-output python3 -c "
import pandas as pd, os
raw = '${RAW_DIR}'
tmp = '${TMP_DIR}'
for f in ['train.csv', 'val.csv', 'test.csv']:
    fp = os.path.join(raw, f)
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        df = df.rename(columns={'Tc': 'tc_supercon'})
        df.to_csv(os.path.join(tmp, f), index=False)
"

conda run -n mattergen --no-capture-output \
    csv-to-dataset \
        --csv-folder   "${TMP_DIR}" \
        --dataset-name alex_atombench_tc \
        --cache-folder "${CACHE_ROOT}"

rm -rf "${TMP_DIR}"

echo "Done: mattergen alex_atombench_tc cache → ${CACHE_ROOT}/alex_atombench_tc"

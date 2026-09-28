#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-/workspace/chemrasayan}
AUDIT_SCRIPT=${AUDIT_SCRIPT:-${ROOT}/src/evaluation/openadmet/openadmet_step2_leakage_audit.py}
STEP2_DIR=${STEP2_DIR:-}
PREPARED_DIR=${PREPARED_DIR:-/mnt/data/moljepa_benchmarks/prepared}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/results_openadmet_step2_leakage_audit}
SHARD_ID=${SHARD_ID:?SHARD_ID is required}
NUM_SHARDS=${NUM_SHARDS:-12}
WORKERS=${WORKERS:-8}

export PIP_ROOT_USER_ACTION=ignore
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

pip_retry() {
  local attempt=1
  until "$@"; do
    if (( attempt >= 5 )); then return 1; fi
    echo "pip attempt ${attempt} failed; retrying in 15 seconds" >&2
    sleep 15
    attempt=$((attempt + 1))
  done
}
pip_retry python -m pip install --quiet \
  'numpy==1.26.4' \
  'pandas==2.2.3' \
  'rdkit==2025.3.6' \
  'tables==3.10.1' \
  'FPSim2==0.7.3'

mkdir -p "${OUTPUT_DIR}"
STEP2_ARGS=()
if [[ -n "${STEP2_DIR}" ]]; then
  STEP2_ARGS=(--step2-dir "${STEP2_DIR}")
fi

# One pod creates the standardized corpus and exact FPSim2 index.  Other pods
# block on the shared PVC lock, then reuse the completed immutable index.
(
  flock -x 9
  python "${AUDIT_SCRIPT}" prepare \
    "${STEP2_ARGS[@]}" \
    --prepared-dir "${PREPARED_DIR}" \
    --output-dir "${OUTPUT_DIR}"
  if [[ ! -s "${OUTPUT_DIR}/SETUP_COMPLETE" ]]; then
    index_tmp="${OUTPUT_DIR}/step2_ecfp4_2048.build.${SHARD_ID}.$$.h5"
    fpsim2-create-db \
      "${OUTPUT_DIR}/step2_standardized.smi" \
      "${index_tmp}" \
      --fp_type Morgan \
      --fp_params '{"radius": 2, "fpSize": 2048}' \
      --processes "${WORKERS}"
    mv "${index_tmp}" "${OUTPUT_DIR}/step2_ecfp4_2048.h5"
    python "${AUDIT_SCRIPT}" mark-index \
      "${STEP2_ARGS[@]}" \
      --output-dir "${OUTPUT_DIR}"
  fi
) 9>"${OUTPUT_DIR}/setup.lock"

python "${AUDIT_SCRIPT}" search \
  "${STEP2_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --shard-id "${SHARD_ID}" \
  --num-shards "${NUM_SHARDS}" \
  --workers "${WORKERS}"

# The last successful shard to acquire this lock performs aggregation and the
# independent full-corpus brute-force validation.
(
  flock -x 8
  python "${AUDIT_SCRIPT}" finalize \
    "${STEP2_ARGS[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-shards "${NUM_SHARDS}"
) 8>"${OUTPUT_DIR}/finalize.lock"

date -u +'%Y-%m-%dT%H:%M:%SZ' > "${OUTPUT_DIR}/POD_${SHARD_ID}_DONE"

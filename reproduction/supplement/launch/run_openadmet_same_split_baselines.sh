#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-/workspace/chemrasayan}
DATA_DIR=${DATA_DIR:-/mnt/data/moljepa_benchmarks/prepared}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/results_openadmet_same_split_baselines_20260917_r2}
NEURAL_RESULTS_DIR=${NEURAL_RESULTS_DIR:-/mnt/results_rtd_phase4_95m_moljepa}
SHARD_ID=${SHARD_ID:?SHARD_ID is required}
NUM_SHARDS=${NUM_SHARDS:-8}

export PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
python -m pip install --quiet \
  rdkit==2025.3.6 scikit-learn==1.7.2 lightgbm==4.6.0 pandas==2.3.3
test -s "${DATA_DIR}/manifest.json"
test "$(find "${DATA_DIR}" -maxdepth 1 -name '*.csv' -type f | wc -l)" -eq 23
mkdir -p "${OUTPUT_DIR}"

echo "BASELINE shard=${SHARD_ID}/${NUM_SHARDS}"
python "${ROOT}/openadmet_classical_baselines.py" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --shard-id "${SHARD_ID}" \
  --num-shards "${NUM_SHARDS}" \
  --skip-summary \
  2>&1 | tee -a "${OUTPUT_DIR}/log_shard_${SHARD_ID}.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "${OUTPUT_DIR}/SHARD_${SHARD_ID}_COMPLETE"

(
  flock -x 8
  complete=$(find "${OUTPUT_DIR}" -maxdepth 1 -name 'SHARD_*_COMPLETE' -type f | wc -l)
  if [[ "${complete}" -eq "${NUM_SHARDS}" ]]; then
    test "$(find "${NEURAL_RESULTS_DIR}" -maxdepth 1 -name 'results_*.json' -type f | wc -l)" -eq 23
    python "${ROOT}/summarize_openadmet_classical_baselines.py" \
      --results-dir "${OUTPUT_DIR}" \
      --neural-results-dir "${NEURAL_RESULTS_DIR}"
    date -u +'%Y-%m-%dT%H:%M:%SZ' > "${OUTPUT_DIR}/COMPLETE"
  fi
) 8>"${OUTPUT_DIR}/summary.lock"

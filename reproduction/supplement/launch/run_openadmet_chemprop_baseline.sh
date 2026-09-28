#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-/workspace/chemrasayan}
DATA_DIR=${DATA_DIR:-/mnt/data/moljepa_benchmarks/prepared}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/results_openadmet_chemprop_20260917_r2}
SHARD_ID=${SHARD_ID:?SHARD_ID is required}
NUM_SHARDS=${NUM_SHARDS:-8}

export PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
pip_retry() {
  local attempt=1
  until "$@"; do
    if (( attempt >= 5 )); then return 1; fi
    echo "pip attempt ${attempt} failed; retrying in 15 seconds" >&2
    sleep 15
    attempt=$((attempt + 1))
  done
}
pip_retry python -m pip install --quiet torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip_retry python -m pip install --quiet chemprop==2.3.0 rdkit==2026.3.6 cuik_molmaker_pin==2026.3.6 scikit-learn==1.7.2 pandas==2.3.3

python --version
python -c 'import importlib.metadata, torch; print("chemprop", importlib.metadata.version("chemprop"), "torch", torch.__version__, "cuda", torch.cuda.is_available())'
test -s "${DATA_DIR}/manifest.json"
test "$(find "${DATA_DIR}" -maxdepth 1 -name '*.csv' -type f | wc -l)" -eq 23
mkdir -p "${OUTPUT_DIR}"

python "${ROOT}/openadmet_chemprop_baseline.py" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --shard-id "${SHARD_ID}" \
  --num-shards "${NUM_SHARDS}" \
  --epochs 60 \
  --patience 12 \
  --batch-size 64 \
  --accelerator gpu \
  --num-workers 4 \
  --skip-summary \
  2>&1 | tee -a "${OUTPUT_DIR}/log_shard_${SHARD_ID}.txt"

date -u +'%Y-%m-%dT%H:%M:%SZ' > "${OUTPUT_DIR}/SHARD_${SHARD_ID}_COMPLETE"

(
  flock -x 8
  complete=$(find "${OUTPUT_DIR}" -maxdepth 1 -name 'SHARD_*_COMPLETE' -type f | wc -l)
  if [[ "${complete}" -eq "${NUM_SHARDS}" ]]; then
    python "${ROOT}/openadmet_chemprop_baseline.py" \
      --data-dir "${DATA_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --summarize-only
    date -u +'%Y-%m-%dT%H:%M:%SZ' > "${OUTPUT_DIR}/COMPLETE"
  fi
) 8>"${OUTPUT_DIR}/summary.lock"

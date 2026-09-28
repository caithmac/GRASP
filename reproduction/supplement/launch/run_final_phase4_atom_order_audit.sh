#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-/workspace/chemrasayan}
CODE_ROOT=${CODE_ROOT:-${ROOT}/code}
DATA_DIR=${DATA_DIR:-/mnt/data/moljepa_benchmarks/prepared}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/results_final_phase4_atom_order_20260917_r2}
SHARD_ID=${SHARD_ID:?SHARD_ID is required}
NUM_SHARDS=${NUM_SHARDS:-8}
CHECKPOINT=/mnt/checkpoints/sup_pretrain_rtd25_1.5b_phase4/encoder_step_050000.pt
CHECKPOINT_SHA=7940ca66fc8f6785d6ae2e63b3965f69425e872cd28f4a975493fb1972742ee1

mkdir -p "${OUTPUT_DIR}"
export GIT_PYTHON_REFRESH=quiet PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=4
git config --global --add safe.directory '*'
rm -rf /tmp/DeBERTa /tmp/mole_public
cp -r "${CODE_ROOT}/DeBERTa" /tmp/DeBERTa
cp -r "${CODE_ROOT}/mole_public" /tmp/mole_public
echo "torch==$(python -c 'import torch; print(torch.__version__)')" > /tmp/constraints.txt
python -c 'import torchvision; print("torchvision==" + torchvision.__version__)' >> /tmp/constraints.txt
pip_retry() {
  local attempt=1
  until "$@"; do
    if (( attempt >= 5 )); then return 1; fi
    echo "pip attempt ${attempt} failed; retrying in 15 seconds" >&2
    sleep 15
    attempt=$((attempt + 1))
  done
}
pip_retry pip install --quiet -c /tmp/constraints.txt /tmp/DeBERTa /tmp/mole_public
pip_retry pip install --quiet rdkit==2025.3.6 scikit-learn==1.7.2 scipy==1.15.3
export PYTHONPATH=${ROOT}:${PYTHONPATH:-}

test -s "${DATA_DIR}/manifest.json"
test "$(find "${DATA_DIR}" -maxdepth 1 -name '*.csv' -type f | wc -l)" -eq 23
test "$(sha256sum "${CHECKPOINT}" | cut -d' ' -f1)" = "${CHECKPOINT_SHA}" || {
  echo "ERROR: final Phase4 checkpoint hash mismatch" >&2; exit 4;
}

export MOLJEPA_DATA_DIR=${DATA_DIR} RTD_CKPT=${CHECKPOINT} OUTPUT_DIR
export METHODS=full_mix MIX_LAYERS=4,8,10,12
export EPOCHS=60 PATIENCE=12 BATCH_SIZE=64 NUM_WORKERS=4 USE_BF16=1 FULL_LR=1e-5
export ATOM_ORDER_PERMUTATIONS=10

TASKS=(
  expansion_caco2_pappa expansion_caco2_efflux expansion_logd expansion_ksol
  expansion_hlm expansion_mlm expansion_mbpb expansion_mgmb expansion_mppb
  asap_mers asap_sars asap_logd asap_ksol asap_hlm asap_mlm asap_mdr1 pxr
  biogen_solubility biogen_hlm biogen_rlm biogen_hppb biogen_rppb biogen_mdr1
)
RUNNER_SHA=$(sha256sum "${ROOT}/finetune_moljepa_benchmarks.py" | cut -d' ' -f1)
WRAPPER_SHA=$(sha256sum "${ROOT}/run_final_phase4_atom_order_audit.sh" | cut -d' ' -f1)
SOURCE_BUNDLE_SHA=$( {
  find "${CODE_ROOT}/DeBERTa" "${CODE_ROOT}/mole_public" -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum "${ROOT}/encoder_arch.py" "${ROOT}/moljepa_benchmark_config.py"
} | sha256sum | cut -d' ' -f1 )
printf '%s %s %s %s\n' "${CHECKPOINT_SHA}" "${RUNNER_SHA}" "${WRAPPER_SHA}" "${SOURCE_BUNDLE_SHA}" \
  > "${OUTPUT_DIR}/launch_manifest_shard_${SHARD_ID}.txt"

for ((index=SHARD_ID; index<${#TASKS[@]}; index+=NUM_SHARDS)); do
  task=${TASKS[$index]}
  echo "FINAL-PHASE4 task=${task} shard=${SHARD_ID}/${NUM_SHARDS}"
  TASK_NAME=${task} python "${ROOT}/finetune_moljepa_benchmarks.py" 2>&1 | \
    tee -a "${OUTPUT_DIR}/log_${task}.txt"
  python - "${OUTPUT_DIR}/results_${task}.json" "${CHECKPOINT_SHA}" "${RUNNER_SHA}" <<'PY'
import json, pathlib, sys
path, checkpoint_sha, runner_sha = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
data = json.loads(path.read_text(encoding="utf-8"))
assert data["checkpoint_sha256"] == checkpoint_sha
assert data["runner_sha256"] == runner_sha
assert set(data["method_summary"]) == {"full_mix"}
for split in data["splits"].values():
    robustness = split["methods"]["full_mix"]["atom_order_robustness"]
    assert robustness and robustness["permutations"] == 10
    assert all("fraction_changed_smiles" in record for record in robustness["records"])
PY
done
date -u +'%Y-%m-%dT%H:%M:%SZ' > "${OUTPUT_DIR}/SHARD_${SHARD_ID}_COMPLETE"

(
  flock -x 8
  complete=$(find "${OUTPUT_DIR}" -maxdepth 1 -name 'SHARD_*_COMPLETE' -type f | wc -l)
  if [[ "${complete}" -eq "${NUM_SHARDS}" ]]; then
    python "${ROOT}/summarize_atom_order_robustness.py" \
      --results-dir "${OUTPUT_DIR}" \
      --expected-checkpoint-sha256 "${CHECKPOINT_SHA}"
    date -u +'%Y-%m-%dT%H:%M:%SZ' > "${OUTPUT_DIR}/COMPLETE"
  fi
) 8>"${OUTPUT_DIR}/summary.lock"

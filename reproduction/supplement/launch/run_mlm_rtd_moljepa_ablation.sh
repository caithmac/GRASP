#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-/workspace/chemrasayan}
CODE_ROOT=${CODE_ROOT:-${ROOT}/code}
DATA_DIR=${DATA_DIR:-/mnt/data/moljepa_benchmarks/prepared}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/results_mlm_rtd_moljepa23_rerun_20260917_r3}
SHARD_ID=${SHARD_ID:?SHARD_ID is required}
NUM_SHARDS=${NUM_SHARDS:-12}

readonly MLM_CHECKPOINT=/mnt/checkpoints/sup_pretrain_mlm/encoder_weights.pt
readonly RTD_CHECKPOINT=/mnt/checkpoints/sup_pretrain_rtd25/encoder_weights.pt
MLM_SHA=1160ecaea098f0500d5b6138203f7bd7c5852ef8b34d3d5cc908aeed57c04b86
RTD_SHA=daa0d70f4a7ee44807192ccfe8735696cf26e3b6d10dc0a74bc877e586b34dba

mkdir -p "${OUTPUT_ROOT}"
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

test -s "${DATA_DIR}/manifest.json" || { echo "ERROR: missing prepared MolJEPA data" >&2; exit 3; }
test "$(find "${DATA_DIR}" -maxdepth 1 -name '*.csv' -type f | wc -l)" -eq 23 || {
  echo "ERROR: prepared MolJEPA endpoint set is incomplete" >&2; exit 3;
}
test "$(sha256sum "${MLM_CHECKPOINT}" | cut -d' ' -f1)" = "${MLM_SHA}" || { echo "ERROR: MLM checkpoint hash mismatch" >&2; exit 4; }
test "$(sha256sum "${RTD_CHECKPOINT}" | cut -d' ' -f1)" = "${RTD_SHA}" || { echo "ERROR: RTD checkpoint hash mismatch" >&2; exit 4; }
test "${MLM_CHECKPOINT}" != "${RTD_CHECKPOINT}" || { echo "ERROR: objective checkpoint paths are identical" >&2; exit 4; }
test "${MLM_SHA}" != "${RTD_SHA}" || { echo "ERROR: objective checkpoint hashes are identical" >&2; exit 4; }

# Prove that the encoder tensor payloads consumed by the runner differ.
export MLM_CHECKPOINT RTD_CHECKPOINT OUTPUT_ROOT MLM_SHA RTD_SHA
(
flock -x 7
python - <<'PY'
import hashlib, json, os, pathlib, torch
from encoder_arch import load_encoder_state, resolve_encoder_config

path = pathlib.Path(os.environ["OUTPUT_ROOT"]) / "objective_checkpoint_audit.json"
if path.is_file():
    existing = json.loads(path.read_text(encoding="utf-8"))
    if (
        existing.get("mlm_file_sha256") == os.environ["MLM_SHA"]
        and existing.get("rtd_file_sha256") == os.environ["RTD_SHA"]
        and existing.get("mlm_canonical_tensor_sha256")
        != existing.get("rtd_canonical_tensor_sha256")
        and float(existing.get("global_relative_l2_delta", 0.0)) > 0.0
    ):
        print("objective checkpoint tensor audit already current")
        raise SystemExit(0)

def canonical_state(path):
    state = load_encoder_state(path)
    if state is None or not state:
        raise RuntimeError(f"empty encoder state: {path}")
    config = resolve_encoder_config(state, "auto")
    digest = hashlib.sha256()
    normalized = {}
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        normalized[key] = value
    return normalized, config, digest.hexdigest()

mlm, mlm_config, mlm_digest = canonical_state(os.environ["MLM_CHECKPOINT"])
rtd, rtd_config, rtd_digest = canonical_state(os.environ["RTD_CHECKPOINT"])
if mlm.keys() != rtd.keys():
    raise RuntimeError("MLM/RTD encoder key sets differ; comparison is not architecture-matched")
if mlm_config != rtd_config:
    raise RuntimeError("MLM/RTD resolved encoder configurations differ")
sum_sq_delta = 0.0
sum_sq_rtd = 0.0
max_abs_delta = 0.0
tensors_differing = 0
for key in mlm:
    if mlm[key].shape != rtd[key].shape:
        raise RuntimeError(f"shape mismatch for {key}: {mlm[key].shape} vs {rtd[key].shape}")
    delta = mlm[key].float() - rtd[key].float()
    current_max = float(delta.abs().max()) if delta.numel() else 0.0
    tensors_differing += int(current_max > 0.0)
    max_abs_delta = max(max_abs_delta, current_max)
    sum_sq_delta += float(torch.sum(delta.double().square()))
    sum_sq_rtd += float(torch.sum(rtd[key].double().square()))
relative_l2 = (sum_sq_delta / max(sum_sq_rtd, 1e-300)) ** 0.5
if mlm_digest == rtd_digest or tensors_differing == 0 or relative_l2 == 0.0:
    raise RuntimeError("MLM and RTD loaded encoder tensors are identical")
payload = {
    "schema_version": 1,
    "mlm_file_sha256": os.environ["MLM_SHA"],
    "rtd_file_sha256": os.environ["RTD_SHA"],
    "mlm_canonical_tensor_sha256": mlm_digest,
    "rtd_canonical_tensor_sha256": rtd_digest,
    "parameter_tensors": len(mlm),
    "tensors_differing": tensors_differing,
    "maximum_absolute_tensor_delta": max_abs_delta,
    "global_relative_l2_delta": relative_l2,
    "resolved_encoder_config": mlm_config,
}
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(path)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
) 7>"${OUTPUT_ROOT}/checkpoint_audit.lock"

RUNNER_SHA=$(sha256sum "${ROOT}/finetune_moljepa_benchmarks.py" | cut -d' ' -f1)
WRAPPER_SHA=$(sha256sum "${ROOT}/run_mlm_rtd_moljepa_ablation.sh" | cut -d' ' -f1)
JOB_SHA=$(sha256sum "${ROOT}/mlm_rtd_moljepa_ablation_job.yml" | cut -d' ' -f1)
CHECKPOINT_AUDIT_SHA=$(sha256sum "${OUTPUT_ROOT}/objective_checkpoint_audit.json" | cut -d' ' -f1)
SOURCE_BUNDLE_SHA=$( {
  find "${CODE_ROOT}/DeBERTa" "${CODE_ROOT}/mole_public" -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum "${ROOT}/encoder_arch.py" "${ROOT}/moljepa_benchmark_config.py"
} | sha256sum | cut -d' ' -f1 )
export OUTPUT_ROOT SHARD_ID NUM_SHARDS MLM_CHECKPOINT RTD_CHECKPOINT MLM_SHA RTD_SHA RUNNER_SHA WRAPPER_SHA JOB_SHA CHECKPOINT_AUDIT_SHA SOURCE_BUNDLE_SHA
python - <<'PY'
import json, os, pathlib, platform
root = pathlib.Path(os.environ["OUTPUT_ROOT"])
payload = {
    "schema_version": 1,
    "shard_id": int(os.environ["SHARD_ID"]),
    "num_shards": int(os.environ["NUM_SHARDS"]),
    "checkpoints": {
        "mlm_s2": {"path": os.environ["MLM_CHECKPOINT"], "sha256": os.environ["MLM_SHA"]},
        "rtd25_s2": {"path": os.environ["RTD_CHECKPOINT"], "sha256": os.environ["RTD_SHA"]},
    },
    "runner_sha256": os.environ["RUNNER_SHA"],
    "wrapper_sha256": os.environ["WRAPPER_SHA"],
    "job_yaml_sha256": os.environ["JOB_SHA"],
    "checkpoint_audit_sha256": os.environ["CHECKPOINT_AUDIT_SHA"],
    "source_bundle_sha256": os.environ["SOURCE_BUNDLE_SHA"],
    "python": platform.python_version(),
    "hostname": platform.node(),
}
path = root / f"launch_manifest_shard_{int(os.environ['SHARD_ID']):02d}.json"
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
tmp.replace(path)
PY

export MOLJEPA_DATA_DIR=${DATA_DIR}
export METHODS=frozen_mix,full_mix MIX_LAYERS=4,8,10,12
export EPOCHS=60 PATIENCE=12 BATCH_SIZE=64 NUM_WORKERS=4 USE_BF16=1 FULL_LR=1e-5
export ATOM_ORDER_PERMUTATIONS=0

MODELS=(mlm_s2 rtd25_s2)
TASKS=(
  expansion_caco2_pappa expansion_caco2_efflux expansion_logd expansion_ksol
  expansion_hlm expansion_mlm expansion_mbpb expansion_mgmb expansion_mppb
  asap_mers asap_sars asap_logd asap_ksol asap_hlm asap_mlm asap_mdr1 pxr
  biogen_solubility biogen_hlm biogen_rlm biogen_hppb biogen_rppb biogen_mdr1
)

total=$((${#MODELS[@]} * ${#TASKS[@]}))
for ((combo=SHARD_ID; combo<total; combo+=NUM_SHARDS)); do
  model_index=$((combo / ${#TASKS[@]}))
  task_index=$((combo % ${#TASKS[@]}))
  model=${MODELS[$model_index]}
  task=${TASKS[$task_index]}
  if [ "${model}" = mlm_s2 ]; then checkpoint=${MLM_CHECKPOINT}; else checkpoint=${RTD_CHECKPOINT}; fi
  output_dir=${OUTPUT_ROOT}/${model}
  mkdir -p "${output_dir}"
  echo "RUN objective=${model} endpoint=${task} shard=${SHARD_ID}/${NUM_SHARDS}"
  # Keep the objective-specific checkpoint command-scoped. Exporting RTD_CKPT
  # here mutates the shell variable used to select checkpoints in later loops.
  CUDA_VISIBLE_DEVICES=0 TASK_NAME=${task} RTD_CKPT=${checkpoint} OUTPUT_DIR=${output_dir} \
    python "${ROOT}/finetune_moljepa_benchmarks.py" 2>&1 | \
    tee -a "${output_dir}/log_${task}.txt"
  result_file=${output_dir}/results_${task}.json
  expected_sha=${MLM_SHA}
  if [ "${model}" = rtd25_s2 ]; then expected_sha=${RTD_SHA}; fi
  python - "${result_file}" "${expected_sha}" "${RUNNER_SHA}" <<'PY'
import json, pathlib, sys
path, checkpoint_sha, runner_sha = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
data = json.loads(path.read_text(encoding="utf-8"))
assert data["checkpoint_sha256"] == checkpoint_sha, (path, data["checkpoint_sha256"], checkpoint_sha)
assert data["runner_sha256"] == runner_sha, (path, data["runner_sha256"], runner_sha)
assert set(data["method_summary"]) == {"frozen_mix", "full_mix"}, path
assert len(data["splits"]) == 3, path
PY
done

date -u +'%Y-%m-%dT%H:%M:%SZ' > "${OUTPUT_ROOT}/SHARD_${SHARD_ID}_COMPLETE"
completed=$(find "${OUTPUT_ROOT}" -mindepth 2 -maxdepth 2 -name 'results_*.json' -type f | wc -l)
echo "completed objective/endpoint combinations: ${completed}/46"
if [ "${completed}" -eq 46 ]; then
  (
    flock -x 8
    python "${ROOT}/summarize_mlm_rtd_moljepa_ablation.py" --results-dir "${OUTPUT_ROOT}"
    date -u +'%Y-%m-%dT%H:%M:%SZ' > "${OUTPUT_ROOT}/COMPLETE"
  ) 8>"${OUTPUT_ROOT}/summary.lock"
fi

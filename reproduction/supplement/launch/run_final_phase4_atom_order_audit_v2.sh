#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-/workspace/chemrasayan}
CODE_ROOT=${CODE_ROOT:-${ROOT}/code}
DATA_DIR=${DATA_DIR:-/mnt/data/moljepa_benchmarks/prepared}
REFERENCE_DIR=${REFERENCE_DIR:-/mnt/results_rtd_phase4_95m_moljepa}
OUTPUT_PARENT=${OUTPUT_PARENT:-/mnt/results_final_phase4_atom_order_certifiable_v2}
RUN_ID=${RUN_ID:?RUN_ID is required and must be unique}
SHARD_ID=${SHARD_ID:?SHARD_ID is required}
NUM_SHARDS=${NUM_SHARDS:-8}
JOB_UID=${JOB_UID:?JOB_UID is required}
JOB_NAME=${JOB_NAME:?JOB_NAME is required}
POD_UID=${POD_UID:?POD_UID is required}
POD_NAME=${POD_NAME:?POD_NAME is required}
POD_NAMESPACE=${POD_NAMESPACE:?POD_NAMESPACE is required}
IMAGE_REFERENCE=${IMAGE_REFERENCE:?IMAGE_REFERENCE is required}
IMAGE_ID=${IMAGE_ID:?IMAGE_ID is required}
PRELAUNCH_MANIFEST_SHA256=${PRELAUNCH_MANIFEST_SHA256:?PRELAUNCH_MANIFEST_SHA256 is required}
CHECKPOINT=/mnt/checkpoints/sup_pretrain_rtd25_1.5b_phase4/encoder_step_050000.pt
CHECKPOINT_SHA=7940ca66fc8f6785d6ae2e63b3965f69425e872cd28f4a975493fb1972742ee1
RUNNER=${ROOT}/src/evaluation/openadmet/finetune_moljepa_atom_order_certifiable_v2.py
BASE_RUNNER=${ROOT}/src/evaluation/openadmet/finetune_moljepa_benchmarks.py
WRAPPER=${ROOT}/launch/run_final_phase4_atom_order_audit_v2.sh
CERTIFIER=${ROOT}/src/analysis/certify_atom_order_robustness_v2.py
V1_CERTIFIER=${ROOT}/src/analysis/certify_atom_order_robustness.py
PRELAUNCH_MANIFEST=${PRELAUNCH_MANIFEST:?PRELAUNCH_MANIFEST must point to a finalized deployment manifest}
JOB_TEMPLATE=${ROOT}/manifests/final_phase4_atom_order_job_v2.template.yml

[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$ ]] || {
  echo "ERROR: RUN_ID must be an explicit 8-128 character immutable identifier" >&2; exit 2;
}
[[ "${RUN_ID}" != "REPLACE_WITH_UNIQUE_RUN_ID" ]] || {
  echo "ERROR: replace the RUN_ID placeholder before submission" >&2; exit 2;
}
[[ "${SHARD_ID}" =~ ^[0-7]$ && "${NUM_SHARDS}" = "8" ]] || {
  echo "ERROR: v2 requires exactly eight indexed shards" >&2; exit 3;
}
[[ "${IMAGE_REFERENCE}" =~ @sha256:([0-9a-f]{64})$ ]] || {
  echo "ERROR: IMAGE_REFERENCE must be pinned by an immutable registry digest" >&2; exit 3;
}
[[ "${IMAGE_ID}" = "sha256:${BASH_REMATCH[1]}" ]] || {
  echo "ERROR: IMAGE_ID digest differs from IMAGE_REFERENCE" >&2; exit 3;
}
for path in "${RUNNER}" "${BASE_RUNNER}" "${WRAPPER}" "${CERTIFIER}" \
            "${V1_CERTIFIER}" "${PRELAUNCH_MANIFEST}" "${JOB_TEMPLATE}"; do
  test -s "${path}" || { echo "ERROR: missing source ${path}" >&2; exit 4; }
done
test -s "${DATA_DIR}/manifest.json"
test "$(find "${DATA_DIR}" -maxdepth 1 -name '*.csv' -type f | wc -l)" -eq 23
test "$(sha256sum "${CHECKPOINT}" | cut -d' ' -f1)" = "${CHECKPOINT_SHA}" || {
  echo "ERROR: final Phase4 checkpoint hash mismatch" >&2; exit 5;
}

CERTIFIABLE_RUNNER_SHA256=$(sha256sum "${RUNNER}" | cut -d' ' -f1)
BASE_RUNNER_SHA256=$(sha256sum "${BASE_RUNNER}" | cut -d' ' -f1)
WRAPPER_SHA256=$(sha256sum "${WRAPPER}" | cut -d' ' -f1)
CERTIFIER_SHA256=$(sha256sum "${CERTIFIER}" | cut -d' ' -f1)
V1_CERTIFIER_SHA256=$(sha256sum "${V1_CERTIFIER}" | cut -d' ' -f1)
SOURCE_BUNDLE_SHA256=$(python - "${ROOT}" "${CODE_ROOT}" <<'PY'
import hashlib, pathlib, sys
root, code = map(pathlib.Path, sys.argv[1:])
entries = []
for name in ("DeBERTa", "mole_public"):
    directory = code / name
    entries.extend((f"{name}/{path.relative_to(directory).as_posix()}", path)
                   for path in directory.rglob("*")
                   if path.is_file()
                   and not {".git", "__pycache__"}.intersection(path.parts)
                   and path.suffix != ".pyc")
entries.extend((path.name, path) for path in
               (root / "encoder_arch.py", root / "moljepa_benchmark_config.py"))
digest = hashlib.sha256()
for relative, path in sorted(entries):
    digest.update(relative.encode("utf-8") + b"\0")
    digest.update(hashlib.sha256(path.read_bytes()).digest())
print(digest.hexdigest())
PY
)
PREPARED_MANIFEST_SHA256=$(sha256sum "${DATA_DIR}/manifest.json" | cut -d' ' -f1)
REFERENCE_RESULTS_SHA256=$(python - "${REFERENCE_DIR}" <<'PY'
import hashlib, pathlib, sys
directory = pathlib.Path(sys.argv[1])
expected = {
    "results_expansion_caco2_pappa.json", "results_expansion_caco2_efflux.json",
    "results_expansion_logd.json", "results_expansion_ksol.json",
    "results_expansion_hlm.json", "results_expansion_mlm.json",
    "results_expansion_mbpb.json", "results_expansion_mgmb.json",
    "results_expansion_mppb.json", "results_asap_mers.json",
    "results_asap_sars.json", "results_asap_logd.json", "results_asap_ksol.json",
    "results_asap_hlm.json", "results_asap_mlm.json", "results_asap_mdr1.json",
    "results_pxr.json", "results_biogen_solubility.json", "results_biogen_hlm.json",
    "results_biogen_rlm.json", "results_biogen_hppb.json",
    "results_biogen_rppb.json", "results_biogen_mdr1.json",
}
observed = {path.name for path in directory.glob("results_*.json") if path.is_file()}
if observed != expected:
    raise RuntimeError("canonical reference result inventory differs from the expected 23 files")
digest = hashlib.sha256()
for name in sorted(expected):
    path = directory / name
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(hashlib.sha256(path.read_bytes()).digest())
print(digest.hexdigest())
PY
)
JOB_TEMPLATE_SHA256=$(sha256sum "${JOB_TEMPLATE}" | cut -d' ' -f1)
test "$(sha256sum "${PRELAUNCH_MANIFEST}" | cut -d' ' -f1)" = "${PRELAUNCH_MANIFEST_SHA256}" || {
  echo "ERROR: independently pinned prelaunch manifest hash mismatch" >&2; exit 5;
}
export RUN_ID JOB_UID JOB_NAME POD_UID POD_NAME POD_NAMESPACE IMAGE_REFERENCE IMAGE_ID SHARD_ID
export CERTIFIABLE_RUNNER_SHA256 BASE_RUNNER_SHA256 WRAPPER_SHA256 CERTIFIER_SHA256
export V1_CERTIFIER_SHA256 SOURCE_BUNDLE_SHA256 REFERENCE_RESULTS_SHA256
export CHECKPOINT_SHA NUM_SHARDS PRELAUNCH_MANIFEST_SHA256
export PRELAUNCH_MANIFEST PREPARED_MANIFEST_SHA256 JOB_TEMPLATE_SHA256
python - <<'PY'
import json, os, pathlib
manifest = json.loads(pathlib.Path(os.environ["PRELAUNCH_MANIFEST"]).read_text(encoding="utf-8"))
expected = {
    "schema_version": 1,
    "run_id": os.environ["RUN_ID"],
    "checkpoint_sha256": os.environ["CHECKPOINT_SHA"],
    "prepared_manifest_sha256": os.environ["PREPARED_MANIFEST_SHA256"],
    "image_reference": os.environ["IMAGE_REFERENCE"],
    "image_id": os.environ["IMAGE_ID"],
    "source_bundle_sha256": os.environ["SOURCE_BUNDLE_SHA256"],
    "reference_results_sha256": os.environ["REFERENCE_RESULTS_SHA256"],
    "job_template_sha256": os.environ["JOB_TEMPLATE_SHA256"],
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise RuntimeError(f"prelaunch deployment manifest differs at {key}")
source_hashes = {
    "certifiable_runner_sha256": os.environ["CERTIFIABLE_RUNNER_SHA256"],
    "base_runner_sha256": os.environ["BASE_RUNNER_SHA256"],
    "wrapper_sha256": os.environ["WRAPPER_SHA256"],
    "certifier_sha256": os.environ["CERTIFIER_SHA256"],
    "v1_certifier_sha256": os.environ["V1_CERTIFIER_SHA256"],
}
if manifest.get("source_hashes") != source_hashes:
    raise RuntimeError("prelaunch deployment manifest source hashes differ")
required_runtime = {
    "python", "python_implementation", "rdkit_distribution", "rdkit_module",
    "numpy", "pandas", "scipy", "torch", "torch_geometric", "scikit_learn",
    "cuda_runtime",
}
if set(manifest.get("expected_runtime", {})) != required_runtime:
    raise RuntimeError("prelaunch deployment manifest runtime contract is incomplete")
if any(str(value).startswith("REPLACE_WITH_")
       for value in manifest["expected_runtime"].values()):
    raise RuntimeError("prelaunch deployment manifest still contains a runtime placeholder")
if manifest.get("posthoc_runtime_required") != ["cudnn"]:
    raise RuntimeError("prelaunch deployment manifest post-hoc runtime contract differs")
PY

mkdir -p "${OUTPUT_PARENT}"
OUTPUT_DIR=${OUTPUT_PARENT}/${RUN_ID}
INIT_LOCK=${OUTPUT_PARENT}/.${RUN_ID}.init.lock
export OUTPUT_DIR
(
  flock -x 9
  python - <<'PY'
import json, os, pathlib, re, tempfile
from datetime import datetime, timezone

root = pathlib.Path(os.environ["OUTPUT_DIR"])
receipt_path = root / "RUN_RECEIPT.json"
contract = {
    "schema_version": 2,
    "run_id": os.environ["RUN_ID"],
    "job_uid": os.environ["JOB_UID"],
    "job_name": os.environ["JOB_NAME"],
    "image_reference": os.environ["IMAGE_REFERENCE"],
    "image_id": os.environ["IMAGE_ID"],
    "expected_shards": int(os.environ["NUM_SHARDS"]),
    "expected_endpoints": 23,
    "expected_splits": 69,
    "expected_permutations": 690,
    "checkpoint_sha256": os.environ["CHECKPOINT_SHA"],
    "prelaunch_manifest_sha256": os.environ["PRELAUNCH_MANIFEST_SHA256"],
    "prepared_manifest_sha256": os.environ["PREPARED_MANIFEST_SHA256"],
    "reference_results_sha256": os.environ["REFERENCE_RESULTS_SHA256"],
    "source_hashes": {
        "certifiable_runner_sha256": os.environ["CERTIFIABLE_RUNNER_SHA256"],
        "base_runner_sha256": os.environ["BASE_RUNNER_SHA256"],
        "wrapper_sha256": os.environ["WRAPPER_SHA256"],
        "certifier_sha256": os.environ["CERTIFIER_SHA256"],
        "v1_certifier_sha256": os.environ["V1_CERTIFIER_SHA256"],
        "source_bundle_sha256": os.environ["SOURCE_BUNDLE_SHA256"],
    },
}
deployment = json.loads(pathlib.Path(os.environ["PRELAUNCH_MANIFEST"]).read_text(encoding="utf-8"))
contract["expected_runtime"] = deployment["expected_runtime"]
contract["posthoc_runtime_required"] = deployment["posthoc_runtime_required"]
if not root.exists():
    root.mkdir(mode=0o750)
    payload = dict(contract)
    payload["run_started_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload["initiating_pod"] = {
        "uid": os.environ["POD_UID"], "name": os.environ["POD_NAME"],
        "namespace": os.environ["POD_NAMESPACE"],
    }
    temporary = receipt_path.with_name("RUN_RECEIPT.json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, receipt_path)
else:
    if not receipt_path.is_file():
        raise RuntimeError("run root pre-exists without an immutable receipt")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    for key, value in contract.items():
        if payload.get(key) != value:
            raise RuntimeError(f"pre-existing run receipt differs at {key}")
PY
) 9>"${INIT_LOCK}"

RUN_STARTED_AT_UTC=$(python -c 'import json,os,pathlib; print(json.loads((pathlib.Path(os.environ["OUTPUT_DIR"])/"RUN_RECEIPT.json").read_text())["run_started_at_utc"])')
RUN_RECEIPT_SHA256=$(sha256sum "${OUTPUT_DIR}/RUN_RECEIPT.json" | cut -d' ' -f1)
POD_STARTED_AT_UTC=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
export RUN_STARTED_AT_UTC RUN_RECEIPT_SHA256 POD_STARTED_AT_UTC OUTPUT_DIR

LAUNCH_MANIFEST=${OUTPUT_DIR}/launch_manifest_shard_${SHARD_ID}.json
test ! -e "${LAUNCH_MANIFEST}" || {
  echo "ERROR: shard launch manifest already exists; RUN_ID is not fresh" >&2; exit 6;
}
python - "${LAUNCH_MANIFEST}" <<'PY'
import json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 2,
    "run_id": os.environ["RUN_ID"],
    "run_started_at_utc": os.environ["RUN_STARTED_AT_UTC"],
    "pod_started_at_utc": os.environ["POD_STARTED_AT_UTC"],
    "job_uid": os.environ["JOB_UID"], "job_name": os.environ["JOB_NAME"],
    "pod_uid": os.environ["POD_UID"], "pod_name": os.environ["POD_NAME"],
    "pod_namespace": os.environ["POD_NAMESPACE"],
    "shard_id": int(os.environ["SHARD_ID"]), "num_shards": int(os.environ["NUM_SHARDS"]),
    "image_reference": os.environ["IMAGE_REFERENCE"],
    "image_id": os.environ["IMAGE_ID"],
    "run_receipt_sha256": os.environ["RUN_RECEIPT_SHA256"],
    "prelaunch_manifest_sha256": os.environ["PRELAUNCH_MANIFEST_SHA256"],
    "checkpoint_sha256": os.environ["CHECKPOINT_SHA"],
    "reference_results_sha256": os.environ["REFERENCE_RESULTS_SHA256"],
    "source_hashes": {
        "certifiable_runner_sha256": os.environ["CERTIFIABLE_RUNNER_SHA256"],
        "base_runner_sha256": os.environ["BASE_RUNNER_SHA256"],
        "wrapper_sha256": os.environ["WRAPPER_SHA256"],
        "certifier_sha256": os.environ["CERTIFIER_SHA256"],
        "v1_certifier_sha256": os.environ["V1_CERTIFIER_SHA256"],
        "source_bundle_sha256": os.environ["SOURCE_BUNDLE_SHA256"],
    },
}
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY

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
python - <<'PY'
import importlib.metadata, rdkit
assert importlib.metadata.version("rdkit") == "2025.3.6"
assert rdkit.__version__ == "2025.03.6"
PY
export PYTHONPATH=${ROOT}:${PYTHONPATH:-}
export MOLJEPA_DATA_DIR=${DATA_DIR} RTD_CKPT=${CHECKPOINT}
export METHODS=full_mix MIX_LAYERS=4,8,10,12
export EPOCHS=60 PATIENCE=12 BATCH_SIZE=64 NUM_WORKERS=4 USE_BF16=1 FULL_LR=1e-5
export ATOM_ORDER_PERMUTATIONS=10

TASKS=(
  expansion_caco2_pappa expansion_caco2_efflux expansion_logd expansion_ksol
  expansion_hlm expansion_mlm expansion_mbpb expansion_mgmb expansion_mppb
  asap_mers asap_sars asap_logd asap_ksol asap_hlm asap_mlm asap_mdr1 pxr
  biogen_solubility biogen_hlm biogen_rlm biogen_hppb biogen_rppb biogen_mdr1
)
SHARD_TASKS=()
for ((index=SHARD_ID; index<${#TASKS[@]}; index+=NUM_SHARDS)); do
  task=${TASKS[$index]}
  SHARD_TASKS+=("${task}")
  echo "CERTIFIABLE-V2 run=${RUN_ID} task=${task} shard=${SHARD_ID}/${NUM_SHARDS}"
  TASK_NAME=${task} python "${RUNNER}" 2>&1 | tee -a "${OUTPUT_DIR}/log_${task}.txt"
done

MARKER=${OUTPUT_DIR}/SHARD_${SHARD_ID}_COMPLETE.json
export MARKER SHARD_TASKS_TEXT="${SHARD_TASKS[*]}"
python - <<'PY'
import hashlib, json, os, pathlib
from datetime import datetime, timezone
root = pathlib.Path(os.environ["OUTPUT_DIR"])
tasks = os.environ["SHARD_TASKS_TEXT"].split()
result_hashes = {}
for task in tasks:
    path = root / f"results_{task}.json"
    if not path.is_file():
        raise RuntimeError(f"missing shard result {path.name}")
    result_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "schema_version": 2, "run_id": os.environ["RUN_ID"],
    "job_uid": os.environ["JOB_UID"], "job_name": os.environ["JOB_NAME"],
    "pod_uid": os.environ["POD_UID"], "pod_name": os.environ["POD_NAME"],
    "pod_namespace": os.environ["POD_NAMESPACE"], "shard_id": int(os.environ["SHARD_ID"]),
    "pod_started_at_utc": os.environ["POD_STARTED_AT_UTC"],
    "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "run_receipt_sha256": os.environ["RUN_RECEIPT_SHA256"],
    "prelaunch_manifest_sha256": os.environ["PRELAUNCH_MANIFEST_SHA256"],
    "reference_results_sha256": os.environ["REFERENCE_RESULTS_SHA256"],
    "launch_manifest_sha256": hashlib.sha256(
        (root / f"launch_manifest_shard_{os.environ['SHARD_ID']}.json").read_bytes()
    ).hexdigest(),
    "tasks": tasks, "result_sha256": result_hashes,
}
path = pathlib.Path(os.environ["MARKER"])
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY

(
  flock -x 8
  complete=$(find "${OUTPUT_DIR}" -maxdepth 1 -name 'SHARD_*_COMPLETE.json' -type f | wc -l)
  if [[ "${complete}" -eq "${NUM_SHARDS}" ]]; then
    python "${CERTIFIER}" \
      --results-dir "${OUTPUT_DIR}" \
      --prepared-dir "${DATA_DIR}" \
      --reference-dir "${REFERENCE_DIR}" \
      --output-dir "${OUTPUT_DIR}/certified_atom_order_evidence" \
      --runner-source "${RUNNER}" \
      --base-runner-source "${BASE_RUNNER}" \
      --wrapper-source "${WRAPPER}" \
      --certifier-source "${CERTIFIER}" \
      --v1-certifier-source "${V1_CERTIFIER}" \
      --prelaunch-manifest "${PRELAUNCH_MANIFEST}" \
      --job-template-source "${JOB_TEMPLATE}"
    export CERTIFIED_OUTPUT=${OUTPUT_DIR}/certified_atom_order_evidence
    python - <<'PY'
import hashlib, json, os, pathlib
from datetime import datetime, timezone
root = pathlib.Path(os.environ["OUTPUT_DIR"])
certified = pathlib.Path(os.environ["CERTIFIED_OUTPUT"])
files = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(certified.iterdir()) if path.is_file()
}
payload = {
    "schema_version": 2, "run_id": os.environ["RUN_ID"],
    "job_uid": os.environ["JOB_UID"],
    "completed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "run_receipt_sha256": os.environ["RUN_RECEIPT_SHA256"],
    "prelaunch_manifest_sha256": os.environ["PRELAUNCH_MANIFEST_SHA256"],
    "certified_output_sha256": files,
}
path = root / "COMPLETE.json"
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
  fi
) 8>"${OUTPUT_DIR}/.finalize.lock"

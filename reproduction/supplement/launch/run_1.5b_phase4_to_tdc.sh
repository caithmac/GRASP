#!/bin/bash
# Resumable end-to-end pipeline:
# p3 RTD weights -> exposure-correct Step-1 continuation -> 50K Step-2 -> TDC.
set -euo pipefail

ROOT=/workspace/chemrasayan
CODE_ROOT=${ROOT}/code
DATA=/mnt/data/zinc20_1.5b
SOURCE=/mnt/checkpoints/rtd_25pct_1.5b/mole_rtd_25pct_1.5b_p3.ckpt
SOURCE_SHA=0b6f0067113853fad79ece1bbfc75e9d391081ff30fa9ca1ffb777c840de2b15
PHASE4=/mnt/checkpoints/rtd_25pct_1.5b_phase4
PLAN=${PHASE4}/phase4_plan.json
FINAL=${PHASE4}/phase4_final_weights.ckpt
S2_OUTPUT=/mnt/checkpoints/sup_pretrain_rtd25_1.5b_phase4
RESULTS=/mnt/results_1.5b_phase4_s2_50k_tdc
ARTIFACT=/mnt/artifacts/1.5b_phase4_s2_50k_tdc_results.tar.gz
PRETRAIN_LOG=/mnt/logs/pretrain_rtd25_1.5b_phase4.log
STEP2_LOG=/mnt/logs/sup_pretrain_rtd25_1.5b_phase4.log

mkdir -p "${PHASE4}" "${S2_OUTPUT}" "${RESULTS}" /mnt/logs /mnt/artifacts
test -s "${SOURCE}" || { echo "ERROR: missing ${SOURCE}" >&2; exit 2; }
ACTUAL_SHA=$(sha256sum "${SOURCE}" | cut -d' ' -f1)
test "${ACTUAL_SHA}" = "${SOURCE_SHA}" || {
  echo "ERROR: p3 source hash mismatch: ${ACTUAL_SHA}" >&2; exit 3;
}
test -s "${DATA}/val.parquet" || { echo "ERROR: missing fixed val.parquet" >&2; exit 4; }

FREE_KB=$(df -Pk /mnt | awk 'NR==2 {print $4}')
MIN_FREE_KB=$((12 * 1024 * 1024))
df -h /mnt
test "${FREE_KB}" -ge "${MIN_FREE_KB}" || {
  echo "ERROR: Phase 4 needs at least 12 GiB free on /mnt" >&2; exit 5;
}
test "$(nvidia-smi --list-gpus | wc -l)" -eq 4 || {
  echo "ERROR: this pipeline requires exactly four visible GPUs" >&2; exit 6;
}
nvidia-smi --query-gpu=index,name,memory.total --format=csv

export GIT_PYTHON_REFRESH=quiet
export PIP_ROOT_USER_ACTION=ignore
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
git config --global --add safe.directory '*'
test ! -e /tmp/DeBERTa && test ! -e /tmp/mole_public || {
  echo "ERROR: expected clean /tmp code destinations" >&2; exit 10;
}
cp -r "${CODE_ROOT}/DeBERTa" /tmp/DeBERTa
cp -r "${CODE_ROOT}/mole_public" /tmp/mole_public
echo "torch==$(python -c 'import torch; print(torch.__version__)')" > /tmp/constraints.txt
python -c 'import torchvision; print("torchvision==" + torchvision.__version__)' >> /tmp/constraints.txt
pip install --quiet -c /tmp/constraints.txt /tmp/DeBERTa /tmp/mole_public
pip install --quiet pyarrow PyTDC scikit-learn scipy wandb

python "${ROOT}/plan_rtd_phase4.py" \
  --data-dir "${DATA}" \
  --prior-examples 736000000 \
  --effective-batch 256 \
  --target-passes 1.0 \
  --output "${PLAN}"
MAX_STEPS=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["additional_optimizer_steps"])' "${PLAN}")
test "${MAX_STEPS}" -gt 0 || { echo "ERROR: continuation plan has no work" >&2; exit 7; }

sha256sum \
  "${SOURCE}" \
  "${ROOT}/encoder_arch.py" \
  "${ROOT}/plan_rtd_phase4.py" \
  "${ROOT}/run_1.5b_phase4_to_tdc.sh" \
  "${ROOT}/supervised_pretrain_ddp.py" \
  "${ROOT}/finetune_benchmarks.py" \
  "${ROOT}/summarize_tdc_run.py" \
  > "${PHASE4}/pipeline_code_sha256.txt"

RESUME=${PHASE4}/resume/last.ckpt
COMPLETION=${PHASE4}/resume/completed.ckpt
rm -f "${COMPLETION}"
TRAIN_ARGS=(
  model=pretrain_rtd_25pct_phase4
  '~logger.project=mole'
  '~logger.log_model=true'
  model.hyperparameters.datamodule.data="${DATA}"
  model.hyperparameters.datamodule.batch_size=32
  model.hyperparameters.datamodule.num_workers=4
  model.data.trainer.devices=4
  model.data.trainer.accumulate_grad_batches=2
  model.data.trainer.max_steps="${MAX_STEPS}"
  model.hyperparameters.pl_module.lr_scheduler.num_training_steps="${MAX_STEPS}"
  model.data.trainer.callbacks.0.dirpath="${PHASE4}/resume"
  model.data.trainer.callbacks.1.dirpath="${PHASE4}/milestones"
  model.data.trainer.callbacks.2.dirpath="${PHASE4}/best"
  +completion_ckpt_path="${COMPLETION}"
)
if [ -s "${RESUME}" ]; then
  echo "=== STEP 1 PHASE 4: full-state resume ==="
  python - "${RESUME}" "${MAX_STEPS}" <<'PY'
import sys, torch
p, maximum = sys.argv[1], int(sys.argv[2])
x = torch.load(p, map_location="cpu", weights_only=False)
if not x.get("optimizer_states") or not x.get("lr_schedulers"):
    raise SystemExit("ERROR: rolling checkpoint lacks optimizer/scheduler state")
step = int(x.get("global_step", -1))
if not 0 <= step <= maximum:
    raise SystemExit(f"ERROR: invalid resume step {step}/{maximum}")
print(f"Validated full resume state at phase4 step {step}/{maximum}")
PY
  TRAIN_ARGS+=(checkpoint_path=null resume_ckpt_path="${RESUME}")
else
  echo "=== STEP 1 PHASE 4: initialize from verified p3 weights ==="
  TRAIN_ARGS+=(checkpoint_path="${SOURCE}" resume_ckpt_path=null)
fi

torchrun --standalone --nproc_per_node=4 -m mole.cli.mole_train "${TRAIN_ARGS[@]}" \
  2>&1 | tee -a "${PRETRAIN_LOG}"
test -s "${COMPLETION}" || { echo "ERROR: missing exact completion checkpoint" >&2; exit 11; }

# Export a compact, immutable final weights file from the rolling full state.
python - "${COMPLETION}" "${MAX_STEPS}" "${FINAL}" "${PLAN}" <<'PY'
import os, sys, torch
source, expected, output, plan = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
x = torch.load(source, map_location="cpu", weights_only=False)
step = int(x.get("global_step", -1))
if step != expected:
    raise SystemExit(f"ERROR: Phase 4 ended at {step}, expected {expected}")
tmp = output + ".tmp"
torch.save({"state_dict": x["state_dict"], "global_step": step, "phase4_plan": plan}, tmp)
os.replace(tmp, output)
print(f"Exported final Phase-4 weights: {output} ({step} continuation steps)")
PY
rm -f "${COMPLETION}"
sha256sum "${FINAL}" | tee "${PHASE4}/final_sha256.txt"

# Repeat the already matched Step-2 recipe.  The 50K checkpoint is fixed in
# advance from the earlier p3 experiment; TDC test scores do not choose it.
export RTD_CKPT=${FINAL}
export ENCODER_ARCH=rtd25_step1
export STRICT_ENCODER_LOAD=1
export DATA_DIR=/mnt/data/chembl_supervised_paperlike
export OUTPUT_DIR=${S2_OUTPUT}
export MAX_STEPS=80000
export WARMUP_STEPS=10000
export SAVE_EVERY=10000
export BATCH_SIZE=32
export GRAD_ACCUM=4
export NUM_WORKERS=4
export LR=5e-6
export DROPOUT=0.1

echo "=== STEP 2: matched 4-GPU DDP ==="
torchrun --standalone --nproc_per_node=4 "${ROOT}/supervised_pretrain_ddp.py" \
  2>&1 | tee -a "${STEP2_LOG}"

FIXED_SNAPSHOT=${S2_OUTPUT}/encoder_step_050000.pt
test -s "${FIXED_SNAPSHOT}" || { echo "ERROR: missing fixed 50K Step-2 snapshot" >&2; exit 8; }
sha256sum "${FIXED_SNAPSHOT}" | tee "${RESULTS}/benchmark_checkpoint_sha256.txt"
cp "${PLAN}" "${RESULTS}/"
cp "${PHASE4}/final_sha256.txt" "${RESULTS}/"
cp "${PHASE4}/pipeline_code_sha256.txt" "${RESULTS}/"
cp "${S2_OUTPUT}/training_manifest.json" "${RESULTS}/step2_training_manifest.json"

TASKS=(
  bbb_martins hia_hou lipophilicity_astrazeneca herg ames dili
  bioavailability_ma caco2_wang pgp_broccatelli solubility_aqsoldb ppbr_az
  vdss_lombardo cyp2c9_veith cyp2d6_veith cyp3a4_veith
  cyp2c9_substrate_carbonmangels cyp2d6_substrate_carbonmangels
  cyp3a4_substrate_carbonmangels half_life_obach clearance_microsome_az
  clearance_hepatocyte_az ld50_zhu
)
export RTD_CKPT=${FIXED_SNAPSHOT}
export TDC_DATA_DIR=/mnt/tdc_data
export N_SEEDS=3
export SKIP_RANDOM=1
export EPOCHS=20
export BATCH_SIZE=64
export LR=1e-4
export NUM_WORKERS=4
export OMP_NUM_THREADS=4
export POOL_MODE=attn
export WANDB_MODE=disabled

benchmark_worker() {
  local gpu=$1 task result log rc failures=0
  for i in "${!TASKS[@]}"; do
    [ $((i % 4)) -eq "${gpu}" ] || continue
    task=${TASKS[$i]}
    result=${RESULTS}/results_${task}.json
    log=${RESULTS}/finetune_${task}.log
    if [ -s "${result}" ]; then
      echo "GPU ${gpu}: SKIP ${task}"
      continue
    fi
    echo "GPU ${gpu}: RUN ${task}"
    set +e
    CUDA_VISIBLE_DEVICES=${gpu} TASK_NAME=${task} RESULTS_PATH=${result} \
      python "${ROOT}/finetune_benchmarks.py" 2>&1 | tee "${log}"
    rc=${PIPESTATUS[0]}
    set -e
    if [ "${rc}" -ne 0 ]; then
      echo "FAILED rc=${rc}: ${task}" | tee -a "${RESULTS}/failures.txt"
      failures=$((failures + 1))
    fi
  done
  return "${failures}"
}

echo "=== TDC: fixed 50K Step-2 checkpoint x 22 official tasks ==="
pids=()
for gpu in 0 1 2 3; do benchmark_worker "${gpu}" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
test "${failed}" -eq 0 || { echo "ERROR: one or more TDC tasks failed" >&2; exit 9; }

python "${ROOT}/summarize_tdc_run.py" --results-dir "${RESULTS}" \
  --expected-tasks 22 --exclude-task clintox
tail -n 5000 "${PRETRAIN_LOG}" > "${RESULTS}/phase4_training_tail.log"
tail -n 5000 "${STEP2_LOG}" > "${RESULTS}/step2_training_tail.log"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "${RESULTS}/COMPLETE"
tar -czf "${ARTIFACT}" -C /mnt "$(basename "${RESULTS}")"

echo "=== PIPELINE COMPLETE ==="
echo "Phase-4 final: ${FINAL}"
echo "Step-2 fixed:  ${FIXED_SNAPSHOT}"
echo "Results:       ${RESULTS}"
echo "Artifact:      ${ARTIFACT}"

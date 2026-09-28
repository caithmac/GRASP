#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-/workspace/chemrasayan}
TDC_DATA_DIR=${TDC_DATA_DIR:-/mnt/data/tdc/raw}
RESULTS=${RESULTS:-/mnt/results_tdc_classical_baselines}
REFERENCE_MANIFEST=${REFERENCE_MANIFEST:-${ROOT}/provenance/tdc_classical_split_reference.json}
RUNNER=${RUNNER:-${ROOT}/src/evaluation/tdc/tdc_classical_baselines.py}
FINALIZER=${FINALIZER:-${ROOT}/src/evaluation/tdc/finalize_tdc_classical_baselines.py}
MODE=${MODE:-task}
mkdir -p "${RESULTS}"

pip_retry() {
  local attempt
  for attempt in 1 2 3 4 5; do
    python -m pip install --quiet "$@" && return 0
    echo "pip attempt ${attempt}/5 failed" >&2
    sleep 15
  done
  return 1
}

export PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
# These versions reproduce the audited historical PyTDC 0.4.1 split/runtime.
pip_retry 'numpy==1.26.4' 'pandas==1.5.3' 'rdkit-pypi==2022.9.5' \
  'scikit-learn==1.5.2' 'scipy==1.15.3'

if [ "${MODE}" = finalize ]; then
  FINAL_ARGS=(
    --results-dir "${RESULTS}"
    --reference-manifest "${REFERENCE_MANIFEST}"
  )
  if [ -n "${COMPARATOR_DIR:-}" ]; then
    FINAL_ARGS+=(--comparator-dir "${COMPARATOR_DIR}")
  fi
  python "${FINALIZER}" "${FINAL_ARGS[@]}"
  exit 0
fi

TASKS=(bbb_martins clintox hia_hou lipophilicity_astrazeneca herg ames dili bioavailability_ma caco2_wang pgp_broccatelli solubility_aqsoldb ppbr_az vdss_lombardo cyp2c9_veith cyp2d6_veith cyp3a4_veith cyp2c9_substrate_carbonmangels cyp2d6_substrate_carbonmangels cyp3a4_substrate_carbonmangels half_life_obach clearance_microsome_az clearance_hepatocyte_az ld50_zhu)
INDEX=${JOB_COMPLETION_INDEX:?JOB_COMPLETION_INDEX is required}
TASK=${TASKS[${INDEX}]}
python "${RUNNER}" \
  --task "${TASK}" \
  --data-dir "${TDC_DATA_DIR}" \
  --output-dir "${RESULTS}" \
  --reference-manifest "${REFERENCE_MANIFEST}" \
  --n-estimators 512

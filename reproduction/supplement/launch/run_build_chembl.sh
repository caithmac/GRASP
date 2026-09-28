#!/bin/bash
set -euo pipefail

NFS_DIR=/workspace/chemrasayan
CODE_DIR=${NFS_DIR}/code

echo "=== Build ChEMBL supervised pretraining dataset ==="

export PIP_ROOT_USER_ACTION=ignore

pip install -q PyTDC scipy rdkit-pypi 2>/dev/null || pip install -q PyTDC scipy rdkit 2>/dev/null

export CHEMBL_DB_PATH=/mnt/data/chembl/chembl_36_sqlite/chembl_36.db
export OUTPUT_DIR=/mnt/data/chembl_supervised_paperlike
export TDC_DATA_DIR=/mnt/tdc_data
export MIN_ASSAY_MOLS=500
export CHEMBL_VERSION=36

# Clear stale dataset (642 assays) to force full rebuild
echo "Clearing stale chembl_supervised_paperlike..."
rm -rf /mnt/data/chembl_supervised_paperlike
mkdir -p /mnt/data/chembl_supervised_paperlike

python ${NFS_DIR}/build_chembl_activity.py

echo "=== Done ==="

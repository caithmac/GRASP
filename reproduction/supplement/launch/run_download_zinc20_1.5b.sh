#!/bin/bash
# Download ZINC20 1.54B SMILES from HuggingFace to NFS/PVC.
# Run on EIDF login node with nohup or in a CPU K8s pod.
set -euo pipefail

export PIP_ROOT_USER_ACTION=ignore
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "== Installing dependencies =="
pip install --quiet datasets pyarrow pandas tqdm

echo "== Starting download =="
# --max_mols 1000000 for testing, remove for full run
python /workspace/chemrasayan/download_zinc20_1.5b.py \
    --output_dir /mnt/data/zinc20_1.5b \
    2>&1 | tee /mnt/download_zinc20_1.5b.log

echo "== Download COMPLETE =="

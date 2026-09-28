#!/bin/bash
# MolE RTD-25% pre-training on ZINC20 1.54B — 4×H200, on-the-fly parquet streaming
# No pre-tokenization: StreamingParquetDataset tokenizes on-the-fly to save storage.
set -euo pipefail

CODE_DIR=/workspace/chemrasayan/code/mole_public
DATA_DIR=/mnt/data/zinc20_1.5b          # parquet shards (on-the-fly, no .pt needed)
LOG_DIR=/mnt/logs
CKPT_DIR=/mnt/checkpoints/rtd_25pct_1.5b
CONFIG_PATH=/tmp/mole_public/mole/training/configs/model/pretrain_rtd_25pct.yaml

mkdir -p "$LOG_DIR" "$CKPT_DIR"

export GIT_PYTHON_REFRESH=quiet
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
git config --global --add safe.directory '*'

export PIP_ROOT_USER_ACTION=ignore
export PIP_DISABLE_PIP_VERSION_CHECK=1

# ── Copy code to /tmp (writable) for pip wheel builds ───────────────────
cp -r /workspace/chemrasayan/code/DeBERTa /tmp/DeBERTa
cp -r "$CODE_DIR" /tmp/mole_public

# ── Pin torch to container's version ────────────────────────────────────
TORCH_VER=$(python -c "import torch; print(torch.__version__)")
echo "torch==${TORCH_VER}" > /tmp/constraints.txt
echo "torchvision==$(python -c 'import torchvision; print(torchvision.__version__)')" >> /tmp/constraints.txt

# ── Install DeBERTa + MolE ──────────────────────────────────────────────
pip install --quiet -c /tmp/constraints.txt /tmp/DeBERTa
pip install --quiet -c /tmp/constraints.txt /tmp/mole_public

# ── Launch 4-GPU pretraining ────────────────────────────────────────────
# Effective batch = batch_size * nproc = 64 * 4 = 256
# 3M steps = 1.54B molecules / 256 batch ≈ 1 full pass
# Checkpoint every 500k steps (7 files = ~2.6 GB — tight 50GB PVC budget)
# StreamingParquetDataset auto-detected when .parquet shards found (no .pt shards)
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  -m mole.cli.mole_train \
  model=pretrain_rtd_25pct \
  '~logger.project=mole' \
  '~logger.log_model=true' \
  model.hyperparameters.datamodule.data="$DATA_DIR" \
  model.hyperparameters.datamodule.batch_size=64 \
  model.hyperparameters.datamodule.num_workers=4 \
  +model.hyperparameters.datamodule.max_atoms=96 \
  model.data.trainer.strategy=ddp_find_unused_parameters_true \
  model.data.trainer.devices=4 \
  model.data.trainer.max_steps=3000000 \
  model.hyperparameters.pl_module.lr_scheduler.num_training_steps=3000000 \
  model.data.trainer.callbacks.0.every_n_train_steps=500000 \
  model.data.trainer.callbacks.0.dirpath="$CKPT_DIR" \
  model.data.trainer.callbacks.0.filename='mole_rtd_25pct_1.5b-{step}' \
  '~model.hyperparameters.datamodule.validation_data' \
  2>&1 | tee "$LOG_DIR/pretrain_rtd_25pct_1.5b.log"

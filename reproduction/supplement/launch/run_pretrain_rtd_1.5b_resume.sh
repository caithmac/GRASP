#!/bin/bash
# Resume 1.5B RTD-25% from step 2.5M → 3M (500K more steps)
# last.ckpt = global_step 2,500,000. Weights-only resume, LR restarts.
# 4×H200, on-the-fly parquet streaming.
set -euo pipefail

CODE_DIR=/workspace/chemrasayan/code/mole_public
DATA_DIR=/mnt/data/zinc20_1.5b
LOG_DIR=/mnt/logs
CKPT_DIR=/mnt/checkpoints/rtd_25pct_1.5b
RESUME_FROM="$CKPT_DIR/last.ckpt"
CONFIG_PATH=/tmp/mole_public/mole/training/configs/model/pretrain_rtd_25pct.yaml

mkdir -p "$LOG_DIR" "$CKPT_DIR"

export GIT_PYTHON_REFRESH=quiet
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
git config --global --add safe.directory '*'

export PIP_ROOT_USER_ACTION=ignore
export PIP_DISABLE_PIP_VERSION_CHECK=1

cp -r /workspace/chemrasayan/code/DeBERTa /tmp/DeBERTa
cp -r "$CODE_DIR" /tmp/mole_public

TORCH_VER=$(python -c "import torch; print(torch.__version__)")
echo "torch==${TORCH_VER}" > /tmp/constraints.txt
echo "torchvision==$(python -c 'import torchvision; print(torchvision.__version__)')" >> /tmp/constraints.txt

pip install --quiet -c /tmp/constraints.txt /tmp/DeBERTa
pip install --quiet -c /tmp/constraints.txt /tmp/mole_public

# Resume: load 2.5M weights, train 500K → 3M total
# checkpoint_path = weights only (no optimizer/LR state restored)
# Step counter resets to 0. Cosine LR: warmup → 0 over 500K.
torchrun \
  --nproc_per_node=3 \
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
  model.data.trainer.devices=3 \
  model.data.trainer.max_steps=500000 \
  model.hyperparameters.pl_module.lr_scheduler.num_training_steps=500000 \
  model.hyperparameters.pl_module.checkpoint_path="$RESUME_FROM" \
  model.data.trainer.callbacks.0.every_n_train_steps=250000 \
  model.data.trainer.callbacks.0.dirpath="$CKPT_DIR" \
  model.data.trainer.callbacks.0.filename='mole_rtd_25pct_1.5b_p3' \
  2>&1 | tee "$LOG_DIR/pretrain_rtd_25pct_1.5b_resume.log"

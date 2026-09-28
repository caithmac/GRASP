"""Supervised pretraining (MolE step 2): multi-task BCE on ChEMBL activity data.

Starts from ZINC 415M RTD encoder. Adds a multi-task classification head
(one binary output per ChEMBL assay). Trains 60k steps with NaN masking.
Saves only the encoder weights for downstream use.

Follows MolE paper hyperparameters:
  - LR 5e-6, linear warmup 10k steps, linear decay over remaining 50k steps
  - Batch 512, no weight decay, gradient clip 1.0

Env vars:
  RTD_CKPT       /mnt/checkpoints/415m/last.ckpt
  DATA_DIR       /mnt/data/chembl_supervised
  OUTPUT_DIR     /mnt/checkpoints/sup_pretrain
  MAX_STEPS      60000
  WARMUP_STEPS   10000
  BATCH_SIZE     512
  LR             5e-6
  DROPOUT        0.1
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch

from DeBERTa.deberta.config import ModelConfig
from mole.training.data.datasets import MolDataset
from mole.training.data.utils import open_dictionary
from mole.training.models.mole import AtomEnvEmbeddings
from encoder_arch import architecture_summary, load_encoder_state, resolve_encoder_config


RTD_CKPT     = os.environ.get("RTD_CKPT",     "/mnt/checkpoints/415m/last.ckpt")
DATA_DIR     = os.environ.get("DATA_DIR",      "/mnt/data/chembl_supervised")
OUTPUT_DIR   = os.environ.get("OUTPUT_DIR",    "/mnt/checkpoints/sup_pretrain")
MAX_STEPS    = int(os.environ.get("MAX_STEPS",    "60000"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "10000"))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE",   "512"))
GRAD_ACCUM   = int(os.environ.get("GRAD_ACCUM",   "1"))
LR           = float(os.environ.get("LR",          "5e-6"))
DROPOUT      = float(os.environ.get("DROPOUT",     "0.1"))
ENCODER_ARCH = os.environ.get("ENCODER_ARCH", "auto").lower()
STRICT_ENCODER_LOAD = os.environ.get("STRICT_ENCODER_LOAD", "0") == "1"


def resolve_vocab_path() -> str:
    p = os.environ.get("VOCAB_PATH")
    if p and Path(p).exists():
        return p
    import mole
    return str(
        Path(mole.__path__[0])
        / "training" / "data" / "vocabularies"
        / "vocabulary_207atomenvs_radius0_ZINC_guacamole.pkl"
    )


def load_rtd_encoder(ckpt_path: str):
    return load_encoder_state(ckpt_path)


class MultiTaskHead(nn.Module):
    def __init__(self, hidden_size: int, num_tasks: int, dropout: float = 0.1):
        super().__init__()
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_tasks)

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(cls_hidden))


class MolESupervised(nn.Module):
    def __init__(self, encoder_state, num_tasks: int, dropout: float = 0.1):
        super().__init__()
        encoder_config = resolve_encoder_config(encoder_state, ENCODER_ARCH)
        cfg = ModelConfig.from_dict(encoder_config)
        self.encoder = AtomEnvEmbeddings(cfg)
        miss, unexp = self.encoder.load_state_dict(encoder_state, strict=False)
        print(
            f"  encoder[{ENCODER_ARCH} -> {architecture_summary(encoder_config)}]: "
            f"missing={len(miss)} unexpected={len(unexp)}"
        )
        if miss:
            print(f"    missing keys: {list(miss)}")
        if unexp:
            print(f"    unexpected keys: {list(unexp)}")
        if STRICT_ENCODER_LOAD and (miss or unexp):
            raise RuntimeError(
                f"Strict encoder load failed: missing={len(miss)} unexpected={len(unexp)}"
            )
        self.head = MultiTaskHead(cfg.hidden_size, num_tasks, dropout)

    def forward(self, batch) -> torch.Tensor:
        input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        relative_pos = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)

        def run_encoder(input_ids, input_mask, relative_pos):
            out = self.encoder(
                input_ids, input_mask,
                attention_mask=input_mask,
                relative_pos=relative_pos,
            )
            return out["hidden_states"][-1]

        # Recompute encoder intermediates during backward instead of storing every layer activation.
        # use_reentrant=False required: inputs are integer tensors (no requires_grad);
        # reentrant mode silently skips gradient computation for encoder params, leaving the
        # encoder frozen at its init weights while only the head trains.
        last_hidden = grad_checkpoint(
            run_encoder, input_ids, input_mask, relative_pos, use_reentrant=False,
        )
        cls = last_hidden[:, 0, :]
        return self.head(cls)


def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """BCE loss ignoring NaN positions."""
    mask = ~torch.isnan(labels)
    if not mask.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask].float())


def lr_schedule(step: int) -> float:
    """Linear warmup then linear decay."""
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return max(0.0, 1.0 - progress)


def make_loader(smiles: list[str], labels: np.ndarray, dictionary) -> DataLoader:
    ds = MolDataset(
        smiles=pd.Series(smiles),
        dictionary_inp=dictionary,
        radius_inp=0,
        useFeatures_inp=False,
        cls_token=True,
        labels=labels,
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    print(f"RTD ckpt={RTD_CKPT}")
    print(f"Data dir={DATA_DIR}")
    print(f"Steps={MAX_STEPS} warmup={WARMUP_STEPS} batch={BATCH_SIZE} lr={LR}")

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1] Loading ChEMBL activity data...")
    data_dir = Path(DATA_DIR)
    smiles   = (data_dir / "smiles.txt").read_text().strip().splitlines()
    labels   = np.load(str(data_dir / "labels.npy"))        # [N, n_assays] float16
    assay_ids = (data_dir / "assay_ids.txt").read_text().strip().splitlines()
    n_assays  = len(assay_ids)
    print(f"  {len(smiles):,} molecules, {n_assays:,} assays")
    print(f"  Labels shape: {labels.shape}, sparsity: {np.isnan(labels).mean()*100:.1f}%")

    # ── Load encoder ──────────────────────────────────────────────────────
    print("\n[2] Loading ZINC 415M RTD encoder...")
    encoder_state = load_rtd_encoder(RTD_CKPT)

    # ── Vocab ─────────────────────────────────────────────────────────────
    print("\n[3] Loading vocabulary...")
    dictionary = open_dictionary(resolve_vocab_path())

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n[4] Building model...")
    model = MolESupervised(encoder_state, num_tasks=n_assays, dropout=DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {n_params:,}")

    # No weight decay as per MolE paper
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    # Snapshot encoder weights to verify they actually move during training
    init_encoder_snapshot = {
        k: v.detach().clone().cpu() for k, v in model.encoder.state_dict().items()
    }

    # ── Data loader ───────────────────────────────────────────────────────
    print("\n[5] Building data loader...")
    loader = make_loader(smiles, labels, dictionary)
    loader_iter = iter(loader)

    # ── Training loop ─────────────────────────────────────────────────────
    print("\n[6] Training...")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    log_every = 500

    model.train()
    running_loss = 0.0
    running_n    = 0
    optim.zero_grad()

    for step in range(1, MAX_STEPS + 1):
        accum_loss = 0.0
        for _ in range(GRAD_ACCUM):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            batch = batch.to(device)
            logits = model(batch)

            # PyG Batch concatenates target_labels along dim=0 → [B*n_assays]; reshape back
            targets = batch.target_labels.view(logits.shape[0], n_assays)

            loss = masked_bce_loss(logits, targets) / GRAD_ACCUM
            loss.backward()
            accum_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        enc_grad_norm = math.sqrt(sum(
            (p.grad.detach().float().norm() ** 2).item()
            for p in model.encoder.parameters() if p.grad is not None
        ))
        head_grad_norm = math.sqrt(sum(
            (p.grad.detach().float().norm() ** 2).item()
            for p in model.head.parameters() if p.grad is not None
        ))

        # LR schedule
        for g in optim.param_groups:
            g["lr"] = LR * lr_schedule(step)

        optim.step()
        optim.zero_grad()

        running_loss += accum_loss
        running_n    += 1

        if step % log_every == 0:
            avg_loss = running_loss / running_n
            current_lr = LR * lr_schedule(step)
            print(
                f"  step {step:>6}/{MAX_STEPS} "
                f"loss={avg_loss:.4f} "
                f"lr={current_lr:.2e} "
                f"enc_gnorm={enc_grad_norm:.3e} "
                f"head_gnorm={head_grad_norm:.3e}",
                flush=True,
            )
            running_loss = 0.0
            running_n    = 0

            if enc_grad_norm == 0.0:
                raise RuntimeError(
                    "Encoder gradient norm is zero — encoder is not training. "
                    "Check grad_checkpoint use_reentrant setting."
                )

    # ── Verify encoder actually moved during training ────────────────────
    print("\n[7] Verifying encoder weights drifted from init...")
    final_sd = model.encoder.state_dict()
    total_delta = 0.0
    total_norm  = 0.0
    max_delta_layer = ("", 0.0)
    for k, v_init in init_encoder_snapshot.items():
        v_final = final_sd[k].detach().cpu().float()
        delta = (v_final - v_init.float()).norm().item()
        total_delta += delta ** 2
        total_norm  += v_final.norm().item() ** 2
        if delta > max_delta_layer[1]:
            max_delta_layer = (k, delta)
    total_delta = math.sqrt(total_delta)
    total_norm  = math.sqrt(total_norm)
    rel = total_delta / max(total_norm, 1e-12)
    print(f"  ||Δ encoder|| = {total_delta:.4e}")
    print(f"  ||encoder||   = {total_norm:.4e}")
    print(f"  relative drift = {rel:.4e}")
    print(f"  max-drift layer: {max_delta_layer[0]} ({max_delta_layer[1]:.4e})")
    if rel < 1e-6:
        raise RuntimeError(
            f"Encoder did not drift from init (rel drift {rel:.2e}). "
            "Saved weights would be identical to ZINC RTD — aborting."
        )

    # ── Save encoder only ─────────────────────────────────────────────────
    print("\n[8] Saving encoder weights...")
    encoder_path = Path(OUTPUT_DIR) / "encoder_weights.pt"
    torch.save(model.encoder.state_dict(), str(encoder_path))
    print(f"  Encoder saved to {encoder_path}")
    print("\nSupervised pretraining complete.")


if __name__ == "__main__":
    main()

"""Distributed MolE Step-2 training with resumable 10k encoder snapshots.

Run with torchrun. The global effective batch is:
    BATCH_SIZE * WORLD_SIZE * GRAD_ACCUM

Only encoder snapshots are retained at every SAVE_EVERY interval. A single
rolling full training state is overwritten atomically so interrupted jobs can
resume without retaining eight copies of Adam state.
"""
from __future__ import annotations

import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from mole.training.data.datasets import MolDataset
from mole.training.data.utils import open_dictionary
from supervised_pretrain import (
    BATCH_SIZE,
    DATA_DIR,
    DROPOUT,
    GRAD_ACCUM,
    LR,
    MAX_STEPS,
    OUTPUT_DIR,
    RTD_CKPT,
    WARMUP_STEPS,
    MolESupervised,
    load_rtd_encoder,
    lr_schedule,
    masked_bce_loss,
    resolve_vocab_path,
)

SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "10000"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
SEED = int(os.environ.get("SEED", "42"))
RESUME_STATE = os.environ.get(
    "RESUME_STATE", str(Path(OUTPUT_DIR) / "training_state_latest.pt")
)


def rank0_print(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs, flush=True)


def atomic_torch_save(obj, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, destination)


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    random.seed(SEED + rank)
    np.random.seed(SEED + rank)
    torch.manual_seed(SEED + rank)
    torch.cuda.manual_seed_all(SEED + rank)

    output_dir = Path(OUTPUT_DIR)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    rank0_print("=== Distributed MolE Step 2 ===")
    rank0_print(f"source={RTD_CKPT}")
    rank0_print(f"encoder_arch={os.environ.get('ENCODER_ARCH', 'auto')}")
    rank0_print(f"world_size={world_size} local_batch={BATCH_SIZE} accum={GRAD_ACCUM}")
    rank0_print(f"global_effective_batch={BATCH_SIZE * world_size * GRAD_ACCUM}")
    rank0_print(
        f"steps={MAX_STEPS} warmup={WARMUP_STEPS} lr={LR} "
        f"snapshot_every={SAVE_EVERY}"
    )
    if BATCH_SIZE * world_size * GRAD_ACCUM != 512:
        raise RuntimeError("Global effective batch must remain 512 for a fair Step-2 rerun")

    data_dir = Path(DATA_DIR)
    smiles = (data_dir / "smiles.txt").read_text().strip().splitlines()
    labels = np.load(str(data_dir / "labels.npy"))
    assay_ids = (data_dir / "assay_ids.txt").read_text().strip().splitlines()
    n_assays = len(assay_ids)
    rank0_print(
        f"data={len(smiles):,} molecules, {n_assays:,} assays, "
        f"sparsity={np.isnan(labels).mean() * 100:.1f}%"
    )

    source_encoder = load_rtd_encoder(RTD_CKPT)
    dictionary = open_dictionary(resolve_vocab_path())
    model = MolESupervised(source_encoder, n_assays, DROPOUT).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_step = 0
    resume_path = Path(RESUME_STATE)
    if resume_path.is_file():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state.get("source_checkpoint") != RTD_CKPT:
            raise RuntimeError(
                f"Resume source changed: {state.get('source_checkpoint')} != {RTD_CKPT}"
            )
        if not 0 <= int(state["step"]) <= MAX_STEPS:
            raise RuntimeError(f"Invalid resume step {state['step']}/{MAX_STEPS}")
        expected_batch = BATCH_SIZE * world_size * GRAD_ACCUM
        if state.get("global_effective_batch") != expected_batch:
            raise RuntimeError(
                "Resume global batch changed: "
                f"{state.get('global_effective_batch')} != {expected_batch}"
            )
        model.module.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        rank0_print(f"resumed={resume_path} at optimizer step {start_step}")

    dataset = MolDataset(
        smiles=pd.Series(smiles),
        dictionary_inp=dictionary,
        radius_inp=0,
        useFeatures_inp=False,
        cls_token=True,
        labels=labels,
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
        pin_memory=True,
    )
    # Restore deterministic sampler progress. With the configured dataset and
    # save interval, snapshots land on epoch boundaries; the remainder logic
    # also handles future configurations where they do not.
    batches_consumed = start_step * GRAD_ACCUM
    data_epoch = batches_consumed // len(loader)
    batches_to_skip = batches_consumed % len(loader)
    sampler.set_epoch(data_epoch)
    loader_iter = iter(loader)
    for _ in range(batches_to_skip):
        next(loader_iter)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_n = 0

    for step in range(start_step + 1, MAX_STEPS + 1):
        accumulated = 0.0
        for micro_step in range(GRAD_ACCUM):
            try:
                batch = next(loader_iter)
            except StopIteration:
                data_epoch += 1
                sampler.set_epoch(data_epoch)
                loader_iter = iter(loader)
                batch = next(loader_iter)

            # Avoid four all-reduces per optimizer update. Only the final
            # microbatch synchronizes the accumulated gradients across ranks.
            sync_context = model.no_sync() if micro_step < GRAD_ACCUM - 1 else nullcontext()
            with sync_context:
                batch = batch.to(device, non_blocking=True)
                logits = model(batch)
                targets = batch.target_labels.view(logits.shape[0], n_assays)
                loss = masked_bce_loss(logits, targets) / GRAD_ACCUM
                loss.backward()
            accumulated += loss.detach().item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        encoder_grad_sq = sum(
            (p.grad.detach().float().norm() ** 2).item()
            for p in model.module.encoder.parameters()
            if p.grad is not None
        )
        encoder_grad_norm = math.sqrt(encoder_grad_sq)

        current_lr = LR * lr_schedule(step)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        loss_tensor = torch.tensor(accumulated, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        running_loss += loss_tensor.item() / world_size
        running_n += 1

        if step % 500 == 0 and rank == 0:
            print(
                f"step {step:>6}/{MAX_STEPS} "
                f"loss={running_loss / running_n:.4f} lr={current_lr:.2e} "
                f"grad_norm={float(grad_norm):.3e} "
                f"enc_gnorm={encoder_grad_norm:.3e}",
                flush=True,
            )
            if encoder_grad_norm == 0.0:
                raise RuntimeError("Encoder gradient norm is zero")
            running_loss = 0.0
            running_n = 0

        if step % SAVE_EVERY == 0 or step == MAX_STEPS:
            dist.barrier()
            if rank == 0:
                encoder_path = output_dir / f"encoder_step_{step:06d}.pt"
                atomic_torch_save(model.module.encoder.state_dict(), encoder_path)
                atomic_torch_save(
                    {
                        "step": step,
                        "model": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "source_checkpoint": RTD_CKPT,
                        "encoder_arch": os.environ.get("ENCODER_ARCH", "auto"),
                        "world_size": world_size,
                        "global_effective_batch": BATCH_SIZE * world_size * GRAD_ACCUM,
                    },
                    resume_path,
                )
                rank0_print(f"saved encoder snapshot: {encoder_path}")
            dist.barrier()

    if rank == 0:
        final_path = output_dir / f"encoder_step_{MAX_STEPS:06d}.pt"
        alias = output_dir / "encoder_weights.pt"
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        os.link(final_path, alias)  # hard link: conventional name, zero duplicate storage

        # Verify the final encoder differs from the 3M Step-1 source.
        final_state = model.module.encoder.state_dict()
        delta_sq = 0.0
        norm_sq = 0.0
        for key, source_value in source_encoder.items():
            final_value = final_state[key].detach().cpu().float()
            delta_sq += (final_value - source_value.float()).norm().item() ** 2
            norm_sq += final_value.norm().item() ** 2
        relative_drift = math.sqrt(delta_sq) / max(math.sqrt(norm_sq), 1e-12)
        rank0_print(f"final relative encoder drift={relative_drift:.6e}")
        if relative_drift < 1e-6:
            raise RuntimeError("Final encoder did not move from its Step-1 initialization")

        manifest = {
            "source_checkpoint": RTD_CKPT,
            "encoder_arch": os.environ.get("ENCODER_ARCH", "auto"),
            "max_steps": MAX_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "save_every": SAVE_EVERY,
            "world_size": world_size,
            "per_gpu_batch_size": BATCH_SIZE,
            "gradient_accumulation": GRAD_ACCUM,
            "global_effective_batch": BATCH_SIZE * world_size * GRAD_ACCUM,
            "learning_rate": LR,
            "dropout": DROPOUT,
            "molecules": len(smiles),
            "assays": n_assays,
            "relative_encoder_drift": relative_drift,
        }
        (output_dir / "training_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
